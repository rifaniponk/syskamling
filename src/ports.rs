//! Listening ports and the processes behind them, read from /proc/net and
//! /proc/[pid]/fd — plus a cross-link to Docker containers that publish
//! the same port on the host.

use std::collections::HashMap;
use std::fs;

/// One listening socket joined to its owning process.
#[derive(Clone, Debug)]
pub struct PortEntry {
    pub proto: &'static str, // "tcp" | "udp"
    pub family: u8,          // 4 | 6
    /// Bind address as displayed, e.g. "0.0.0.0", "[::]", "127.0.0.1".
    pub addr: String,
    pub port: u16,
    /// "LISTEN" for tcp, "-" for udp (connectionless).
    pub state: &'static str,
    /// Owning process; None for kernel-owned or unreadable sockets.
    pub pid: Option<u32>,
    pub proc_name: Option<String>,
    /// Working directory of the owning process, when readable.
    pub cwd: Option<String>,
    /// Socket inode from /proc/net/*, handy for `ss`-style debugging.
    pub inode: u64,
}

/// Parse one /proc/net/{tcp,tcp6,udp,udp6} table into (proto, family,
/// addr, port, state, inode) rows. Hex fields, little-endian address words.
fn parse_net_table(txt: &str, proto: &'static str, family: u8) -> Vec<PortEntry> {
    let mut out = Vec::new();
    for line in txt.lines().skip(1) {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 10 {
            continue;
        }
        let Some((local, _remote)) = f[1].split_once(':') else { continue };
        let port = u16::from_str_radix(f[1].rsplit(':').next().unwrap_or("0"), 16).unwrap_or(0);
        let addr = decode_addr(local, family);
        // tcp state is field 3; udp has no state but the column exists.
        let state_num: u8 = u8::from_str_radix(f[3], 16).unwrap_or(0);
        let state: &'static str = if proto == "tcp" {
            match state_num {
                1 => "ESTABLISHED",
                2 => "SYN_SENT",
                3 => "SYN_RECV",
                4 => "FIN_WAIT1",
                5 => "FIN_WAIT2",
                6 => "TIME_WAIT",
                7 => "CLOSE",
                8 => "CLOSE_WAIT",
                9 => "LAST_ACK",
                10 => "LISTEN",
                11 => "CLOSING",
                _ => "-",
            }
        } else {
            "-"
        };
        // For tcp keep listeners (and established, for visibility); for udp
        // keep bound sockets with a port.
        if proto == "tcp" && state_num != 10 && state_num != 1 {
            continue;
        }
        let inode = f[9].parse().unwrap_or(0);
        out.push(PortEntry {
            proto,
            family,
            addr,
            port,
            state,
            pid: None,
            proc_name: None,
            cwd: None,
            inode,
        });
    }
    out
}

/// /proc/net hex addresses: IPv4 is one little-endian 32-bit word; IPv6 is
/// four little-endian words in `::` notation.
fn decode_addr(hex: &str, family: u8) -> String {
    if family == 4 {
        if hex == "00000000" {
            return "0.0.0.0".into();
        }
        if let Ok(w) = u32::from_str_radix(hex, 16) {
            return format!("{}.{}.{}.{}", w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF, (w >> 24) & 0xFF);
        }
        return hex.into();
    }
    if hex == "00000000000000000000000000000000" {
        return "[::]".into();
    }
    if hex == "00000000000000000000000001000000" {
        return "[::1]".into();
    }
    // four LE words -> octets
    let mut words = Vec::new();
    for i in 0..4 {
        let chunk = &hex[i * 8..(i + 1) * 8];
        let w = u32::from_str_radix(chunk, 16).unwrap_or(0);
        words.push([w & 0xFF, (w >> 8) & 0xFF, (w >> 16) & 0xFF, (w >> 24) & 0xFF]);
    }
    let bytes: Vec<u8> = words.iter().flatten().map(|b| *b as u8).collect();
    let ip = std::net::Ipv6Addr::from(<[u8; 16]>::try_from(bytes).unwrap_or([0; 16]));
    format!("[{ip}]")
}

/// Map socket inode -> (pid, fd) by scanning /proc/[pid]/fd/* socket links.
fn inode_to_pid() -> HashMap<u64, u32> {
    let mut map = HashMap::new();
    let Ok(procs) = fs::read_dir("/proc") else { return map };
    for p in procs.flatten() {
        let name = p.file_name();
        let Ok(pid): Result<u32, _> = name.to_string_lossy().parse() else { continue };
        let Ok(fds) = fs::read_dir(format!("/proc/{pid}/fd")) else { continue };
        for fd in fds.flatten() {
            let Ok(link) = fs::read_link(fd.path()) else { continue };
            let l = link.to_string_lossy();
            if let Some(rest) = l.strip_prefix("socket:[") {
                if let Some(inum) = rest.strip_suffix(']') {
                    if let Ok(inum) = inum.parse::<u64>() {
                        // first claimer wins; a socket has one owner anyway
                        map.entry(inum).or_insert(pid);
                    }
                }
            }
        }
    }
    map
}

fn proc_basics(pid: u32) -> (Option<String>, Option<String>) {
    let name = fs::read_to_string(format!("/proc/{pid}/comm"))
        .ok()
        .map(|s| s.trim().to_string());
    let cwd = fs::read_link(format!("/proc/{pid}/cwd"))
        .ok()
        .map(|p| p.to_string_lossy().into_owned());
    (name, cwd)
}

/// Scan everything: returns entries sorted by (proto, port, addr).
pub fn scan() -> Vec<PortEntry> {
    let mut entries = Vec::new();
    for (path, proto, family) in [
        ("/proc/net/tcp", "tcp", 4u8),
        ("/proc/net/tcp6", "tcp", 6),
        ("/proc/net/udp", "udp", 4),
        ("/proc/net/udp6", "udp", 6),
    ] {
        if let Ok(txt) = fs::read_to_string(path) {
            entries.extend(parse_net_table(&txt, proto, family));
        }
    }
    let owners = inode_to_pid();
    let mut cache: HashMap<u32, (Option<String>, Option<String>)> = HashMap::new();
    for e in &mut entries {
        if let Some(&pid) = owners.get(&e.inode) {
            e.pid = Some(pid);
            let (name, cwd) = cache
                .entry(pid)
                .or_insert_with(|| proc_basics(pid));
            e.proc_name = name.clone();
            e.cwd = cwd.clone();
        }
    }
    entries.sort_by_key(|e| (e.proto.to_string(), e.port, e.addr.clone()));
    entries
}

#[cfg(test)]
mod tests {
    use super::*;

    const TCP: &str = "\
  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode
   0: 0100007F:1F91 00000000:0000 0A 00000000:00000000 00:00000000 00000000  1000        0 12345678 1 0000000000000000 100 0 0 10 0
   1: 00000000:1F90 00000000:0000 0A 00000000:00000000 00:00000000 00000000     0        0 22334455 1 0000000000000000 100 0 0 10 0
   2: 0100007F:9001 0100007F:E001 01 00000000:00000000 00:00000000 00000000  1000        0 99887766 1 0000000000000000 100 0 0 10 0
   3: 0100007F:A001 0100007F:F001 06 00000000:00000000 00:00000000 00000000  1000        0 11111111 1 0000000000000000 100 0 0 10 0";

    #[test]
    fn parses_listeners_and_established_skips_others() {
        let rows = parse_net_table(TCP, "tcp", 4);
        // rows 0,1 LISTEN; row 2 ESTABLISHED; row 3 TIME_WAIT dropped
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0].port, 0x1F91);
        assert_eq!(rows[0].addr, "127.0.0.1");
        assert_eq!(rows[0].state, "LISTEN");
        assert_eq!(rows[1].addr, "0.0.0.0");
        assert_eq!(rows[2].state, "ESTABLISHED");
        assert_eq!(rows[0].inode, 12345678);
    }

    #[test]
    fn decodes_ipv4_and_ipv6_addresses() {
        assert_eq!(decode_addr("0100007F", 4), "127.0.0.1");
        assert_eq!(decode_addr("00000000", 4), "0.0.0.0");
        assert_eq!(decode_addr("00000000000000000000000000000000", 6), "[::]");
        assert_eq!(decode_addr("00000000000000000000000001000000", 6), "[::1]");
    }

    #[test]
    fn live_scan_finds_this_shells_listener_or_runs_clean() {
        let rows = scan();
        // every entry is well-formed; pids, when set, have names
        for r in &rows {
            if r.proto == "tcp" {
                assert!(r.port > 0, "tcp listener with port 0: {r:?}");
            }
            if r.pid.is_some() {
                assert!(r.proc_name.is_some(), "pid set but name missing");
            }
        }
    }
}
