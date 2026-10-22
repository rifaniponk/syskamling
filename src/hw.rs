//! Host hardware vitals read from /proc and /sys — the Task-Manager side
//! of SysKamling. A Sampler carries the previous counter snapshot so each
//! scan returns usage *rates* (CPU %, bytes/s) instead of raw totals.

use std::fs;
use std::time::Instant;

/// One snapshot of host vitals, all rates computed against the prior scan.
#[derive(Clone, Debug, Default)]
pub struct HwSample {
    /// Aggregate CPU busy percent across all cores (0..=100*ncpu scaled to 0..100).
    pub cpu_pct: f64,
    /// Per-core busy percent.
    pub per_core: Vec<f64>,
    pub load1: f64,
    pub load5: f64,
    pub load15: f64,
    pub ncpu: usize,
    // memory, bytes
    pub mem_total: u64,
    pub mem_avail: u64,
    pub swap_total: u64,
    pub swap_free: u64,
    // disk I/O since last scan, bytes/s (summed over physical devices)
    pub disk_read_bps: f64,
    pub disk_write_bps: f64,
    // network since last scan, bytes/s (summed over non-loop interfaces)
    pub net_rx_bps: f64,
    pub net_tx_bps: f64,
    /// Hottest coretemp/mtemp sensor reading, Celsius, if any.
    pub temp_c: Option<f64>,
    /// Uptime seconds.
    pub uptime: f64,
}

/// Per-device counters kept between scans for rate computation.
struct Counters {
    at: Instant,
    /// aggregate: [user, nice, system, idle, iowait, irq, softirq, steal]
    cpu_all: [u64; 8],
    cpu_cores: Vec<[u64; 8]>,
    disk: Vec<(String, u64, u64)>, // name, sectors read, sectors written
    net: Vec<(String, u64, u64)>,  // iface, rx bytes, tx bytes
}

#[derive(Default)]
pub struct Sampler {
    prev: Option<Counters>,
}

impl Sampler {
    /// Read /proc & /sys once; rates come from the delta vs the previous call.
    pub fn sample(&mut self) -> HwSample {
        let mut s = HwSample::default();
        s.uptime = read_f64("/proc/uptime", 0);

        // ---- CPU ----------------------------------------------------------
        let (all, cores) = read_cpu_times();
        s.ncpu = cores.len().max(1);
        let now = Instant::now();
        if let Some(p) = &self.prev {
            let dt = now.duration_since(p.at).as_secs_f64().max(1e-3);
            s.cpu_pct = busy_pct(&p.cpu_all, &all);
            s.per_core = p
                .cpu_cores
                .iter()
                .zip(&cores)
                .map(|(a, b)| busy_pct(a, b))
                .collect();
            // disk: fields 1..=sectors read/written (512 B sectors)
            let mut dr = 0u64;
            let mut dw = 0u64;
            for (name, r, w) in &s_diskstats() {
                if let Some((_, pr, pw)) = p.disk.iter().find(|(n, _, _)| n == name) {
                    dr += r.saturating_sub(*pr);
                    dw += w.saturating_sub(*pw);
                }
            }
            s.disk_read_bps = dr as f64 * 512.0 / dt;
            s.disk_write_bps = dw as f64 * 512.0 / dt;
            let mut rx = 0u64;
            let mut tx = 0u64;
            for (name, r, t) in &s_netdev() {
                if let Some((_, pr, pt)) = p.net.iter().find(|(n, _, _)| n == name) {
                    rx += r.saturating_sub(*pr);
                    tx += t.saturating_sub(*pt);
                }
            }
            s.net_rx_bps = rx as f64 / dt;
            s.net_tx_bps = tx as f64 / dt;
        }
        self.prev = Some(Counters {
            at: now,
            cpu_all: all,
            cpu_cores: cores.clone(),
            disk: s_diskstats(),
            net: s_netdev(),
        });

        // ---- load average --------------------------------------------------
        if let Ok(u) = fs::read_to_string("/proc/loadavg") {
            let mut it = u.split_whitespace();
            s.load1 = it.next().and_then(|w| w.parse().ok()).unwrap_or(0.0);
            s.load5 = it.next().and_then(|w| w.parse().ok()).unwrap_or(0.0);
            s.load15 = it.next().and_then(|w| w.parse().ok()).unwrap_or(0.0);
        }

        // ---- memory --------------------------------------------------------
        let mi = fs::read_to_string("/proc/meminfo").unwrap_or_default();
        for line in mi.lines() {
            let mut it = line.split_whitespace();
            let (Some(key), Some(val)) = (it.next(), it.next()) else { continue };
            let kb: u64 = val.parse().unwrap_or(0);
            match key {
                "MemTotal:" => s.mem_total = kb * 1024,
                "MemAvailable:" => s.mem_avail = kb * 1024,
                "SwapTotal:" => s.swap_total = kb * 1024,
                "SwapFree:" => s.swap_free = kb * 1024,
                _ => {}
            }
        }

        // ---- temperature (millidegrees from coretemp/hwmon) ----------------
        s.temp_c = hottest_sensor();
        s
    }
}

fn read_f64(path: &str, field: usize) -> f64 {
    fs::read_to_string(path)
        .ok()
        .and_then(|s| s.split_whitespace().nth(field).and_then(|w| w.parse().ok()))
        .unwrap_or(0.0)
}

/// Parse /proc/stat: aggregate "cpu " line and every "cpuN" line into
/// busy-component tuples.
fn read_cpu_times() -> ([u64; 8], Vec<[u64; 8]>) {
    let txt = fs::read_to_string("/proc/stat").unwrap_or_default();
    let mut all = [0u64; 8];
    let mut cores = Vec::new();
    for line in txt.lines() {
        let mut it = line.split_whitespace();
        let Some(name) = it.next() else { continue };
        let vals: [u64; 8] = std::array::from_fn(|_i| {
            it.next().and_then(|w| w.parse().ok()).unwrap_or(0)
        });
        if name == "cpu" {
            all = vals;
        } else if name.starts_with("cpu") {
            cores.push(vals);
        }
    }
    (all, cores)
}

/// Busy percent between two 8-field /proc/stat tuples.
/// Fields: user nice system idle iowait irq softirq steal.
fn busy_pct(prev: &[u64; 8], cur: &[u64; 8]) -> f64 {
    let d: Vec<u64> = (0..8).map(|i| cur[i].saturating_sub(prev[i])).collect();
    let total: u64 = d.iter().sum();
    if total == 0 {
        return 0.0;
    }
    // idle = idle + iowait
    let idle = d[3] + d[4];
    (total - idle) as f64 / total as f64 * 100.0
}

/// /proc/diskstats -> (name, sectors_read, sectors_written) for physical
/// devices only (skip partitions, loop/ram/zram and other virtual ones).
fn s_diskstats() -> Vec<(String, u64, u64)> {
    let txt = fs::read_to_string("/proc/diskstats").unwrap_or_default();
    let mut out = Vec::new();
    for line in txt.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 14 {
            continue;
        }
        let name = f[2];
        if name.starts_with("loop")
            || name.starts_with("ram")
            || name.starts_with("zram")
            || name.starts_with("dm-")
        {
            continue;
        }
        // partitions end in a digit right after a letter-containing base that
        // already exists as a device (nvme0n1p2, sda2): keep whole disks only.
        if is_partition(name) {
            continue;
        }
        let reads = f[5].parse().unwrap_or(0);
        let writes = f[9].parse().unwrap_or(0);
        out.push((name.to_string(), reads, writes));
    }
    out
}

fn is_partition(name: &str) -> bool {
    // nvme0n1p2, mmcblk0p1: 'p<digits>' suffix where the base names a disk
    if let Some(i) = name.rfind('p') {
        let (base, tail) = (&name[..i], &name[i + 1..]);
        if !tail.is_empty()
            && tail.chars().all(|c| c.is_ascii_digit())
            && base.chars().next_back().is_some_and(|c| c.is_ascii_alphanumeric())
            && (base.contains('n') || base.contains("blk"))
        {
            return true;
        }
    }
    // sdX1 style: trailing digits after a short all-letter prefix
    let trimmed = name.trim_end_matches(|c: char| c.is_ascii_digit());
    trimmed != name && trimmed.len() <= 4 && trimmed.chars().all(|c| c.is_ascii_alphabetic())
}

/// /proc/net/dev -> (iface, rx_bytes, tx_bytes), skipping loopback.
fn s_netdev() -> Vec<(String, u64, u64)> {
    let txt = fs::read_to_string("/proc/net/dev").unwrap_or_default();
    let mut out = Vec::new();
    for line in txt.lines().skip(2) {
        let Some((name, rest)) = line.split_once(':') else { continue };
        let f: Vec<&str> = rest.split_whitespace().collect();
        if f.len() < 9 {
            continue;
        }
        let name = name.trim();
        if name == "lo" {
            continue;
        }
        let rx = f[0].parse().unwrap_or(0);
        let tx = f[8].parse().unwrap_or(0);
        out.push((name.to_string(), rx, tx));
    }
    out
}

/// Scan hwmon directories for the hottest temperature input, in °C.
fn hottest_sensor() -> Option<f64> {
    let entries = fs::read_dir("/sys/class/hwmon").ok()?;
    let mut best: Option<f64> = None;
    for e in entries.flatten() {
        let Ok(labels) = fs::read_dir(e.path()) else { continue };
        for l in labels.flatten() {
            let name = l.file_name().to_string_lossy().into_owned();
            // files are temp1_input, temp2_input, …
            let Some(rest) = name.strip_prefix("temp") else { continue };
            let numeric = rest.chars().next().is_some_and(|c| c.is_ascii_digit());
            let tail = if numeric { &rest[1..] } else { rest };
            if tail != "input" {
                continue;
            }
            if let Ok(v) = fs::read_to_string(l.path()) {
                if let Ok(milli) = v.trim().parse::<f64>() {
                    let c = milli / 1000.0;
                    if c > 0.0 && c < 150.0 && best.map(|b| c > b).unwrap_or(true) {
                        best = Some(c);
                    }
                }
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn busy_pct_computes_delta_usage() {
        let prev = [100, 0, 100, 700, 100, 0, 0, 0]; // 1000 total, 800 idle
        let cur = [200, 0, 200, 1500, 100, 0, 0, 0]; // delta: 100+0+100+800+0 = 1000, idle 800
        assert!((busy_pct(&prev, &cur) - 20.0).abs() < 0.01);
    }

    #[test]
    fn busy_pct_zero_total_is_zero() {
        let a = [1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(busy_pct(&a, &a), 0.0);
    }

    #[test]
    fn diskstats_skips_partitions_and_virtual() {
        let rows = s_diskstats();
        for (name, _, _) in &rows {
            assert!(!name.starts_with("loop"), "loop device leaked in: {name}");
            assert!(!is_partition(name), "partition leaked in: {name}");
        }
    }

    #[test]
    fn is_partition_classifies_common_names() {
        assert!(is_partition("sda2"));
        assert!(is_partition("nvme0n1p1"));
        assert!(is_partition("mmcblk0p3"));
        assert!(!is_partition("sda"));
        assert!(!is_partition("nvme0n1"));
        assert!(!is_partition("vda"));
    }

    #[test]
    fn netdev_has_real_interfaces_not_loopback() {
        let rows = s_netdev();
        assert!(!rows.iter().any(|(n, _, _)| n == "lo"));
    }

    #[test]
    fn live_sample_reports_plausible_vitals() {
        let mut s = Sampler::default();
        let a = s.sample();
        std::thread::sleep(Duration::from_millis(120));
        let b = s.sample();
        assert!(b.mem_total > 0, "mem_total must be read");
        assert!(b.ncpu >= 1);
        assert!(b.uptime > 0.0);
        assert!(b.cpu_pct >= 0.0 && b.cpu_pct <= 100.0, "cpu_pct out of range: {}", b.cpu_pct);
        assert_eq!(b.per_core.len(), b.ncpu);
        // second sample has a real interval: rates are finite and non-negative
        assert!(b.disk_read_bps >= 0.0 && b.net_rx_bps >= 0.0);
        assert!(a.cpu_pct == 0.0, "first sample has no rate yet");
    }
}
