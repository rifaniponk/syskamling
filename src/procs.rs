//! Process scanner: every process on the box with live CPU% and memory,
//! Task-Manager style. Reads /proc, keeps the previous tick counters so the
//! delta over the scan interval gives a real busy percentage.

use std::collections::HashMap;
use std::fs;
use std::time::{Duration, Instant};

/// One process row.
#[derive(Clone, Debug)]
pub struct ProcInfo {
    pub pid: u32,
    pub ppid: u32,
    pub name: String,
    /// joined argv, NULs replaced by spaces, elided to 200 chars
    pub cmdline: String,
    /// "R" running, "S" sleeping, "D" uninterruptible, "Z" zombie…
    pub state: char,
    /// CPU percent of one core, 0..=100*ncpu (same scale as Docker stats)
    pub cpu_pct: f64,
    /// resident set size, bytes
    pub rss_bytes: u64,
    pub threads: u32,
    /// epoch seconds the process started
    pub started: i64,
    /// process owner uid
    pub uid: u32,
}

/// Incremental scanner: call `scan()` once per refresh; the second and every
/// later call reports real CPU percentages for the elapsed interval.
pub struct ProcScanner {
    prev: HashMap<u32, (u64, Instant)>,
    prev_at: Option<Instant>,
}

impl Default for ProcScanner {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcScanner {
    pub fn new() -> Self {
        Self { prev: HashMap::new(), prev_at: None }
    }

    pub fn scan(&mut self) -> Vec<ProcInfo> {
        let btime = boot_epoch_secs();
        let ticks = 100.0_f64; // KERNEL_TICKS on every common Linux port
        let now = Instant::now();
        let dt = self.prev_at.map(|t| now.duration_since(t)).unwrap_or(Duration::ZERO);
        let page = page_size();

        let mut rows = Vec::new();
        let mut cur: HashMap<u32, (u64, Instant)> = HashMap::new();

        let dir = match fs::read_dir("/proc") {
            Ok(d) => d,
            Err(_) => return rows,
        };
        for entry in dir.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let Ok(pid): Result<u32, _> = name.parse() else { continue };

            let stat = match fs::read(format!("/proc/{pid}/stat")) {
                Ok(b) => String::from_utf8_lossy(&b).into_owned(),
                Err(_) => continue,
            };
            // comm is in parentheses and may contain spaces/parens: split at
            // the LAST ')' then parse the fields after it.
            let Some(close) = stat.rfind(')') else { continue };
            let comm = stat[..close].find('(').map(|i| stat[i + 1..close].to_string()).unwrap_or_default();
            let rest: Vec<&str> = stat[close + 1..].split_whitespace().collect();
            // rest[0]=state rest[1]=ppid rest[11]=utime rest[12]=stime
            // rest[17]=threads rest[21]=vsize rest[23]=rss (pages)
            if rest.len() < 24 {
                continue;
            }
            let state = rest[0].chars().next().unwrap_or('?');
            let ppid: u32 = rest[1].parse().unwrap_or(0);
            let utime: u64 = rest[11].parse().unwrap_or(0);
            let stime: u64 = rest[12].parse().unwrap_or(0);
            let threads: u32 = rest[17].parse().unwrap_or(0);
            let rss_pages: u64 = rest[23].parse().unwrap_or(0);
            let starttime: u64 = rest[22].parse().unwrap_or(0);
            let cpu_ticks = utime + stime;
            cur.insert(pid, (cpu_ticks, now));

            let cpu_pct = match self.prev.get(&pid) {
                Some((p, _)) if dt > Duration::ZERO => {
                    let d = cpu_ticks.saturating_sub(*p) as f64 / ticks / dt.as_secs_f64() * 100.0;
                    d
                }
                _ => 0.0,
            };

            let cmdline = fs::read(format!("/proc/{pid}/cmdline"))
                .ok()
                .map(|b| {
                    let s = b
                        .split(|c| *c == 0)
                        .filter(|s| !s.is_empty())
                        .map(|s| String::from_utf8_lossy(s).into_owned())
                        .collect::<Vec<_>>()
                        .join(" ");
                    if s.chars().count() > 200 {
                        s.chars().take(197).collect::<String>() + "…"
                    } else {
                        s
                    }
                })
                .unwrap_or_default();

            let uid = fs::metadata(format!("/proc/{pid}"))
                .ok()
                .map(|m| uid_of(&m))
                .unwrap_or(0);

            rows.push(ProcInfo {
                pid,
                ppid,
                name: comm,
                cmdline,
                state,
                cpu_pct,
                rss_bytes: rss_pages * page,
                threads,
                started: (btime as f64 + starttime as f64 / ticks) as i64,
                uid,
            });
        }

        self.prev = cur;
        self.prev_at = Some(now);
        rows
    }
}

/// uid of a /proc/<pid> directory owner; std has no portable accessor, so
/// read the numeric owner straight from `stat`.
fn uid_of(meta: &fs::Metadata) -> u32 {
    use std::os::unix::fs::MetadataExt;
    meta.uid()
}

fn boot_epoch_secs() -> u64 {
    fs::read_to_string("/proc/stat")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("btime "))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|w| w.parse().ok())
        })
        .unwrap_or(0)
}

/// System page size via the `getpagesize` libc symbol glibc always provides.
fn page_size() -> u64 {
    extern "C" {
        fn getpagesize() -> i32;
    }
    let v = unsafe { getpagesize() };
    if v > 0 {
        v as u64
    } else {
        4096
    }
}

/// Sort rows by the given column; secondary sort is always pid so the order
/// is stable and deterministic.
pub fn sort_rows(rows: &mut [ProcInfo], col: SortCol, desc: bool) {
    use SortCol::*;
    rows.sort_by(|a, b| {
        let ord = match col {
            Name => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
            Pid => a.pid.cmp(&b.pid),
            Cpu => a.cpu_pct.total_cmp(&b.cpu_pct),
            Mem => a.rss_bytes.cmp(&b.rss_bytes),
        };
        let ord = if desc { ord.reverse() } else { ord };
        ord.then(a.pid.cmp(&b.pid))
    });
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SortCol {
    Name,
    Pid,
    Cpu,
    Mem,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(pid: u32, name: &str, cpu: f64, mem: u64) -> ProcInfo {
        ProcInfo {
            pid,
            ppid: 0,
            name: name.into(),
            cmdline: String::new(),
            state: 'S',
            cpu_pct: cpu,
            rss_bytes: mem,
            threads: 1,
            started: 0,
            uid: 1000,
        }
    }

    #[test]
    fn sort_by_memory_desc_then_pid() {
        let mut rows = vec![row(2, "b", 1.0, 100), row(3, "c", 2.0, 300), row(4, "d", 3.0, 300)];
        sort_rows(&mut rows, SortCol::Mem, true);
        assert_eq!(rows.iter().map(|r| r.pid).collect::<Vec<_>>(), vec![3, 4, 2]);
    }

    #[test]
    fn sort_by_cpu_asc_and_name_case_insensitive() {
        let mut rows = vec![row(1, "zsh", 5.0, 1), row(2, "Bash", 9.0, 1), row(3, "apache", 1.0, 1)];
        sort_rows(&mut rows, SortCol::Cpu, false);
        assert_eq!(rows.iter().map(|r| r.pid).collect::<Vec<_>>(), vec![3, 1, 2]);
        sort_rows(&mut rows, SortCol::Name, false);
        assert_eq!(rows.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(), vec!["apache", "Bash", "zsh"]);
    }

    #[test]
    fn live_scan_finds_this_test_process_with_plausible_numbers() {
        let mut s = ProcScanner::new();
        s.scan();
        std::thread::sleep(Duration::from_millis(120));
        let rows = s.scan();
        assert!(!rows.is_empty(), "/proc scan found nothing");
        // this very test binary must appear
        let me = std::process::id();
        let me = rows.iter().find(|r| r.pid == me).expect("own pid missing");
        assert!(!me.name.is_empty());
        assert!(me.rss_bytes > 0, "rss must be non-zero for a live process");
        assert!(me.cpu_pct >= 0.0 && me.cpu_pct < 100.0 * 256.0, "cpu_pct implausible: {}", me.cpu_pct);
        assert!(me.started > 1_600_000_000, "start time implausible: {}", me.started);
        // the init process exists and owns the tree
        assert!(rows.iter().any(|r| r.pid == 1));
    }
}
