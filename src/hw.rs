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
    /// Hottest coretemp/mtemp sensor reading, Celsius, if any. Kept for
    /// summary strips that just want "the hottest thing on the box".
    pub temp_c: Option<f64>,
    /// Uptime seconds.
    pub uptime: f64,
    /// Named temperature sensors worth showing (CPU package, GPU edge,
    /// NVMe composite, …) — hottest per device group first.
    pub sensors: Vec<TempSensor>,
    /// CPU package temperature (k10temp Tctl / coretemp Package), if known.
    pub cpu_temp_c: Option<f64>,
    /// Hottest GPU sensor, if any.
    pub gpu_temp_c: Option<f64>,
    /// Per physical disk: live rates plus lifetime counters.
    pub disks: Vec<DiskStat>,
}

/// One named temperature sensor, e.g. ("CPU", "Tctl", 72.4).
#[derive(Clone, Debug)]
pub struct TempSensor {
    pub group: &'static str, // "cpu" | "gpu" | "nvme" | "other"
    pub label: String,
    pub c: f64,
}

/// Live + lifetime stats for one whole physical disk.
#[derive(Clone, Debug, Default)]
pub struct DiskStat {
    pub name: String,
    pub read_bps: f64,
    pub write_bps: f64,
    /// Lifetime bytes read/written (sectors × 512).
    pub total_read: u64,
    pub total_write: u64,
    /// Lifetime completed IOs.
    pub reads: u64,
    pub writes: u64,
    /// Weighted-busy milliseconds since boot (for the avg-queue figure).
    pub ms_weighted: u64,
    /// NVMe drive temperature, Celsius, when exposed via hwmon.
    pub temp_c: Option<f64>,
    /// Seconds of uptime at sample time, used for lifetime averages.
    pub uptime: f64,
}

/// Full /proc/diskstats row for one whole physical device.
pub struct DiskRow {
    pub name: String,
    pub reads: u64,          // completed read IOs
    pub reads_sectors: u64,  // 512 B sectors
    pub writes: u64,
    pub writes_sectors: u64,
    pub ms_weighted: u64,
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
            // disk: full stats rows (sectors in 512 B units) -> per-disk
            // live rates plus lifetime counters, and the summed totals.
            let cur_disks = diskstats_full();
            let mut dr = 0u64;
            let mut dw = 0u64;
            for d in &cur_disks {
                let Some(prev) = p.disk.iter().find(|(n, _, _)| *n == d.name) else {
                    continue;
                };
                let rd = d.reads_sectors.saturating_sub(prev.1);
                let wr = d.writes_sectors.saturating_sub(prev.2);
                dr += rd;
                dw += wr;
                s.disks.push(DiskStat {
                    name: d.name.clone(),
                    read_bps: rd as f64 * 512.0 / dt,
                    write_bps: wr as f64 * 512.0 / dt,
                    total_read: d.reads_sectors * 512,
                    total_write: d.writes_sectors * 512,
                    reads: d.reads,
                    writes: d.writes,
                    ms_weighted: d.ms_weighted,
                    temp_c: None,
                    uptime: s.uptime,
                });
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
            disk: cur_disk_keys(),
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

        // ---- temperatures (millidegrees from hwmon, named per device) ----
        let sensors = hwmon_sensors();
        s.temp_c = sensors.iter().map(|x| x.c).fold(None::<f64>, |a, b| Some(a.map_or(b, |a| a.max(b))));
        s.cpu_temp_c = sensors
            .iter()
            .filter(|x| x.group == "cpu")
            .map(|x| x.c)
            .fold(None::<f64>, |a, b| Some(a.map_or(b, |a| a.max(b))));
        s.gpu_temp_c = sensors
            .iter()
            .filter(|x| x.group == "gpu")
            .map(|x| x.c)
            .fold(None::<f64>, |a, b| Some(a.map_or(b, |a| a.max(b))));
        for d in &mut s.disks {
            // NVMe hwmon dirs live under the controller, e.g. /sys/class/nvme/nvme0/hwmon1
            if let Some(n) = d.name.strip_prefix("nvme") {
                let idx: usize = n.chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().unwrap_or(0);
                let ctrl = format!("/sys/class/nvme/nvme{idx}");
                if let Ok(entries) = fs::read_dir(&ctrl) {
                    for e in entries.flatten() {
                        let p = e.path();
                        if p.file_name().is_some_and(|f| f.to_string_lossy().starts_with("hwmon")) {
                            if let Ok(v) = fs::read_to_string(p.join("temp1_input")) {
                                if let Ok(m) = v.trim().parse::<f64>() {
                                    d.temp_c = Some(m / 1000.0);
                                }
                            }
                            break;
                        }
                    }
                }
            }
        }
        s.sensors = sensors;
        // fallback for any NVMe disk whose controller dir exposed no hwmon:
        // the generic scan above already saw the drive's composite sensor
        if s.sensors.iter().any(|x| x.group == "nvme") {
            let nvme_temp = s.sensors.iter().filter(|x| x.group == "nvme").map(|x| x.c).fold(None::<f64>, |a, b| Some(a.map_or(b, |a| a.max(b))));
            for d in &mut s.disks {
                if d.temp_c.is_none() && d.name.starts_with("nvme") {
                    d.temp_c = nvme_temp;
                }
            }
        }
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

/// Keys-only snapshot for the previous-scan slot.
fn cur_disk_keys() -> Vec<(String, u64, u64)> {
    diskstats_full()
        .into_iter()
        .map(|d| (d.name, d.reads_sectors, d.writes_sectors))
        .collect()
}

/// Full physical-device rows: name plus the counter fields we need.
/// Field layout (0-based after name): 3=reads, 5=reads_sectors, 7=writes,
/// 9=writes_sectors, 12=ms_busy, 15=ms_weighted (per kernel Documentation).
fn diskstats_full() -> Vec<DiskRow> {
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
        if is_partition(name) {
            continue;
        }
        let g = |i: usize| f.get(i).and_then(|v| v.parse().ok()).unwrap_or(0);
        out.push(DiskRow {
            name: name.to_string(),
            reads: g(3),
            reads_sectors: g(5),
            writes: g(7),
            writes_sectors: g(9),
            ms_weighted: g(13),
        });
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

/// hwmon driver names grouped by what they measure.
fn sensor_group(driver: &str) -> &'static str {
    if driver.starts_with("k10temp")
        || driver.starts_with("k8temp")
        || driver.starts_with("coretemp")
        || driver.starts_with("zenpower")
        || driver.contains("thermal")
    {
        "cpu"
    } else if driver.starts_with("amdgpu")
        || driver.starts_with("nvidia")
        || driver.starts_with("nouveau")
        || driver.starts_with("i915")
        || driver.starts_with("radeon")
    {
        "gpu"
    } else if driver.starts_with("nvme") {
        "nvme"
    } else {
        "other"
    }
}

/// Scan every hwmon directory and return the named temperature inputs that
/// are worth showing: package/edge/composite sensors first, one hottest per
/// device. Sensors without labels are included only for "other" devices.
fn hwmon_sensors() -> Vec<TempSensor> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/class/hwmon") else {
        return out;
    };
    for e in entries.flatten() {
        let dir = e.path();
        let driver = fs::read_to_string(dir.join("name"))
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        let group = sensor_group(&driver);
        let Ok(files) = fs::read_dir(&dir) else { continue };
        let mut named: Vec<(String, f64)> = Vec::new();
        let mut unnamed: Vec<f64> = Vec::new();
        for l in files.flatten() {
            let fname = l.file_name().to_string_lossy().into_owned();
            let Some(rest) = fname.strip_prefix("temp") else { continue };
            let numeric = rest.chars().next().is_some_and(|c| c.is_ascii_digit());
            if !numeric || &rest[1..] != "input" {
                continue;
            }
            let idx = &rest[..rest.len() - "input".len()];
            let Ok(v) = fs::read_to_string(l.path()) else { continue };
            let Ok(milli) = v.trim().parse::<f64>() else { continue };
            let c = milli / 1000.0;
            if c <= 0.0 || c > 150.0 {
                continue;
            }
            let label = fs::read_to_string(dir.join(format!("temp{idx}_label")))
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            if !label.is_empty() {
                named.push((label, c));
            } else {
                unnamed.push(c);
            }
        }
        // Skip devices nobody cares about (battery gauges, WiFi PHYs, NIC
        // PHYs) unless they carry a labelled sensor of a known group.
        if group == "other" && named.is_empty() {
            continue;
        }
        if named.is_empty() {
            if let Some(c) = unnamed.into_iter().fold(None::<f64>, |a, b| Some(a.map_or(b, |a| a.max(b)))) {
                out.push(TempSensor { group, label: driver, c });
            }
            continue;
        }
        // keep the meaningful labels: Tctl, edge, Composite; drop per-core
        // duplicates (Tccd*) and secondary NVMe sensors ("Sensor 1/2")
        named.retain(|(l, _)| {
            !(l.starts_with("Tccd") || l.starts_with("Sensor") || l.starts_with("aux"))
        });
        for (label, c) in named {
            out.push(TempSensor { group, label, c });
        }
    }
    out
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
        let rows = diskstats_full();
        for d in &rows {
            assert!(!d.name.starts_with("loop"), "loop device leaked in: {}", d.name);
            assert!(!is_partition(&d.name), "partition leaked in: {}", d.name);
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

    #[test]
    fn live_sample_reports_named_sensors() {
        let mut s = Sampler::default();
        s.sample();
        std::thread::sleep(Duration::from_millis(120));
        let b = s.sample();
        // on any Linux box with hwmon sensors there is at least one CPU temp;
        // on a bare VM there may be none, but groups must be consistent.
        for x in &b.sensors {
            assert!(x.c > 0.0 && x.c < 150.0, "implausible sensor {}={}", x.label, x.c);
            assert!(matches!(x.group, "cpu" | "gpu" | "nvme" | "other"));
            assert!(!x.label.is_empty());
        }
        // derived aggregates must agree with the sensor list
        if let Some(t) = b.cpu_temp_c {
            assert!(b.sensors.iter().any(|x| x.group == "cpu" && (x.c - t).abs() < 0.01));
        }
        if let Some(t) = b.temp_c {
            assert!(b.sensors.iter().any(|x| (x.c - t).abs() < 0.01));
        }
    }

    #[test]
    fn per_disk_stats_have_lifetime_counters_and_rates() {
        let mut s = Sampler::default();
        s.sample();
        std::thread::sleep(Duration::from_millis(120));
        let b = s.sample();
        assert!(!b.disks.is_empty(), "expected at least one physical disk");
        for d in &b.disks {
            assert!(d.read_bps >= 0.0 && d.write_bps >= 0.0);
            assert!(d.uptime > 0.0);
            if let Some(t) = d.temp_c {
                assert!(t > 0.0 && t < 150.0, "implausible disk temp {t}");
            }
        }
        // NVMe drives expose hwmon temps on real hardware; if one is present
        // it must carry a temperature.
        if b.disks.iter().any(|d| d.name.starts_with("nvme")) {
            assert!(b.disks.iter().filter(|d| d.name.starts_with("nvme")).all(|d| d.temp_c.is_some()));
        }
    }

    #[test]
    fn sensor_group_classifies_common_drivers() {
        assert_eq!(sensor_group("k10temp"), "cpu");
        assert_eq!(sensor_group("coretemp"), "cpu");
        assert_eq!(sensor_group("amdgpu"), "gpu");
        assert_eq!(sensor_group("nvidia"), "gpu");
        assert_eq!(sensor_group("nvme"), "nvme");
        assert_eq!(sensor_group("hidpp_battery_0"), "other");
        assert_eq!(sensor_group("mt7925_phy0"), "other");
    }
}
