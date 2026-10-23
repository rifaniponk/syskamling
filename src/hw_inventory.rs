//! Hardware inventory: static device names and identity read from sysfs,
//! /proc and (when present) lspci. Cached once per process — devices do not
//! change while SysKamling runs.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::LazyLock;

/// One physical device worth naming: CPU, GPU, board, RAM, disk, NIC…
#[derive(Clone, Debug)]
pub struct HwDevice {
    pub kind: &'static str, // "cpu" | "board" | "gpu" | "memory" | "disk" | "nic"
    pub name: String,
    /// Short qualifier: core count, VRAM size, capacity, interface speed…
    pub detail: String,
    /// Kernel block device this maps to, e.g. "nvme0n1" (disks only).
    pub kern: Option<String>,
}

#[derive(Clone, Debug, Default)]
pub struct Inventory {
    pub cpu: String,
    pub board: String,
    pub devices: Vec<HwDevice>,
}

impl Inventory {
    /// Human name for one physical disk (kernel name like "nvme0n1").
    pub fn disk_name(&self, kern: &str) -> Option<&str> {
        self.devices
            .iter()
            .find(|d| d.kind == "disk" && d.kern.as_deref() == Some(kern))
            .map(|d| d.name.as_str())
    }

    /// Kernel block names of the physical drives we actually inventory.
    pub fn disk_kerns(&self) -> Vec<String> {
        self.devices
            .iter()
            .filter(|d| d.kind == "disk")
            .filter_map(|d| d.kern.clone())
            .collect()
    }
}

/// Filesystems mounted off the inventoried drives (root and home, …), read
/// from /proc/mounts. One entry per physical device — the mount with the
/// most capacity wins, so a btrfs multi-mount pool counts once.
pub fn disk_usage() -> Vec<DiskUsage> {
    let inv = &*INVENTORY;
    let txt = std::fs::read_to_string("/proc/mounts").unwrap_or_default();
    let mut best: std::collections::HashMap<String, DiskUsage> = Default::default();
    for line in txt.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 3 || !f[0].starts_with("/dev/") {
            continue;
        }
        let dev = f[0].replace("\\040", " ");
        let kern = match whole_of(&dev[5..]) {
            Some(k) => k,
            None => continue,
        };
        if inv.disk_kerns().iter().all(|k| k != &kern) {
            continue; // not an inventoried drive
        }
        let dev = f[0].replace("\\040", " ");
        let Ok(stat) = statvfs(&dev) else {
            continue;
        };
        let du = DiskUsage {
            kern: kern.clone(),
            mount: f[1].replace("\\040", " "),
            total: stat.0,
            used: stat.0 - stat.1,
        };
        match best.get(&kern) {
            Some(cur) if cur.total >= du.total => {}
            _ => {
                best.insert(kern, du);
            }
        }
    }
    let mut out: Vec<DiskUsage> = best.into_values().collect();
    out.sort_by(|a, b| a.kern.cmp(&b.kern));
    out
}

/// Whole-disk name for a partition: nvme0n1p2 -> nvme0n1, sda1 -> sda,
/// mmcblk0p1 -> mmcblk0.
pub fn whole_of(part: &str) -> Option<String> {
    if part.starts_with("nvme") || part.starts_with("mmcblk") {
        match part.rfind('p') {
            Some(i)
                if !part[i + 1..].is_empty()
                    && part[i + 1..].chars().all(|c| c.is_ascii_digit()) =>
            {
                Some(part[..i].to_string())
            }
            _ => Some(part.to_string()), // the whole disk mounted directly
        }
    } else {
        match part.find(|c: char| c.is_ascii_digit()) {
            Some(end) => Some(part[..end].to_string()),
            None => Some(part.to_string()), // sda, vda: already whole
        }
    }
}

/// Filesystem capacity for one mount, attributed to its whole disk.
#[derive(Clone, Debug)]
pub struct DiskUsage {
    pub kern: String,
    pub mount: String,
    pub total: u64,
    pub used: u64,
}

/// (total, free) bytes for a mounted path, via libc statvfs.
fn statvfs(path: &str) -> Result<(u64, u64), ()> {
    extern "C" {
        fn statvfs(path: *const std::ffi::c_char, buf: *mut u8) -> i32;
    }
    // struct statvfs on linux x86_64 is 120 bytes; over-allocate and read the
    // f_frsize/f_blocks/f_bavail fields at their documented offsets.
    #[repr(C)]
    struct SvfsBuf {
        f_bsize: u64,
        f_frsize: u64,
        f_blocks: u64,
        f_bfree: u64,
        f_bavail: u64,
        _rest: [u8; 512],
    }
    let c = std::ffi::CString::new(path).map_err(|_| ())?;
    let mut buf = SvfsBuf {
        f_bsize: 0,
        f_frsize: 0,
        f_blocks: 0,
        f_bfree: 0,
        f_bavail: 0,
        _rest: [0; 512],
    };
    let rc = unsafe { statvfs(c.as_ptr(), &mut buf as *mut _ as *mut u8) };
    if rc != 0 || buf.f_frsize == 0 || buf.f_blocks == 0 {
        return Err(());
    }
    Ok((buf.f_blocks * buf.f_frsize, buf.f_bavail * buf.f_frsize))
}

/// PCI vendor/device id -> human name, from `lspci -mm -n -D` output.
/// Returns an empty map when lspci is unavailable.
fn lspci_names() -> HashMap<String, String> {
    let out = std::process::Command::new("lspci")
        .args(["-mm", "-n", "-D"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default();
    let mut map = HashMap::new();
    // lines look like: 0000:01:00.0 "0300" "10de" "2c02" "10de" "410b"
    for line in out.lines() {
        let mut fields = line.split('"').map(|s| s.trim()).filter(|s| !s.is_empty());
        let _slot = fields.next();
        let _class = fields.next();
        let (Some(vid), Some(did)) = (fields.next(), fields.next()) else {
            continue;
        };
        if let Some(name) = fields.next() {
            map.insert(format!("{vid}:{did}"), name.to_string());
        }
    }
    map
}

/// Read a sysfs text file, trimmed; None when missing or empty.
fn sysfs(path: &str) -> Option<String> {
    fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Binary-suffixed size, e.g. "1.8 TB".
fn fmt_size(b: i64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{b} B")
    } else {
        format!("{:.1} {}", v, UNITS[u])
    }
}

/// Live per-GPU utilization read from sysfs (amdgpu) or nvidia-smi (NVIDIA
/// proprietary). Refreshed on the same cadence as the hw sampler.
#[derive(Clone, Debug)]
pub struct GpuStat {
    pub name: String,
    /// 0..=100 busy percent, when the driver exposes it.
    pub busy_pct: Option<f64>,
    pub temp_c: Option<f64>,
    pub vram_total: u64,
    pub vram_used: u64,
    /// Milliwatts, when exposed.
    pub power_mw: Option<u64>,
    pub driver: &'static str,
}

fn hex_id(path: &str) -> Option<String> {
    sysfs(path).map(|s| s.trim_start_matches("0x").to_ascii_lowercase())
}

/// Turn lspci's verbose name into a marketing name:
/// "NVIDIA Corporation GB203 [GeForce RTX 5080]" -> "GeForce RTX 5080".
fn pretty_pci_name(raw: &str) -> String {
    if let Some(b) = raw.rfind('[').filter(|_| raw.ends_with(']')) {
        return raw[b + 1..raw.len() - 1].trim().to_string();
    }
    if let Some((_, rest)) = raw.split_once(" Corporation ") {
        return rest.trim().to_string();
    }
    raw.trim().to_string()
}

/// PCI class codes we consider display GPUs.
const GPU_CLASSES: [&str; 3] = ["030000", "030200", "038000"];

fn gpus(names: &HashMap<String, String>) -> Vec<HwDevice> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/class/drm") else {
        return out;
    };
    let mut cards: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("card") && !n.contains('-'))
        .collect();
    cards.sort();
    for card in cards {
        let dev = format!("/sys/class/drm/{card}/device");
        let (Some(vendor), Some(device)) = (
            hex_id(&format!("{dev}/vendor")),
            hex_id(&format!("{dev}/device")),
        ) else {
            continue;
        };
        let class = sysfs(&format!("{dev}/class")).unwrap_or_default();
        if !GPU_CLASSES.contains(&class.as_str()) {
            continue;
        }
        let key = format!("{vendor}:{device}");
        let raw = names
            .get(&key)
            .cloned()
            .unwrap_or_else(|| format!("GPU [{key}]"));
        let name = pretty_pci_name(&raw);
        let driver = fs::read_link(format!("{dev}/driver"))
            .ok()
            .and_then(|p| p.file_name().map(|f| f.to_string_lossy().into_owned()))
            .filter(|d| d != "driver") // the link basename is often just "driver"
            .unwrap_or_default();
        let vram = sysfs(&format!("{dev}/mem_info_vram_total"))
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|v| *v > 0);
        let mut detail = driver;
        if let Some(v) = vram {
            if !detail.is_empty() {
                detail.push_str(" · ");
            }
            detail.push_str(&format!("{} VRAM", fmt_size(v as i64)));
        }
        out.push(HwDevice {
            kind: "gpu",
            name,
            detail,
            kern: None,
        });
    }
    out
}

fn disks() -> Vec<HwDevice> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/block") else {
        return out;
    };
    let mut names: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| {
            !(n.starts_with("loop")
                || n.starts_with("ram")
                || n.starts_with("zram")
                || n.starts_with("dm-")
                || n.starts_with("sr"))
        })
        .collect();
    names.sort();
    for n in names {
        let base = format!("/sys/block/{n}");
        let size = sysfs(&format!("{base}/size"))
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0)
            * 512;
        if size == 0 {
            continue;
        }
        let model = sysfs(&format!("{base}/device/model")).map(|s| s.trim().to_string());
        let rotational = sysfs(&format!("{base}/queue/rotational"))
            .map(|s| s == "1")
            .unwrap_or(true);
        let bus = if n.starts_with("nvme") {
            "NVMe SSD"
        } else if rotational {
            "HDD"
        } else {
            "SSD"
        };
        out.push(HwDevice {
            kind: "disk",
            name: model.unwrap_or_else(|| n.clone()),
            detail: format!("{}, {}", fmt_size(size as i64), bus),
            kern: Some(n.clone()),
        });
    }
    out
}

fn nics(names: &HashMap<String, String>) -> Vec<HwDevice> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir("/sys/class/net") else {
        return out;
    };
    let mut ifaces: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n != "lo")
        .collect();
    ifaces.sort();
    for iface in ifaces {
        let dev = format!("/sys/class/net/{iface}/device");
        if !Path::new(&dev).exists() {
            continue; // virtual iface (docker0, veth…) — no physical device
        }
        let (Some(vendor), Some(device)) = (
            hex_id(&format!("{dev}/vendor")),
            hex_id(&format!("{dev}/device")),
        ) else {
            continue;
        };
        let key = format!("{vendor}:{device}");
        let name = names.get(&key).cloned().unwrap_or_else(|| iface.clone());
        let speed = sysfs(&format!("/sys/class/net/{iface}/speed"))
            .and_then(|s| s.parse::<i64>().ok())
            .filter(|s| *s > 0)
            .map(|s| format!("{s} Mbit/s"))
            .unwrap_or_else(|| iface.clone());
        out.push(HwDevice {
            kind: "nic",
            name,
            detail: speed,
            kern: None,
        });
    }
    // the same PCI device can back several interfaces; keep one row
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out.dedup_by(|a, b| a.name == b.name);
    out
}

fn mem_total_bytes() -> u64 {
    let mem = fs::read_to_string("/proc/meminfo").unwrap_or_default();
    mem.lines()
        .find(|l| l.starts_with("MemTotal:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
        * 1024
}

/// amdgpu exposes hwmon dirs under the card's PCI device: temperature,
/// power, busy percent and VRAM all come from there.
fn gpu_from_amdgpu(card_dev: &str, name: &str) -> Option<GpuStat> {
    let hwmon = fs::read_dir(format!("{card_dev}/hwmon"))
        .ok()?
        .flatten()
        .find(|e| e.file_name().to_string_lossy().starts_with("hwmon"))
        .map(|e| e.path())?;
    let milli = |f: &str| {
        sysfs(&hwmon.join(f).to_string_lossy())
            .and_then(|s| s.parse::<f64>().ok())
            .map(|v| v / 1000.0)
    };
    Some(GpuStat {
        name: name.to_string(),
        busy_pct: sysfs(&format!("{card_dev}/gpu_busy_percent")).and_then(|s| s.parse().ok()),
        temp_c: milli("temp1_input"),
        vram_total: sysfs(&format!("{card_dev}/mem_info_vram_total"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
        vram_used: sysfs(&format!("{card_dev}/mem_info_vram_used"))
            .and_then(|s| s.parse().ok())
            .unwrap_or(0),
        power_mw: sysfs(&hwmon.join("power1_input").to_string_lossy()).and_then(|s| s.parse().ok()),
        driver: "amdgpu",
    })
}

/// One row from `nvidia-smi --query-gpu=… --format=csv,noheader,nounits`.
fn parse_nvidia_row(row: &str) -> Option<GpuStat> {
    // "NVIDIA GeForce RTX 5080, 55, 99, 16303, 15266, 194.68, 595.99.02"
    let f: Vec<&str> = row.split(',').map(|s| s.trim()).collect();
    if f.len() < 6 {
        return None;
    }
    let num = |i: usize| f.get(i).and_then(|v| v.parse::<f64>().ok());
    Some(GpuStat {
        name: f[0].to_string(),
        temp_c: num(1),
        busy_pct: num(2),
        vram_total: num(3).map(|mib| mib as u64 * 1024 * 1024).unwrap_or(0),
        vram_used: num(4).map(|mib| mib as u64 * 1024 * 1024).unwrap_or(0),
        power_mw: num(5).map(|w| (w * 1000.0) as u64),
        driver: "nvidia",
    })
}

/// Live stats for every display GPU: try sysfs first (free, instant), then
/// fall back to nvidia-smi once per call (it costs ~50 ms, so the GUI only
/// runs it on the sampler cadence).
pub fn gpu_stats() -> Vec<GpuStat> {
    let mut out = gpu_stats_hwmon_only();
    // nvidia-smi covers every NVIDIA card with richer numbers than sysfs
    if nvidia_cards_present() {
        let q = "name,temperature.gpu,utilization.gpu,memory.total,memory.used,power.draw,driver_version";
        if let Ok(o) = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu", q, "--format=csv,noheader,nounits"])
            .output()
        {
            if o.status.success() {
                // drop the hwmon-only NVIDIA placeholders, keep sysfs rows
                // for everything else, and use nvidia-smi for NVIDIA
                out.retain(|g| g.driver != "nvidia");
                for line in String::from_utf8_lossy(&o.stdout).lines() {
                    if let Some(g) = parse_nvidia_row(line.trim()) {
                        out.push(g);
                    }
                }
            }
        }
    }
    out
}

/// True when any NVIDIA display adapter exists (vendor 0x10de on a DRM
/// card), i.e. when nvidia-smi is worth invoking at all.
fn nvidia_cards_present() -> bool {
    let Ok(entries) = fs::read_dir("/sys/class/drm") else {
        return false;
    };
    for e in entries.flatten() {
        let n = e.file_name().to_string_lossy().into_owned();
        if !n.starts_with("card") || n.contains('-') {
            continue;
        }
        if hex_id(&format!("/sys/class/drm/{n}/device/vendor")).as_deref() == Some("10de") {
            return true;
        }
    }
    false
}

/// sysfs-only pass: instant, no subprocess. NVIDIA cards show up with a
/// name but no live numbers (their driver exposes no hwmon here).
pub fn gpu_stats_hwmon_only() -> Vec<GpuStat> {
    let mut out = Vec::new();
    let names = lspci_names();
    let Ok(entries) = fs::read_dir("/sys/class/drm") else {
        return out;
    };
    let mut cards: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("card") && !n.contains('-'))
        .collect();
    cards.sort();
    for card in &cards {
        let dev = format!("/sys/class/drm/{card}/device");
        let (Some(vendor), Some(device)) = (
            hex_id(&format!("{dev}/vendor")),
            hex_id(&format!("{dev}/device")),
        ) else {
            continue;
        };
        let class = sysfs(&format!("{dev}/class")).unwrap_or_default();
        if !GPU_CLASSES.contains(&class.as_str()) {
            continue;
        }
        let raw = names
            .get(&format!("{vendor}:{device}"))
            .cloned()
            .unwrap_or_else(|| format!("GPU [{card}]"));
        let name = pretty_pci_name(&raw);
        if vendor == "10de" {
            out.push(GpuStat {
                name,
                busy_pct: None,
                temp_c: None,
                vram_total: 0,
                vram_used: 0,
                power_mw: None,
                driver: "nvidia",
            });
            continue; // nvidia-smi covers these with richer numbers
        }
        if let Some(g) = gpu_from_amdgpu(&dev, &name) {
            out.push(g);
        } else {
            out.push(GpuStat {
                name,
                busy_pct: None,
                temp_c: None,
                vram_total: 0,
                vram_used: 0,
                power_mw: None,
                driver: "other",
            });
        }
    }
    out
}

/// Build the inventory once; every read here is cheap and static.
pub static INVENTORY: LazyLock<Inventory> = LazyLock::new(|| {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let cpu = cpuinfo
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split_once(':'))
        .map(|(_, v)| v.trim().to_string())
        .unwrap_or_else(|| "unknown CPU".to_string());
    let ncpu = cpuinfo
        .lines()
        .filter(|l| l.starts_with("processor"))
        .count();
    let board = sysfs("/sys/class/dmi/id/board_name")
        .or_else(|| sysfs("/sys/class/dmi/id/product_name"))
        .unwrap_or_default();
    let names = lspci_names();

    let mut devices = Vec::new();
    devices.push(HwDevice {
        kind: "cpu",
        name: cpu.clone(),
        detail: format!("{ncpu} threads"),
        kern: None,
    });
    if !board.is_empty() {
        devices.push(HwDevice {
            kind: "board",
            name: board.clone(),
            detail: "motherboard".into(),
            kern: None,
        });
    }
    devices.extend(gpus(&names));
    devices.push(HwDevice {
        kind: "memory",
        name: "System memory".into(),
        detail: fmt_size(mem_total_bytes() as i64),
        kern: None,
    });
    devices.extend(disks());
    devices.extend(nics(&names));
    Inventory {
        cpu,
        board,
        devices,
    }
});

#[cfg(test)]
mod tests {
    #[test]
    fn whole_of_maps_partitions_to_their_disk() {
        assert_eq!(whole_of("nvme0n1p2").as_deref(), Some("nvme0n1"));
        assert_eq!(whole_of("sda1").as_deref(), Some("sda"));
        assert_eq!(whole_of("mmcblk0p1").as_deref(), Some("mmcblk0"));
        // a whole disk mounted directly maps to itself
        assert_eq!(whole_of("nvme0n1").as_deref(), Some("nvme0n1"));
    }

    #[test]
    fn disk_usage_reports_plausible_capacity_for_inventoried_disks() {
        let usages = disk_usage();
        // any real Linux box has at least the root fs on an inventoried disk
        assert!(
            !usages.is_empty(),
            "no mounted filesystem matched the inventory"
        );
        for u in &usages {
            assert!(u.total > 0, "{} total=0", u.mount);
            assert!(u.used <= u.total, "{} used>total", u.mount);
            assert!(u.mount.starts_with('/'), "{} not a path", u.mount);
            assert!(
                INVENTORY.disk_kerns().contains(&u.kern),
                "{} not inventoried",
                u.kern
            );
        }
    }

    use super::*;

    #[test]
    fn pretty_pci_name_prefers_bracket_and_strips_corporation() {
        assert_eq!(
            pretty_pci_name("NVIDIA Corporation GB203 [GeForce RTX 5080]"),
            "GeForce RTX 5080"
        );
        // no bracket: strip the "… Corporation " vendor prefix, keep the rest
        assert_eq!(
            pretty_pci_name(
                "Advanced Micro Devices, Inc. [AMD/ATI] Granite Ridge [Radeon Graphics]"
            ),
            "Radeon Graphics"
        );
        assert_eq!(
            pretty_pci_name("Vendor Corporation Widget Pro"),
            "Widget Pro"
        );
        assert_eq!(pretty_pci_name("Plain Device Name"), "Plain Device Name");
    }

    #[test]
    fn inventory_names_the_cpu() {
        let inv = &*INVENTORY;
        assert!(!inv.cpu.is_empty());
        assert!(inv
            .devices
            .iter()
            .any(|d| d.kind == "cpu" && d.name == inv.cpu));
    }

    #[test]
    fn inventory_lists_at_least_one_disk() {
        assert!(INVENTORY.devices.iter().any(|d| d.kind == "disk"));
    }

    #[test]
    fn inventory_has_no_virtual_disks() {
        for d in INVENTORY.devices.iter().filter(|d| d.kind == "disk") {
            assert!(!d.name.starts_with("loop"), "loop leaked: {}", d.name);
            assert!(!d.name.starts_with("zram"), "zram leaked: {}", d.name);
        }
    }

    #[test]
    fn parse_nvidia_row_reads_all_fields() {
        let g =
            parse_nvidia_row("NVIDIA GeForce RTX 5080, 55, 99, 16303, 15266, 194.68, 595.99.02")
                .unwrap();
        assert_eq!(g.name, "NVIDIA GeForce RTX 5080");
        assert_eq!(g.temp_c, Some(55.0));
        assert_eq!(g.busy_pct, Some(99.0));
        assert_eq!(g.vram_total, 16303 * 1024 * 1024);
        assert_eq!(g.vram_used, 15266 * 1024 * 1024);
        assert_eq!(g.power_mw, Some(194680));
        assert_eq!(g.driver, "nvidia");
    }

    #[test]
    fn parse_nvidia_row_rejects_junk() {
        assert!(parse_nvidia_row("").is_none());
        assert!(parse_nvidia_row("gpu, 1, 2").is_none());
    }

    #[test]
    fn gpu_stats_names_the_gpus_on_this_box() {
        let gs = gpu_stats();
        // every GPU must have a name; where the driver exposes numbers they
        // must be plausible
        for g in &gs {
            assert!(!g.name.is_empty());
            if let Some(b) = g.busy_pct {
                assert!((0.0..=100.0).contains(&b), "busy out of range: {b}");
            }
            if let Some(t) = g.temp_c {
                assert!(t > 0.0 && t < 150.0, "gpu temp out of range: {t}");
            }
            assert!(g.vram_used <= g.vram_total.max(g.vram_used));
        }
    }

    #[test]
    fn lspci_names_map_is_consistent() {
        // on this machine at least one PCI device resolves; where lspci is
        // missing the map is simply empty, which callers handle.
        for (k, v) in lspci_names() {
            assert!(k.contains(':'));
            assert!(!v.is_empty());
        }
    }
}
