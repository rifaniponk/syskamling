//! Types returned by the Docker Engine API.
//!
//! Field names match the daemon JSON exactly; most endpoints use
//! `rename_all = "camelCase"`, while the stats structures use snake_case
//! because the daemon emits raw cgroup-style keys there.
#![allow(dead_code)] // wire-format fields kept for fidelity even when unused

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContainerSummary {
    #[serde(rename = "Id")]
    pub id: String,
    #[serde(rename = "Names")]
    pub names: Vec<String>,
    #[serde(rename = "Image")]
    pub image: String,
    #[serde(rename = "State")]
    pub state: String,
    #[serde(rename = "Status")]
    pub status: String,
    #[serde(default, rename = "Labels")]
    pub labels: std::collections::BTreeMap<String, String>,
    #[serde(default, rename = "Ports")]
    pub ports: Vec<Port>,
    #[serde(default, rename = "SizeRw")]
    pub size_rw: Option<i64>,
    #[serde(default, rename = "SizeRootFs")]
    pub size_root_fs: Option<i64>,
    #[serde(default, rename = "Created")]
    pub created: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Port {
    #[serde(default)]
    pub ip: Option<String>,
    #[serde(default)]
    pub private_port: Option<u16>,
    #[serde(default)]
    pub public_port: Option<u16>,
    #[serde(default)]
    pub typ: Option<String>,
}

impl ContainerSummary {
    /// The daemon prefixes names with `/`; strip it and take the first name.
    pub fn short_name(&self) -> String {
        self.names
            .first()
            .map(|n| n.trim_start_matches('/').to_string())
            .unwrap_or_else(|| self.id[..12].to_string())
    }

    /// Compose project label, if the container belongs to a project.
    pub fn project(&self) -> Option<&str> {
        self.labels
            .get("com.docker.compose.project")
            .map(|s| s.as_str())
    }

    /// Human-readable port mapping, e.g. `0.0.0.0:8080->80/tcp`.
    pub fn port_summary(&self) -> String {
        self.ports
            .iter()
            .filter_map(|p| {
                let public = p.public_port?;
                let private = p.private_port?;
                let proto = p.typ.as_deref().unwrap_or("tcp");
                let ip = p.ip.as_deref().unwrap_or("0.0.0.0");
                Some(format!("{ip}:{public}->{private}/{proto}"))
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageSummary {
    #[serde(rename = "Id")]
    pub id: String,
    #[serde(default, rename = "RepoTags")]
    pub repo_tags: Vec<String>,
    #[serde(rename = "Created")]
    pub created: i64,
    #[serde(rename = "Size")]
    pub size: i64,
    #[serde(default)]
    pub shared_size: i64,
    #[serde(default)]
    pub containers: i64,
}

impl ImageSummary {
    pub fn tag(&self) -> String {
        self.repo_tags
            .first()
            .cloned()
            .unwrap_or_else(|| self.id[..19.min(self.id.len())].to_string())
    }

    pub fn short_id(&self) -> &str {
        let id = self.id.trim_start_matches("sha256:");
        &id[..12.min(id.len())]
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VolumeEntry {
    #[serde(rename = "Name")]
    pub name: String,
    #[serde(rename = "Driver")]
    pub driver: String,
    #[serde(default, rename = "CreatedAt")]
    pub created: String,
    #[serde(default, rename = "UsageData")]
    pub usage_data: Option<VolumeUsage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VolumeUsage {
    #[serde(default, rename = "Size")]
    pub size: Option<i64>,
    #[serde(default, rename = "RefCount")]
    pub ref_count: i64,
}

/// One sample from `/containers/{id}/stats` (stream mode).
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct StatsSample {
    #[serde(default)]
    pub read: String,
    #[serde(default)]
    pub cpu_stats: CpuStats,
    #[serde(default)]
    pub precpu_stats: CpuStats,
    #[serde(default)]
    pub memory_stats: MemoryStats,
    #[serde(default)]
    pub blkio_stats: BlkioStats,
    #[serde(default)]
    pub pids_stats: PidsStats,
    #[serde(default)]
    pub networks: std::collections::BTreeMap<String, NetStats>,
    /// cgroup v2 stats (`memory` / `cpu` blocks) the modern daemon emits;
    /// used as a fallback when the v1-style blocks are absent.
    #[serde(default)]
    pub memory: Option<cgroup2::Memory>,
    #[serde(default)]
    pub cpu: Option<cgroup2::Cpu>,
    #[serde(default)]
    pub blkio: Option<cgroup2::Blkio>,
}

mod cgroup2 {
    use serde::Deserialize;

    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct Memory {
        #[serde(default)]
        pub usage: i64,
        #[serde(default)]
        pub limit: i64,
    }

    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct Cpu {
        #[serde(default)]
        pub system_cpu_usage: Option<i64>,
        #[serde(default)]
        pub online_cpus: Option<usize>,
        #[serde(default)]
        pub usage: CpuUsage,
    }

    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct CpuUsage {
        #[serde(default)]
        pub kernel: i64,
        #[serde(default)]
        pub user: i64,
        #[serde(default)]
        pub total: i64,
    }

    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct Blkio {
        #[serde(default)]
        pub io_service_bytes_recursive: Vec<IoEntry>,
    }

    #[derive(Debug, Clone, Default, Deserialize)]
    pub struct IoEntry {
        #[serde(rename = "Op")]
        pub op: String,
        pub value: i64,
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CpuStats {
    #[serde(default)]
    pub cpu_usage: CpuUsage,
    #[serde(default)]
    pub online_cpus: Option<usize>,
    #[serde(default)]
    pub system_cpu_usage: Option<i64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CpuUsage {
    #[serde(default, rename = "total_usage")]
    pub total_usage: i64,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct MemoryStats {
    #[serde(default)]
    pub usage: i64,
    #[serde(default)]
    pub max_usage: Option<i64>,
    #[serde(default)]
    pub limit: i64,
    #[serde(default)]
    pub stats: Option<serde_json::Value>,
    #[serde(default)]
    pub failcnt: Option<i64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct BlkioStats {
    // The daemon sends `null` instead of `[]` when a cgroup file is empty
    // (common on cgroup v2), so the field is optional.
    #[serde(default, rename = "io_service_bytes_recursive")]
    pub io_service_bytes_recursive: Option<Vec<BlkioEntry>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BlkioEntry {
    #[serde(default)]
    pub op: String,
    #[serde(default)]
    pub value: i64,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct PidsStats {
    #[serde(default)]
    pub current: i64,
    #[serde(default)]
    pub limit: Option<i64>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct NetStats {
    #[serde(default, rename = "rx_bytes")]
    pub rx_bytes: i64,
    #[serde(default, rename = "tx_bytes")]
    pub tx_bytes: i64,
}

impl StatsSample {
    /// CPU percent between two consecutive samples, Docker-CLI style:
    /// Δcontainer / Δsystem × online_cpus × 100. Falls back to the cgroup v2
    /// `cpu` block when the v1-style `cpu_stats` fields are absent.
    pub fn cpu_percent(prev: Option<&StatsSample>, cur: &StatsSample) -> Option<f64> {
        let prev = prev?;
        let dcpu = cur.cpu_total() - prev.cpu_total();
        let dsys = cur.system_cpu_usage()? - prev.system_cpu_usage()?;
        if dcpu < 0 || dsys <= 0 {
            return None;
        }
        let online = cur.online_cpus().unwrap_or(1) as f64;
        Some((dcpu as f64 / dsys as f64) * online * 100.0)
    }

    fn cpu_total(&self) -> i64 {
        if self.cpu_stats.cpu_usage.total_usage > 0 {
            return self.cpu_stats.cpu_usage.total_usage;
        }
        self.cpu.as_ref().map(|c| c.usage.total).unwrap_or(0)
    }

    fn system_cpu_usage(&self) -> Option<i64> {
        self.cpu_stats
            .system_cpu_usage
            .or_else(|| self.cpu.as_ref().and_then(|c| c.system_cpu_usage))
    }

    fn online_cpus(&self) -> Option<usize> {
        self.cpu_stats
            .online_cpus
            .or_else(|| self.cpu.as_ref().and_then(|c| c.online_cpus))
    }

    pub fn mem_usage(&self) -> i64 {
        if self.memory_stats.usage > 0 {
            return self.memory_stats.usage;
        }
        self.memory.as_ref().map(|m| m.usage).unwrap_or(0)
    }

    pub fn mem_limit(&self) -> i64 {
        if self.memory_stats.limit > 0 {
            return self.memory_stats.limit;
        }
        self.memory.as_ref().map(|m| m.limit).unwrap_or(0)
    }

    pub fn mem_percent(&self) -> Option<f64> {
        let limit = self.mem_limit();
        if limit > 0 {
            Some(self.mem_usage() as f64 / limit as f64 * 100.0)
        } else {
            None
        }
    }

    /// Disk read/write bytes for the whole lifetime of the container.
    pub fn blk_read(&self) -> i64 {
        let v1: i64 = self
            .blkio_stats
            .io_service_bytes_recursive
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .filter(|e| e.op.eq_ignore_ascii_case("read"))
            .map(|e| e.value)
            .sum();
        if v1 > 0 {
            return v1;
        }
        self.blkio
            .as_ref()
            .map(|b| {
                b.io_service_bytes_recursive
                    .iter()
                    .filter(|e| e.op.eq_ignore_ascii_case("read"))
                    .map(|e| e.value)
                    .sum()
            })
            .unwrap_or(0)
    }

    pub fn blk_write(&self) -> i64 {
        let v1: i64 = self
            .blkio_stats
            .io_service_bytes_recursive
            .as_deref()
            .unwrap_or(&[])
            .iter()
            .filter(|e| e.op.eq_ignore_ascii_case("write"))
            .map(|e| e.value)
            .sum();
        if v1 > 0 {
            return v1;
        }
        self.blkio
            .as_ref()
            .map(|b| {
                b.io_service_bytes_recursive
                    .iter()
                    .filter(|e| e.op.eq_ignore_ascii_case("write"))
                    .map(|e| e.value)
                    .sum()
            })
            .unwrap_or(0)
    }

    pub fn net_rx(&self) -> i64 {
        self.networks.values().map(|n| n.rx_bytes).sum()
    }

    pub fn net_tx(&self) -> i64 {
        self.networks.values().map(|n| n.tx_bytes).sum()
    }
}

/// Daemon system info (`GET /info`) — the fields the dashboard needs.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SystemInfo {
    #[serde(rename = "NCPU")]
    pub ncpu: usize,
    #[serde(rename = "MemTotal")]
    pub mem_total: i64,
    #[serde(default, rename = "Containers")]
    pub containers: i64,
    #[serde(default, rename = "ContainersRunning")]
    pub containers_running: i64,
    #[serde(default, rename = "ContainersPaused")]
    pub containers_paused: i64,
    #[serde(default, rename = "ContainersStopped")]
    pub containers_stopped: i64,
    #[serde(default, rename = "Images")]
    pub images: i64,
    #[serde(default, rename = "Driver")]
    pub driver: String,
    #[serde(default, rename = "ServerVersion")]
    pub server_version: String,
    #[serde(default, rename = "DockerRootDir")]
    pub docker_root_dir: String,
    #[serde(default, rename = "OperatingSystem")]
    pub operating_system: String,
}

/// Disk-usage summary (`GET /system/df`), the aggregate counters only.
/// The daemon emits these keys in PascalCase, so field renames are explicit.
#[derive(Debug, Clone, Deserialize)]
pub struct DiskSummary {
    #[serde(default, rename = "LayersSize")]
    pub layers_size: i64,
    #[serde(default, rename = "ImageUsage")]
    pub image_usage: Option<UsageTotals>,
    #[serde(default, rename = "ContainerUsage")]
    pub container_usage: Option<UsageTotals>,
    #[serde(default, rename = "VolumeUsage")]
    pub volume_usage: Option<UsageTotals>,
    #[serde(default, rename = "BuildCacheUsage")]
    pub build_cache_usage: Option<UsageTotals>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct UsageTotals {
    // Current daemons report `TotalSize`; keep `Size` for older ones.
    #[serde(default, rename = "TotalSize", alias = "Size")]
    pub total_size: i64,
    #[serde(default, rename = "Reclaimable")]
    pub reclaimable: i64,
    #[serde(default, rename = "TotalCount")]
    pub total_count: i64,
    #[serde(default, rename = "ActiveCount")]
    pub active_count: i64,
}

/// A lifecycle event from `GET /events` (container type only).
/// Docker 29 emits `time`/`timeNano` in lowercase; older daemons used
/// `Time`/`TimeNano`, so both spellings are accepted.
#[derive(Debug, Clone, Deserialize)]
pub struct EventMessage {
    #[serde(rename = "Type")]
    pub typ: String,
    #[serde(rename = "Action")]
    pub action: String,
    #[serde(rename = "time", alias = "Time", default)]
    pub time: i64,
    #[serde(rename = "timeNano", alias = "TimeNano", default)]
    pub time_nano: i64,
    #[serde(rename = "Actor")]
    pub actor: EventActor,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EventActor {
    #[serde(rename = "ID")]
    pub id: String,
    #[serde(default, rename = "Attributes")]
    pub attributes: std::collections::BTreeMap<String, String>,
}

impl EventMessage {
    pub fn container_name(&self) -> String {
        self.actor
            .attributes
            .get("name")
            .cloned()
            .unwrap_or_else(|| self.actor.id[..12.min(self.actor.id.len())].to_string())
    }
}
