//! Graphical desktop front-end (egui/eframe).
//!
//! Same engine facade as the TUI: all Docker calls run on the tokio runtime
//! and report back through an mpsc channel, so the window never blocks.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use egui::{
    Color32, FontData, FontFamily, FontId, Frame, Margin, Pos2, Rect, RichText, ScrollArea, Sense,
    Stroke, TextStyle, Ui, UiBuilder, Vec2,
};

use tokio::sync::mpsc;

use crate::agents::{Activity, AgentScanner, AgentSession, Step, StepKind};
use crate::docker::{
    self, ContainerSummary, DiskSummary, Docker, DockerError, EventMessage, ImageSummary, LogLine,
    StatsSample, StreamKind, SystemInfo,
};
use crate::hw::{HwSample, Sampler};
use crate::hw_inventory as hwinv;
use crate::hw_inventory::{gpu_stats, gpu_stats_hwmon_only, GpuStat};
use crate::icons::{self as iconsmod, TexCache};
use crate::ports::{self as portsmod, PortEntry};
use crate::procs::{self as procsmod, ProcInfo, ProcScanner, SortCol};
use crate::strata;

/// One page of container logs held by the UI.
const LOG_CAP: usize = 5_000;
/// How many lines to pull when first opening a container's logs.
const LOG_TAIL: usize = 1_000;
/// Lines per container when refreshing the consolidated Logs tab.
const ALL_LOG_TAIL: usize = 200;
/// History kept per running container for the dashboard charts.
const HIST: usize = 60;

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
pub enum Tab {
    Dashboard,
    Containers,
    Images,
    Volumes,
    Agents,
    Hardware,
    Processes,
    Ports,
    Strata,
    Logs,
    Settings,
}

/// Sidebar state filter for the container list.
#[derive(PartialEq, Eq, Clone, Copy)]
enum StateFilter {
    All,
    Running,
    Stopped,
}

/// Protocol filter for the Ports tab.
#[derive(PartialEq, Eq, Clone, Copy)]
enum ProtoFilter {
    All,
    Tcp,
    Udp,
}

#[derive(PartialEq, Eq, Clone, Copy)]
enum Pending {
    Start,
    Stop,
    Restart,
    Remove,
}

impl Pending {
    fn verb(self) -> &'static str {
        match self {
            Pending::Start => "start",
            Pending::Stop => "stop",
            Pending::Restart => "restart",
            Pending::Remove => "remove",
        }
    }
}

enum Msg {
    Refreshed(Result<RefreshData, String>),
    LogStart(u64, Result<(), String>),
    LogLines(u64, Vec<LogLine>),
    /// (gen, container id, result) — the id lets the consolidated page drop
    /// just this follower instead of marking the whole page stopped.
    LogDone(u64, String, Result<(), String>),
    ActionDone(Result<(&'static str, String), String>),
    Info(Result<SystemInfo, String>),
    Df(Result<DiskSummary, String>),
    Stats(String, Box<StatsSample>),
    Events(Result<Vec<EventMessage>, String>),
    /// The event stream closed normally (watchdog recycle or daemon hang-up);
    /// reconnect quietly, nothing to report.
    EventsClosed,
    AllLogs(Vec<LogLine>),
    AgentsScanned(Vec<AgentSession>),
    PortsScanned(Vec<PortEntry>),
    ProcsScanned(Vec<crate::procs::ProcInfo>),
    /// an app icon finished decoding on the blocking pool; `epoch` drops
    /// stale results from before a theme/config change
    IconReady {
        name: String,
        epoch: u64,
        rgba: Option<iconsmod::IconRgba>,
    },
    GpusScanned(Vec<GpuStat>),
    StrataHealth(Result<strata::Health, String>),
    StrataMetrics(Result<Box<strata::Metrics>, String>),
}

struct RefreshData {
    containers: Result<Vec<ContainerSummary>, String>,
    images: Result<Vec<ImageSummary>, String>,
    volumes: Result<Vec<docker::VolumeEntry>, String>,
}

/// Rolling per-container stats history for the charts.
#[derive(Default)]
struct StatHist {
    cpu: VecDeque<f64>,
    mem: VecDeque<i64>,
    rx: VecDeque<i64>,
    tx: VecDeque<i64>,
    blk_r: VecDeque<i64>,
    blk_w: VecDeque<i64>,
}

impl StatHist {
    fn push(&mut self, s: &StatsSample, prev: Option<&StatsSample>) {
        let cpu = StatsSample::cpu_percent(prev, s).unwrap_or(0.0);
        push(&mut self.cpu, cpu);
        push(&mut self.mem, s.mem_usage());
        // Net/blk totals reset when a container restarts; a negative delta
        // means the counter wrapped, so restart the series instead.
        let d_rx = delta(prev.map(|p| p.net_rx()), s.net_rx());
        let d_tx = delta(prev.map(|p| p.net_tx()), s.net_tx());
        let d_br = delta(prev.map(|p| p.blk_read()), s.blk_read());
        let d_bw = delta(prev.map(|p| p.blk_write()), s.blk_write());
        if prev.is_some() && (s.net_rx() < prev.map(|p| p.net_rx()).unwrap_or(0)) {
            self.rx.clear();
            self.tx.clear();
            self.blk_r.clear();
            self.blk_w.clear();
        }
        push(&mut self.rx, d_rx);
        push(&mut self.tx, d_tx);
        push(&mut self.blk_r, d_br);
        push(&mut self.blk_w, d_bw);
    }
}

fn push<T: Clone>(q: &mut VecDeque<T>, v: T) {
    q.push_back(v);
    if q.len() > HIST {
        q.pop_front();
    }
}

fn delta(prev: Option<i64>, cur: i64) -> i64 {
    match prev {
        Some(p) if cur >= p => cur - p,
        _ => 0,
    }
}

struct Gui {
    tab: Tab,
    containers: Vec<ContainerSummary>,
    selected: Option<usize>,
    state_filter: StateFilter,
    images: Vec<ImageSummary>,
    volumes: Vec<docker::VolumeEntry>,
    logs: Vec<LogLine>,
    logs_follow: bool,
    logs_container: Option<String>,
    logs_container_name: String,
    logs_active: bool,
    logs_gen: u64,
    pending_logs_gen: Option<u64>,
    confirm_remove: Option<String>,
    pending_action: Option<(String, Pending)>,
    status: String,
    error: Option<String>,
    filter: String,
    /// substring filter on the log page
    log_filter: String,
    selected_image: Option<String>,
    selected_volume: Option<String>,
    last_refresh: String,
    daemon_ok: bool,
    // --- dashboard data ---
    info: Option<SystemInfo>,
    df: Option<DiskSummary>,
    events: Vec<EventMessage>,
    events_active: bool,
    stats: HashMap<String, StatsSample>,
    prev_stats: HashMap<String, StatsSample>,
    hist: HashMap<String, StatHist>,
    stats_gen: u64,
    // --- agents tab ---
    agents: Vec<AgentSession>,
    /// lead pid of the selected session (indexes shift between scans)
    selected_agent: Option<u32>,
    agent_filter: String,
    // --- hardware tab ---
    hw: Option<HwSample>,
    hw_cpu_hist: VecDeque<f64>,
    hw_mem_hist: VecDeque<f64>,
    hw_swap_hist: VecDeque<f64>,
    hw_disk_r_hist: VecDeque<f64>,
    hw_disk_w_hist: VecDeque<f64>,
    hw_net_rx_hist: VecDeque<f64>,
    hw_net_tx_hist: VecDeque<f64>,
    hw_gpu_busy_hist: VecDeque<f64>,
    hw_gpu_vram_hist: VecDeque<f64>,
    hw_cpu_temp_hist: VecDeque<f64>,
    hw_gpu_temp_hist: VecDeque<f64>,
    hw_disk_temp_hist: VecDeque<f64>,
    /// live per-GPU stats (busy, VRAM, temp, power)
    gpus: Vec<GpuStat>,
    // --- ports tab ---
    ports: Vec<PortEntry>,
    /// (proto, addr, port) of the selected socket
    selected_port: Option<(String, String, u16)>,
    port_filter: String,
    /// only sockets with a discoverable owning process
    ports_owned_only: bool,
    ports_proto: ProtoFilter,
    /// Some(4) / Some(6) to show one address family only
    ports_family: Option<u8>,
    /// hide established connections, keep listeners and bound udp
    ports_listening_only: bool,
    /// only sockets bound beyond loopback (reachable from the network)
    ports_exposed_only: bool,
    // --- processes tab ---
    procs: Vec<ProcInfo>,
    proc_filter: String,
    proc_sort: SortCol,
    proc_sort_desc: bool,
    /// pid of the selected process row (None = none)
    selected_proc: Option<u32>,
    /// icon name of the app group currently expanded in the table (None = none)
    proc_group: Option<String>,
    /// decoded application icons for process rows (UI thread only)
    icons: TexCache,
    /// icon names already queued for decode on the blocking pool
    icon_requests: std::collections::HashSet<String>,
    // --- strata tab ---
    strata_cfg: strata::StrataConfig,
    /// endpoint text being typed; applied on Enter / focus loss. Shared by
    /// the Settings tab and the Strata tab, so both stay in sync.
    strata_endpoint_edit: String,
    strata_key_edit: String,
    strata_key_masked: bool,
    /// general settings, edited on the Settings tab
    refresh_edit: f64,
    autoscroll_logs: bool,
    /// test seam: the Settings tab records the endpoint field's rect here
    #[cfg(test)]
    probe_endpoint_rect: Option<egui::Rect>,
    /// test seam: the Settings tab records the Test & save button's rect here
    #[cfg(test)]
    probe_save_rect: Option<egui::Rect>,
    strata_health: Option<strata::Health>,
    strata_metrics: Option<Box<strata::Metrics>>,
    strata_error: Option<String>,
    /// recent request table: the last 12, or every one the server kept
    strata_show_all: bool,
}

impl Gui {
    fn new() -> Self {
        Self {
            tab: Tab::Dashboard,
            containers: Vec::new(),
            selected: None,
            state_filter: StateFilter::All,
            images: Vec::new(),
            volumes: Vec::new(),
            logs: Vec::new(),
            logs_follow: true,
            logs_container: None,
            logs_container_name: String::new(),
            logs_active: false,
            logs_gen: 0,
            pending_logs_gen: None,
            confirm_remove: None,
            pending_action: None,
            status: String::new(),
            error: None,
            filter: String::new(),
            log_filter: String::new(),
            selected_image: None,
            selected_volume: None,
            last_refresh: String::new(),
            daemon_ok: false,
            info: None,
            df: None,
            events: Vec::new(),
            events_active: false,
            stats: HashMap::new(),
            prev_stats: HashMap::new(),
            hist: HashMap::new(),
            stats_gen: 0,
            agents: Vec::new(),
            selected_agent: None,
            agent_filter: String::new(),
            hw: None,
            hw_cpu_hist: VecDeque::new(),
            hw_mem_hist: VecDeque::new(),
            hw_swap_hist: VecDeque::new(),
            hw_disk_r_hist: VecDeque::new(),
            hw_disk_w_hist: VecDeque::new(),
            hw_net_rx_hist: VecDeque::new(),
            hw_net_tx_hist: VecDeque::new(),
            hw_gpu_busy_hist: VecDeque::new(),
            hw_gpu_vram_hist: VecDeque::new(),
            hw_cpu_temp_hist: VecDeque::new(),
            hw_gpu_temp_hist: VecDeque::new(),
            hw_disk_temp_hist: VecDeque::new(),
            gpus: Vec::new(),
            ports: Vec::new(),
            selected_port: None,
            port_filter: String::new(),
            ports_owned_only: false,
            ports_proto: ProtoFilter::All,
            ports_family: None,
            ports_listening_only: false,
            ports_exposed_only: false,
            procs: Vec::new(),
            proc_filter: String::new(),
            proc_sort: SortCol::Mem,
            proc_sort_desc: true,
            selected_proc: None,
            proc_group: None,
            icons: TexCache::new(),
            icon_requests: std::collections::HashSet::new(),
            strata_cfg: strata::StrataConfig::default(),
            strata_endpoint_edit: String::new(),
            strata_key_edit: String::new(),
            strata_key_masked: true,
            refresh_edit: 2.0,
            autoscroll_logs: true,
            #[cfg(test)]
            probe_endpoint_rect: None,
            #[cfg(test)]
            probe_save_rect: None,
            strata_health: None,
            strata_metrics: None,
            strata_error: None,
            strata_show_all: false,
        }
    }

    fn filtered(&self) -> Vec<(usize, &ContainerSummary)> {
        let f = self.filter.to_lowercase();
        self.containers
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                let state_ok = match self.state_filter {
                    StateFilter::All => true,
                    StateFilter::Running => c.state == "running",
                    StateFilter::Stopped => c.state != "running",
                };
                state_ok
                    && (f.is_empty()
                        || c.names.join(",").to_lowercase().contains(&f)
                        || c.image.to_lowercase().contains(&f)
                        || c.state.to_lowercase().contains(&f))
            })
            .collect()
    }

    fn selected_container(&self) -> Option<&ContainerSummary> {
        let list = self.filtered();
        list.into_iter()
            .find(|(i, _)| Some(*i) == self.selected)
            .map(|(_, c)| c)
    }

    /// Sessions matching the Agents-tab filter, with their agents-list index.
    fn filtered_agents(&self) -> Vec<(usize, &AgentSession)> {
        let f = self.agent_filter.to_lowercase();
        self.agents
            .iter()
            .enumerate()
            .filter(|(_, a)| {
                f.is_empty()
                    || a.agent.to_lowercase().contains(&f)
                    || a.vendor.to_lowercase().contains(&f)
                    || a.cwd
                        .as_deref()
                        .map(|c| c.to_lowercase())
                        .unwrap_or_default()
                        .contains(&f)
            })
            .collect()
    }

    /// Sockets matching the Ports-tab filter, with their ports-list index.
    fn filtered_ports(&self) -> Vec<(usize, &PortEntry)> {
        let f = self.port_filter.to_lowercase();
        self.ports
            .iter()
            .enumerate()
            .filter(|(_, p)| {
                if self.ports_owned_only && p.pid.is_none() {
                    return false;
                }
                match self.ports_proto {
                    ProtoFilter::Tcp if p.proto != "tcp" => return false,
                    ProtoFilter::Udp if p.proto != "udp" => return false,
                    _ => {}
                }
                if self.ports_family.is_some_and(|fam| fam != p.family) {
                    return false;
                }
                if self.ports_listening_only && p.state == "ESTABLISHED" {
                    return false;
                }
                if self.ports_exposed_only && p.is_loopback() {
                    return false;
                }
                f.is_empty()
                    || p.proto.contains(&f)
                    || p.addr.contains(&f)
                    || p.port.to_string().contains(&f)
                    || p.proc_name
                        .as_deref()
                        .map(|n| n.to_lowercase())
                        .unwrap_or_default()
                        .contains(&f)
                    || p.state.to_lowercase().contains(&f)
            })
            .collect()
    }

    /// Containers whose published host port matches `port` (the cross-link
    /// from a socket to the container that forwards it).
    fn containers_on_port(&self, port: u16) -> Vec<String> {
        self.containers
            .iter()
            .filter(|c| {
                c.ports
                    .iter()
                    .any(|p| p.public_port == Some(port) || p.private_port == Some(port))
            })
            .map(|c| c.short_name())
            .collect()
    }

    fn push_hw(&mut self, s: HwSample, gpus: Vec<GpuStat>) {
        let push = |q: &mut VecDeque<f64>, v: f64| {
            q.push_back(v);
            while q.len() > HIST {
                q.pop_front();
            }
        };
        let push_opt = |q: &mut VecDeque<f64>, v: Option<f64>| {
            if let Some(v) = v {
                push(q, v);
            }
        };
        push(&mut self.hw_cpu_hist, s.cpu_pct);
        push(&mut self.hw_mem_hist, (s.mem_total - s.mem_avail) as f64);
        push(
            &mut self.hw_swap_hist,
            s.swap_total.saturating_sub(s.swap_free) as f64,
        );
        push(&mut self.hw_disk_r_hist, s.disk_read_bps);
        push(&mut self.hw_disk_w_hist, s.disk_write_bps);
        push(&mut self.hw_net_rx_hist, s.net_rx_bps);
        push(&mut self.hw_net_tx_hist, s.net_tx_bps);
        push_opt(&mut self.hw_cpu_temp_hist, s.cpu_temp_c);
        // GPU aggregates: busiest GPU busy%, most-used VRAM across GPUs,
        // hottest GPU/SSD sensor — one row each regardless of GPU count.
        let busy = gpus
            .iter()
            .filter_map(|g| g.busy_pct)
            .fold(None::<f64>, |a, b| Some(a.map_or(b, |a| a.max(b))));
        let vram = gpus
            .iter()
            .map(|g| g.vram_used as f64)
            .fold(0.0_f64, f64::max);
        let gpu_temp = gpus
            .iter()
            .filter_map(|g| g.temp_c)
            .fold(None::<f64>, |a, b| Some(a.map_or(b, |a| a.max(b))));
        let disk_temp = s
            .disks
            .iter()
            .filter_map(|d| d.temp_c)
            .fold(None::<f64>, |a, b| Some(a.map_or(b, |a| a.max(b))));
        push_opt(&mut self.hw_gpu_busy_hist, busy);
        if vram > 0.0 {
            push(&mut self.hw_gpu_vram_hist, vram);
        }
        push_opt(&mut self.hw_gpu_temp_hist, gpu_temp);
        push_opt(&mut self.hw_disk_temp_hist, disk_temp);
        self.gpus = gpus;
        self.hw = Some(s);
    }

    fn push_logs(&mut self, lines: Vec<LogLine>) {
        if lines.is_empty() {
            return;
        }
        self.logs.extend(lines);
        if self.logs.len() > LOG_CAP {
            let excess = self.logs.len() - LOG_CAP;
            self.logs.drain(..excess);
        }
    }

    /// Consolidated pages re-sort by the leading RFC3339 timestamp so the
    /// per-container tails interleave chronologically.
    fn sort_all_logs(&mut self) {
        if self.logs_container.as_deref() != Some("__all__") {
            return;
        }
        self.logs.sort_by(|a, b| ts_of(&a.text).cmp(ts_of(&b.text)));
    }
}

/// egui application wired to a tokio runtime.
struct SysKamlingApp {
    gui: Gui,
    docker: Arc<Docker>,
    tx: mpsc::UnboundedSender<Msg>,
    rx: mpsc::UnboundedReceiver<Msg>,
    rt: tokio::runtime::Handle,
    /// strong ref so the runtime lives as long as the window
    _rt_keepalive: Arc<tokio::runtime::Runtime>,
    refresh: f64,
    last_refresh: Instant,
    tick: Instant,
    // --- micro animations ---
    /// per-container-id eased 0..1 for the status dot pulse
    dot_anim: HashMap<String, f32>,
    /// container id currently running a spin animation, and its start time
    spin: Option<(String, Instant)>,
    /// when the current logs page opened (fade-in)
    logs_opened: Instant,
    /// number of log lines at last paint, to flash freshly arrived lines
    prev_log_len: usize,
    // --- dashboard timers ---
    last_info: Instant,
    last_df: Instant,
    /// throttle for (re)opening the daemon event stream
    last_events: Instant,
    /// last time the event stream actually delivered a lifecycle event
    last_event: Instant,
    /// container ids with a live stats stream task
    streaming_stats: Vec<String>,
    /// last sign of life per stats stream: spawn time, then each sample
    stats_spawned: HashMap<String, Instant>,
    /// container ids with a live log-follow task feeding the consolidated page
    all_logs_follow: Vec<String>,
    // --- chrome ---
    /// last successful container/image/volume refresh
    last_sync: Option<Instant>,
    /// status text currently shown as a toast, and when it appeared
    toast: (String, Instant),
    // --- agents tab ---
    /// when the scanner task was last kicked
    last_agent_scan: Instant,
    agent_scan_busy: Arc<Mutex<bool>>,
    /// kept across scans: CPU% needs the previous sample, transcripts are cached
    agent_scanner: Arc<Mutex<AgentScanner>>,
    // --- hardware tab ---
    sampler: Sampler,
    last_hw_sample: Instant,
    // --- ports tab ---
    last_port_scan: Instant,
    port_scan_busy: Arc<Mutex<bool>>,
    /// scanner lives behind a mutex so consecutive blocking scans keep
    /// their tick counters and CPU% stays a real delta
    proc_holder: Arc<Mutex<ProcScanner>>,
    /// seed scan already ran for the current tab (screenshot harness sets
    /// the tab before the first update, so on_tab_change never fires)
    proc_seed_done: bool,
    last_proc_scan: Instant,
    proc_scan_busy: Arc<Mutex<bool>>,
    last_gpu_stats: Instant,
    gpu_scan_busy: Arc<Mutex<bool>>,
    // --- strata tab ---
    /// HTTP client rebuilt whenever the endpoint/key change
    strata_client: Arc<strata::HttpClient>,
    last_strata_poll: Instant,
    strata_poll_busy: Arc<Mutex<bool>>,
}

impl SysKamlingApp {
    fn new(
        cc: &eframe::CreationContext<'_>,
        rt: Arc<tokio::runtime::Runtime>,
        socket: &str,
        refresh: f64,
    ) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        apply_theme(&cc.egui_ctx);
        let file_cfg = strata::load_file();
        let strata_cfg = file_cfg.strata.clone();
        let mut gui = Gui::new();
        gui.strata_cfg = strata_cfg.clone();
        gui.strata_endpoint_edit = strata_cfg.endpoint.clone();
        gui.strata_key_edit = strata_cfg.api_key.clone();
        // a saved refresh interval overrides the CLI default
        let mut refresh = refresh;
        if file_cfg.general.refresh_secs > 0.0 {
            refresh = file_cfg.general.refresh_secs;
        }
        gui.refresh_edit = refresh;
        gui.autoscroll_logs = file_cfg.general.autoscroll_logs;
        gui.logs_follow = file_cfg.general.autoscroll_logs;
        let app = Self {
            gui,
            docker: Arc::new(Docker::new(socket)),
            tx,
            rx,
            rt: rt.handle().clone(),
            _rt_keepalive: Arc::clone(&rt),
            refresh,
            last_refresh: Instant::now() - Duration::from_secs(999),
            tick: Instant::now(),
            dot_anim: Default::default(),
            spin: None,
            logs_opened: Instant::now(),
            prev_log_len: 0,
            last_info: Instant::now() - Duration::from_secs(999),
            last_df: Instant::now() - Duration::from_secs(999),
            last_events: Instant::now() - Duration::from_secs(999),
            last_event: Instant::now(),
            streaming_stats: Vec::new(),
            stats_spawned: HashMap::new(),
            all_logs_follow: Vec::new(),
            last_sync: None,
            toast: (String::new(), Instant::now()),
            last_agent_scan: Instant::now() - Duration::from_secs(999),
            agent_scan_busy: Arc::new(Mutex::new(false)),
            agent_scanner: Arc::new(Mutex::new(AgentScanner::default())),
            sampler: Sampler::default(),
            last_hw_sample: Instant::now() - Duration::from_secs(999),
            last_port_scan: Instant::now() - Duration::from_secs(999),
            port_scan_busy: Arc::new(Mutex::new(false)),
            proc_holder: Arc::new(Mutex::new(ProcScanner::new())),
            proc_seed_done: false,
            last_proc_scan: Instant::now() - Duration::from_secs(999),
            proc_scan_busy: Arc::new(Mutex::new(false)),
            last_gpu_stats: Instant::now() - Duration::from_secs(999),
            gpu_scan_busy: Arc::new(Mutex::new(false)),
            strata_client: Arc::new(strata::HttpClient::new(&strata_cfg)),
            last_strata_poll: Instant::now() - Duration::from_secs(999),
            strata_poll_busy: Arc::new(Mutex::new(false)),
        };
        app
    }

    /// Test-only: render the Hardware tab into an offscreen egui context
    /// sized like a short window, then return the ScrollArea geometry the
    /// view records (viewport height, content height, thumb height).
    #[cfg(test)]
    fn new_offscreen(
        ctx: &egui::Context,
        tx: mpsc::UnboundedSender<Msg>,
        rx: mpsc::UnboundedReceiver<Msg>,
        rt: Arc<tokio::runtime::Runtime>,
    ) -> SysKamlingApp {
        apply_theme(ctx);
        let app = SysKamlingApp {
            gui: Gui::new(),
            docker: Arc::new(Docker::new("/nonexistent.sock")),
            tx,
            rx,
            rt: rt.handle().clone(),
            _rt_keepalive: Arc::clone(&rt),
            refresh: 1.0,
            last_refresh: Instant::now() - Duration::from_secs(999),
            tick: Instant::now(),
            dot_anim: Default::default(),
            spin: None,
            logs_opened: Instant::now(),
            prev_log_len: 0,
            last_info: Instant::now() - Duration::from_secs(999),
            last_df: Instant::now() - Duration::from_secs(999),
            last_events: Instant::now() - Duration::from_secs(999),
            last_event: Instant::now(),
            streaming_stats: Vec::new(),
            stats_spawned: HashMap::new(),
            all_logs_follow: Vec::new(),
            last_sync: None,
            toast: (String::new(), Instant::now()),
            last_agent_scan: Instant::now() - Duration::from_secs(999),
            agent_scan_busy: Arc::new(Mutex::new(false)),
            agent_scanner: Arc::new(Mutex::new(AgentScanner::default())),
            sampler: Sampler::default(),
            last_hw_sample: Instant::now() - Duration::from_secs(999),
            last_port_scan: Instant::now() - Duration::from_secs(999),
            port_scan_busy: Arc::new(Mutex::new(false)),
            proc_holder: Arc::new(Mutex::new(ProcScanner::new())),
            proc_seed_done: false,
            last_proc_scan: Instant::now() - Duration::from_secs(999),
            proc_scan_busy: Arc::new(Mutex::new(false)),
            last_gpu_stats: Instant::now() - Duration::from_secs(999),
            gpu_scan_busy: Arc::new(Mutex::new(false)),
            strata_client: Arc::new(strata::HttpClient::new(&strata::StrataConfig::default())),
            last_strata_poll: Instant::now() - Duration::from_secs(999),
            strata_poll_busy: Arc::new(Mutex::new(false)),
        };
        app
    }

    #[cfg(test)]
    fn probe_hardware_scroll(size: egui::Vec2) -> (f32, f32) {
        let ctx = egui::Context::default();
        let (tx, rx) = mpsc::unbounded_channel();
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
        );
        let mut app = Self::new_offscreen(&ctx, tx, rx, rt);
        app.gui.tab = Tab::Hardware;
        app.sample_hardware();
        app.sample_hardware(); // second sample so rates/sparklines populate
        let mut rich = app.gui.hw.clone().unwrap();
        // several inventoried drives stack up cards; fixture names keep the
        // probe identical on any runner (a CI box may inventory zero disks)
        rich.disks = (0..8)
            .map(|i| crate::hw::DiskStat {
                name: format!("testdisk{}", i % 3),
                reads: 10,
                writes: 20,
                total_read: 1000,
                total_write: 2000,
                ms_weighted: 3,
                uptime: 3600.0,
                temp_c: Some(40.0),
                read_bps: 0.0,
                write_bps: 0.0,
            })
            .collect();
        app.gui.hw = Some(rich);
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(egui::Pos2::ZERO, size));
        hw_scroll_probe::reset();
        let inv = crate::hw_inventory::fixture_inventory();
        let usage = [crate::hw_inventory::DiskUsage {
            kern: "testdisk0".into(),
            mount: "/".into(),
            total: 1000,
            used: 400,
        }];
        for _ in 0..3 {
            ctx.begin_pass(raw.clone());
            egui::CentralPanel::default().show(&ctx, |ui| {
                if app.gui.tab == Tab::Hardware {
                    app.hardware_view(ui, &inv, &usage)
                }
            });
            let _ = ctx.end_pass();
        }
        hw_scroll_probe::last()
    }
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

// Near-black canvas with layered surfaces and one blue accent. Green, amber
// and red are kept for state (running, warnings, errors) so they keep their
// meaning across the app.
const BG: Color32 = Color32::from_rgb(11, 13, 18);
const RAIL: Color32 = Color32::from_rgb(15, 18, 24);
const SURFACE: Color32 = Color32::from_rgb(20, 24, 32);
const SURFACE_HI: Color32 = Color32::from_rgb(28, 33, 43);
const ELEVATED: Color32 = Color32::from_rgb(38, 45, 59);
const WELL: Color32 = Color32::from_rgb(9, 11, 15);
const BORDER: Color32 = Color32::from_rgb(35, 41, 53);
const TEXT: Color32 = Color32::from_rgb(230, 233, 239);
const MUTED: Color32 = Color32::from_rgb(140, 148, 166);
const FAINT: Color32 = Color32::from_rgb(96, 104, 123);
const ACCENT: Color32 = Color32::from_rgb(88, 148, 255);
const GREEN: Color32 = Color32::from_rgb(52, 211, 153);
const AMBER: Color32 = Color32::from_rgb(245, 183, 64);
const RED: Color32 = Color32::from_rgb(244, 112, 112);
const VIOLET: Color32 = Color32::from_rgb(167, 139, 250);
const CYAN: Color32 = Color32::from_rgb(56, 205, 230);
const ORANGE: Color32 = Color32::from_rgb(244, 140, 90);

/// Heading face (a real bold font, not egui's color-only "strong").
fn bold(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name("bold".into()))
}

fn apply_theme(ctx: &egui::Context) {
    install_fonts(ctx);
    let mut style = (*ctx.style()).clone();

    let v = &mut style.visuals;
    v.dark_mode = true;
    v.panel_fill = BG;
    v.window_fill = SURFACE;
    v.window_stroke = Stroke::new(1.0_f32, BORDER);
    v.extreme_bg_color = SURFACE;
    v.faint_bg_color = SURFACE;
    v.code_bg_color = SURFACE_HI;
    v.override_text_color = None;
    v.weak_text_color = Some(MUTED);
    v.hyperlink_color = ACCENT;
    v.window_corner_radius = egui::CornerRadius::same(14);
    v.menu_corner_radius = egui::CornerRadius::same(10);
    v.selection.bg_fill = ACCENT.gamma_multiply(0.35);
    v.selection.stroke = Stroke::new(1.0_f32, ACCENT);
    v.text_cursor.stroke = Stroke::new(2.0_f32, ACCENT);
    v.window_shadow = egui::epaint::Shadow {
        offset: [0, 12],
        blur: 32,
        spread: 0,
        color: Color32::from_black_alpha(160),
    };
    v.popup_shadow = egui::epaint::Shadow {
        offset: [0, 6],
        blur: 16,
        spread: 0,
        color: Color32::from_black_alpha(120),
    };

    let w = &mut v.widgets;
    w.noninteractive.bg_fill = SURFACE;
    w.noninteractive.weak_bg_fill = SURFACE;
    w.noninteractive.bg_stroke = Stroke::new(1.0_f32, BORDER);
    w.noninteractive.fg_stroke = Stroke::new(1.0_f32, TEXT);
    w.inactive.bg_fill = SURFACE_HI;
    w.inactive.weak_bg_fill = SURFACE_HI;
    w.inactive.bg_stroke = Stroke::new(1.0_f32, BORDER);
    w.inactive.fg_stroke = Stroke::new(1.0_f32, TEXT);
    w.hovered.bg_fill = ELEVATED;
    w.hovered.weak_bg_fill = ELEVATED;
    w.hovered.bg_stroke = Stroke::new(1.0_f32, Color32::from_rgb(56, 64, 80));
    w.hovered.fg_stroke = Stroke::new(1.0_f32, TEXT);
    w.active.bg_fill = Color32::from_rgb(46, 54, 70);
    w.active.weak_bg_fill = Color32::from_rgb(46, 54, 70);
    w.active.bg_stroke = Stroke::new(1.0_f32, ACCENT);
    w.active.fg_stroke = Stroke::new(1.0_f32, TEXT);
    w.open = w.active;
    for s in [
        &mut w.noninteractive,
        &mut w.inactive,
        &mut w.hovered,
        &mut w.active,
        &mut w.open,
    ] {
        s.corner_radius = egui::CornerRadius::same(8);
        s.expansion = 0.0;
    }

    // egui's built-in open/close animations (button press, window fade) —
    // cheap micro-interactions that come free with repaints.
    style.animation_time = 0.16;

    style.spacing.item_spacing = Vec2::new(8.0, 8.0);
    style.spacing.button_padding = Vec2::new(14.0, 7.0);
    style.spacing.interact_size.y = 30.0;
    style.spacing.scroll = egui::style::ScrollStyle::floating();

    style.text_styles.insert(TextStyle::Heading, bold(22.0));
    style
        .text_styles
        .insert(TextStyle::Body, FontId::proportional(14.0));
    style
        .text_styles
        .insert(TextStyle::Button, FontId::proportional(13.5));
    style
        .text_styles
        .insert(TextStyle::Monospace, FontId::monospace(12.5));
    style
        .text_styles
        .insert(TextStyle::Small, FontId::proportional(12.0));

    ctx.set_style(style);
}

/// System fonts, first match wins: Noto Sans for UI text (with a real bold
/// face for headings), Hack or DejaVu Sans Mono for logs and ids, and DejaVu
/// Sans for symbols (▶ ■ ↻ ⚠) the UI faces lack. egui's bundled fonts stay
/// as the last fallback, so a missing file only changes the look.
fn install_fonts(ctx: &egui::Context) {
    const DIRS: [&str; 5] = [
        "/usr/share/fonts/truetype/noto",
        "/usr/share/fonts/noto",
        "/usr/share/fonts/google-noto",
        "/usr/share/fonts/truetype/dejavu",
        "/usr/share/fonts/TTF",
    ];
    let mut fonts = egui::FontDefinitions::default();
    let mut load = |key: &str, files: &[&str]| -> Option<String> {
        let bytes = files
            .iter()
            .flat_map(|f| DIRS.iter().map(move |d| format!("{d}/{f}")))
            .chain(
                files
                    .iter()
                    .map(|f| format!("/usr/share/fonts/truetype/hack/{f}")),
            )
            .find_map(|p| std::fs::read(p).ok())?;
        fonts
            .font_data
            .insert(key.to_string(), FontData::from_owned(bytes).into());
        Some(key.to_string())
    };
    let ui = load("syskamling-ui", &["NotoSans-Regular.ttf", "DejaVuSans.ttf"]);
    let ui_bold = load(
        "syskamling-bold",
        &["NotoSans-Bold.ttf", "DejaVuSans-Bold.ttf"],
    );
    let mono = load(
        "syskamling-mono",
        &[
            "Hack-Regular.ttf",
            "DejaVuSansMono.ttf",
            "NotoSansMono-Regular.ttf",
        ],
    );
    let symbols = load("syskamling-symbols", &["DejaVuSans.ttf"]);

    let stock_prop = fonts.families[&FontFamily::Proportional].clone();
    let stock_mono = fonts.families[&FontFamily::Monospace].clone();
    let family = |first: Option<String>, stock: &[String]| -> Vec<String> {
        [first, symbols.clone()]
            .into_iter()
            .flatten()
            .chain(stock.iter().cloned())
            .collect()
    };
    let prop = family(ui, &stock_prop);
    let bold_fam = family(ui_bold, &stock_prop);
    let mono_fam = family(mono, &stock_mono);
    fonts.families.insert(FontFamily::Proportional, prop);
    fonts
        .families
        .insert(FontFamily::Name("bold".into()), bold_fam);
    fonts.families.insert(FontFamily::Monospace, mono_fam);
    ctx.set_fonts(fonts);
}

impl eframe::App for SysKamlingApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // a tab set before the first frame (screenshot harness) never fires
        // on_tab_change, so seed the process scan once here too
        if self.gui.tab == Tab::Processes && !self.proc_seed_done {
            self.proc_seed_done = true;
            self.last_proc_scan = Instant::now() - Duration::from_secs(999);
        }
        // Drain all pending engine messages.
        while let Ok(m) = self.rx.try_recv() {
            self.apply(m, ctx);
        }

        // Periodic refresh + continuous repaint at ~4 fps when idle so the
        // list stays live without burning CPU.
        if self.tick.elapsed() >= Duration::from_millis(if self.refresh > 0.0 { 250 } else { 100 })
        {
            self.tick = Instant::now();
            if self.refresh > 0.0
                && self.last_refresh.elapsed() >= Duration::from_secs_f64(self.refresh)
            {
                self.last_refresh = Instant::now();
                spawn_refresh(&self.rt, &self.docker, &self.tx);
            }
            self.reap_stats_streams();
            self.spawn_dashboard_pulls();
            self.spawn_agent_scan();
            self.sample_hardware();
            self.spawn_port_scan();
            self.spawn_proc_scan();
            self.spawn_strata_poll();
            // nvidia-smi (~50 ms) runs off the UI thread only when the
            // Hardware tab is visible; the result lands via the channel.
            if self.gui.tab == Tab::Hardware
                && self.last_gpu_stats.elapsed() >= Duration::from_secs_f64(self.refresh.max(1.0))
            {
                self.last_gpu_stats = Instant::now();
                if !*self.gpu_scan_busy.lock().unwrap() {
                    *self.gpu_scan_busy.lock().unwrap() = true;
                    let busy = Arc::clone(&self.gpu_scan_busy);
                    let tx = self.tx.clone();
                    self.rt.spawn_blocking(move || {
                        let g = gpu_stats();
                        *busy.lock().unwrap() = false;
                        let _ = tx.send(Msg::GpusScanned(g));
                    });
                }
            }
            ctx.request_repaint();
        }
        // A frame that lands between ticks must still schedule the next
        // one, or the loop goes idle until the mouse moves.
        ctx.request_repaint_after(Duration::from_millis(250));

        // Animation bookkeeping: ease status dots toward their target,
        // expire the spin, and schedule fast repaints while anything moves.
        let animating = self.advance_animations();
        if animating {
            ctx.request_repaint_after(Duration::from_millis(33));
        }

        // A changed status line becomes a fresh toast.
        if self.gui.status != self.toast.0 {
            self.toast = (self.gui.status.clone(), Instant::now());
        }

        self.nav_rail(ctx);
        if matches!(
            self.gui.tab,
            Tab::Containers | Tab::Images | Tab::Volumes | Tab::Agents | Tab::Ports
        ) {
            self.list_panel(ctx);
        }
        self.center(ctx);
        self.toast(ctx);
        if let Some(id) = self.gui.confirm_remove.clone() {
            self.remove_dialog(ctx, &id);
        }
    }
}

impl SysKamlingApp {
    /// Kick a /proc agent scan at the refresh cadence. The scan itself is
    /// synchronous /proc reading, so it runs on the tokio blocking pool;
    /// results land in the shared slot and are picked up next tick.
    fn spawn_agent_scan(&mut self) {
        if self.last_agent_scan.elapsed() < Duration::from_secs_f64(self.refresh.max(1.0)) {
            return;
        }
        if *self.agent_scan_busy.lock().unwrap() {
            return;
        }
        self.last_agent_scan = Instant::now();
        *self.agent_scan_busy.lock().unwrap() = true;
        let busy = Arc::clone(&self.agent_scan_busy);
        let scanner = Arc::clone(&self.agent_scanner);
        let tx = self.tx.clone();
        self.rt.spawn_blocking(move || {
            let sessions = scanner.lock().unwrap().scan();
            *busy.lock().unwrap() = false;
            let _ = tx.send(Msg::AgentsScanned(sessions));
        });
    }

    /// Hardware vitals need back-to-back samples for rates, so the sampler
    /// lives on the UI side and is read directly at the refresh cadence.
    /// /proc reads are cheap (microseconds); no thread needed. GPU numbers
    /// come from hwmon here; the expensive nvidia-smi pass runs separately
    /// on the blocking pool when the Hardware tab is visible.
    fn sample_hardware(&mut self) {
        if self.last_hw_sample.elapsed() < Duration::from_secs_f64(self.refresh.max(0.5)) {
            return;
        }
        self.last_hw_sample = Instant::now();
        let s = self.sampler.sample();
        // keep whatever richer nvidia numbers the async scan last produced;
        // hwmon-only fills in everything else instantly
        let nvidia = self
            .gui
            .gpus
            .iter()
            .filter(|g| g.driver == "nvidia")
            .cloned()
            .collect::<Vec<_>>();
        let mut gpus = gpu_stats_hwmon_only();
        gpus.retain(|g| g.driver != "nvidia");
        gpus.extend(nvidia);
        self.gui.push_hw(s, gpus);
    }

    /// Listening-socket scan: /proc/net tables plus an fd walk, so it goes
    /// to the blocking pool like the agent scan.
    fn spawn_port_scan(&mut self) {
        if self.last_port_scan.elapsed() < Duration::from_secs_f64(self.refresh.max(1.0) * 2.0) {
            return;
        }
        if *self.port_scan_busy.lock().unwrap() {
            return;
        }
        self.last_port_scan = Instant::now();
        *self.port_scan_busy.lock().unwrap() = true;
        let busy = Arc::clone(&self.port_scan_busy);
        let tx = self.tx.clone();
        self.rt.spawn_blocking(move || {
            let entries = portsmod::scan();
            *busy.lock().unwrap() = false;
            let _ = tx.send(Msg::PortsScanned(entries));
        });
    }

    /// Process scan: one /proc walk per refresh, on the blocking pool —
    /// the walk touches hundreds of pids. It runs on every tab (like the
    /// agent and port scans) so the nav-rail count and the dashboard's
    /// process card are never blank, whatever screen you are on. The
    /// scanner keeps tick counters between scans so CPU% is a real delta.
    fn spawn_proc_scan(&mut self) {
        if self.refresh <= 0.0 {
            return;
        }
        if self.last_proc_scan.elapsed() < Duration::from_secs_f64(self.refresh.max(1.0)) {
            return;
        }
        if *self.proc_scan_busy.lock().unwrap() {
            return;
        }
        self.last_proc_scan = Instant::now();
        *self.proc_scan_busy.lock().unwrap() = true;
        let busy = Arc::clone(&self.proc_scan_busy);
        let tx = self.tx.clone();
        let holder = Arc::clone(&self.proc_holder);
        self.rt.spawn_blocking(move || {
            let rows = {
                let mut sc = holder.lock().unwrap();
                sc.scan()
            };
            *busy.lock().unwrap() = false;
            let _ = tx.send(Msg::ProcsScanned(rows));
        });
    }

    /// Kick off decoding of the real app icons the latest process scan
    /// named. Decoding happens on the blocking pool (SVG parse + rasterize
    /// can cost a millisecond each); results arrive as Msg::IconReady.
    fn queue_icon_decodes(&mut self, ctx: &egui::Context) {
        let epoch = self.gui.icons.epoch();
        let mut names: Vec<String> = Vec::new();
        for r in &self.gui.procs {
            let n = &r.app_icon;
            if n.is_empty() || self.gui.icons.known(n) || self.gui.icon_requests.contains(n) {
                continue;
            }
            self.gui.icon_requests.insert(n.clone());
            names.push(n.clone());
            if names.len() >= 40 {
                break;
            }
        }
        if names.is_empty() {
            return;
        }
        let tx = self.tx.clone();
        let ctx2 = ctx.clone();
        self.rt.spawn_blocking(move || {
            for name in names {
                let rgba = iconsmod::pixels(&name, 48);
                let _ = tx.send(Msg::IconReady { name, epoch, rgba });
            }
            ctx2.request_repaint();
        });
    }

    /// Strata monitor poll: one /health + /metrics round-trip per refresh,
    /// only while the Strata tab or the dashboard is open, on the blocking pool so a slow or
    /// unreachable server never touches the UI thread.
    fn spawn_strata_poll(&mut self) {
        self.spawn_strata_poll_inner(false)
    }

    /// `force` bypasses the visible-tab gate (used for the first pull when
    /// the tab opens and right after settings change).
    fn spawn_strata_poll_inner(&mut self, force: bool) {
        if (!force && !matches!(self.gui.tab, Tab::Strata | Tab::Dashboard)) || self.refresh <= 0.0
        {
            return;
        }
        if !force
            && self.last_strata_poll.elapsed() < Duration::from_secs_f64(self.refresh.max(1.0))
        {
            return;
        }
        if *self.strata_poll_busy.lock().unwrap() {
            return;
        }
        self.last_strata_poll = Instant::now();
        *self.strata_poll_busy.lock().unwrap() = true;
        let busy = Arc::clone(&self.strata_poll_busy);
        let tx = self.tx.clone();
        let cfg = self.gui.strata_cfg.clone();
        self.rt.spawn_blocking(move || {
            let h = match strata::get_blocking(&cfg, "/health") {
                Ok((200, body)) => {
                    serde_json::from_slice(&body).map_err(|e| format!("/health: bad JSON: {e}"))
                }
                Ok((s, _)) => Err(format!("/health: HTTP {s}")),
                Err(e) => Err(e),
            };
            let _ = tx.send(Msg::StrataHealth(h));
            let m = match strata::get_blocking(&cfg, "/metrics") {
                Ok((200, body)) => strata::parse_metrics(&body),
                Ok((401, _)) => {
                    Err("the server needs an API key (401) — set it in the fields above".into())
                }
                Ok((s, body)) => {
                    let text = String::from_utf8_lossy(&body)
                        .chars()
                        .take(160)
                        .collect::<String>();
                    Err(format!("/metrics: HTTP {s} {text}"))
                }
                Err(e) => Err(e),
            };
            *busy.lock().unwrap() = false;
            let _ = tx.send(Msg::StrataMetrics(m.map(Box::new)));
        });
    }

    /// Dashboard-only pulls: system info, disk usage, event stream,
    /// per-container stats streams, consolidated log snapshots.
    fn spawn_dashboard_pulls(&mut self) {
        if self.last_info.elapsed() >= Duration::from_secs(5) {
            self.last_info = Instant::now();
            spawn_info(&self.rt, &self.docker, &self.tx);
        }
        if self.last_df.elapsed() >= Duration::from_secs(30) {
            self.last_df = Instant::now();
            spawn_df(&self.rt, &self.docker, &self.tx);
        }
        // The event stream stays open for the whole session, not just the
        // dashboard tab, so the feed is warm the moment you land on it.
        if !self.gui.events_active && self.last_events.elapsed() >= Duration::from_secs(5) {
            self.last_events = Instant::now();
            spawn_events(&self.rt, &self.docker, &self.tx);
        } else if self.gui.events_active
            && self.last_event.elapsed() >= Duration::from_secs(180)
            && self.last_events.elapsed() >= Duration::from_secs(5)
        {
            // Stream open but silent past the task-side watchdog's own
            // recycle window: something wedged, force a fresh connection.
            self.gui.events_active = false;
        }
        // Start a stats stream for any running container that doesn't
        // already have one (generation guards stale samples).
        let g = &self.gui;
        let known: Vec<String> = g
            .containers
            .iter()
            .filter(|c| c.state == "running")
            .map(|c| c.id.clone())
            .collect();
        let have: Vec<String> = g.stats.keys().cloned().collect();
        let streaming = &self.streaming_stats;
        let todo: Vec<String> = known
            .iter()
            .filter(|id| !have.contains(id) && !streaming.contains(id))
            .cloned()
            .collect();
        self.streaming_stats.retain(|id| known.contains(id));
        for id in todo {
            self.stats_spawned.insert(id.clone(), Instant::now());
            self.streaming_stats.push(id.clone());
            spawn_stats(&self.rt, &self.docker, &self.tx, &id, g.stats_gen);
        }
    }

    /// Reap stats streams that ended (container stopped, stream dropped) so
    /// they can be re-spawned on the next dashboard tick.
    fn reap_stats_streams(&mut self) {
        let running: Vec<String> = self
            .gui
            .containers
            .iter()
            .filter(|c| c.state == "running")
            .map(|c| c.id.clone())
            .collect();
        let g = &mut self.gui;
        // Streams whose container stopped are dead; streams for running
        // containers stay alive while a sample (or the spawn itself) landed
        // in the last 10 s, so a slow first sample doesn't churn sockets.
        let mut dead: Vec<String> = Vec::new();
        self.streaming_stats.retain(|id| {
            if running.contains(id) {
                let fresh = self
                    .stats_spawned
                    .get(id)
                    .is_some_and(|t| t.elapsed() < Duration::from_secs(10));
                if !fresh {
                    dead.push(id.clone());
                }
                fresh
            } else {
                dead.push(id.clone());
                false
            }
        });
        for id in dead {
            self.stats_spawned.remove(&id);
            g.stats.remove(&id);
            g.prev_stats.remove(&id);
        }
    }

    /// Advance all animation clocks; true while something is still moving.
    fn advance_animations(&mut self) -> bool {
        let now = Instant::now();
        let mut animating = false;

        // Status dots: ease each container's pulse toward 1.0 (running) or
        // fade the glow out (stopped). Values live in 0..=1.
        for c in &self.gui.containers {
            let running = c.state == "running";
            let e = self.dot_anim.entry(c.id.clone()).or_insert(0.0);
            let d = (if running { 1.0 } else { 0.0 }) - *e;
            if d.abs() > 0.01 {
                *e += d * 0.25;
                animating = true;
            } else {
                *e = if running { 1.0 } else { 0.0 };
            }
        }

        // Spinner: a 0.9 s spin once per action, cleared when the action ends.
        if let Some((id, started)) = &self.spin {
            if self
                .gui
                .pending_action
                .as_ref()
                .is_some_and(|(i, _)| i == id)
            {
                let _ = started;
                animating = true;
            } else {
                self.spin = None;
            }
        }

        // Logs fade-in (first second after opening a page).
        if now < self.logs_opened + Duration::from_millis(450) {
            animating = true;
        }

        animating
    }
}

impl SysKamlingApp {
    fn apply(&mut self, m: Msg, ctx: &egui::Context) {
        let g = &mut self.gui;
        match m {
            Msg::Refreshed(Ok(data)) => {
                g.error = None;
                g.daemon_ok = true;
                self.last_sync = Some(Instant::now());
                if g.status.starts_with("Refreshing") {
                    g.status.clear();
                }
                if let Ok(c) = data.containers {
                    g.containers = c;
                    // keep the selection pointing at the same container
                    if let Some(id) = g
                        .selected
                        .and_then(|i| g.containers.get(i))
                        .map(|c| c.id.clone())
                    {
                        g.selected = g.containers.iter().position(|c| c.id == id);
                    }
                    // Consolidated logs follow the container set: pick up
                    // newly created containers, drop removed ones.
                    if g.logs_container.as_deref() == Some("__all__") {
                        let want: Vec<(String, String)> = g
                            .containers
                            .iter()
                            .filter(|c| c.state == "running")
                            .map(|c| (c.id.clone(), c.short_name()))
                            .collect();
                        let have = self.all_logs_follow.clone();
                        for (id, name) in &want {
                            if !have.contains(id) {
                                self.all_logs_follow.push(id.clone());
                                spawn_logs(
                                    &self.rt,
                                    &self.docker,
                                    &self.tx,
                                    id,
                                    g.logs_gen,
                                    Some(name.clone()),
                                );
                            }
                        }
                        self.all_logs_follow
                            .retain(|id| want.iter().any(|(wid, _)| wid == id));
                    }
                }
                if let Ok(i) = data.images {
                    g.images = i;
                }
                if let Ok(v) = data.volumes {
                    g.volumes = v;
                }
                g.last_refresh = now_string();
            }
            Msg::Refreshed(Err(e)) => {
                g.daemon_ok = false;
                g.error = Some(format!("refresh: {e}"));
            }
            Msg::LogStart(gen, Ok(())) => {
                if Some(gen) == g.pending_logs_gen {
                    g.pending_logs_gen = None;
                    g.logs_active = true;
                    g.status = format!("streaming logs from {}", g.logs_container_name);
                } else if g.logs_container.as_deref() == Some("__all__") {
                    g.logs_active = true;
                }
            }
            Msg::LogStart(gen, Err(e)) => {
                if Some(gen) == g.pending_logs_gen {
                    g.pending_logs_gen = None;
                    g.error = Some(format!("logs: {e}"));
                }
            }
            Msg::LogLines(gen, lines) => {
                if gen == g.logs_gen {
                    g.push_logs(lines);
                }
            }
            Msg::LogDone(gen, id, r) => {
                if gen == g.logs_gen {
                    if g.logs_container.as_deref() == Some("__all__") {
                        // One per-container follow ending shouldn't mark the
                        // whole consolidated page as stopped; only when the
                        // last follower ends does the page go idle.
                        self.all_logs_follow.retain(|x| x != &id);
                        if self.all_logs_follow.is_empty() {
                            g.logs_active = false;
                        }
                    } else {
                        g.logs_active = false;
                    }
                    if let Err(e) = r {
                        g.error = Some(format!("logs: {e}"));
                    }
                }
            }
            Msg::ActionDone(Ok((verb, name))) => {
                g.status = format!("{verb} {name}");
                g.pending_action = None;
            }
            Msg::ActionDone(Err(e)) => {
                g.pending_action = None;
                g.error = Some(e);
            }
            Msg::Info(Ok(info)) => {
                g.info = Some(info);
            }
            Msg::Info(Err(e)) => {
                g.error = Some(format!("info: {e}"));
            }
            Msg::Df(Ok(df)) => {
                g.df = Some(df);
            }
            Msg::Df(Err(e)) => {
                g.error = Some(format!("disk usage: {e}"));
            }
            Msg::Stats(id, s) => {
                self.stats_spawned.insert(id.clone(), Instant::now());
                // Drop samples from streams superseded by a newer generation.
                let prev = g.stats.insert(id.clone(), (*s).clone());
                if let Some(p) = prev {
                    g.prev_stats.insert(id.clone(), p);
                }
                // First sample seeds the history; later ones append.
                let prev = g.prev_stats.get(&id);
                g.hist.entry(id).or_default().push(&s, prev);
            }
            Msg::Events(Ok(ev)) => {
                g.events_active = true;
                self.last_event = Instant::now();
                for e in ev {
                    g.events.push(e);
                }
                if g.events.len() > 200 {
                    let excess = g.events.len() - 200;
                    g.events.drain(..excess);
                }
                // A fresh lifecycle event means the view may need to show
                // it: schedule a repaint so the feed updates without waiting
                // for the next 250 ms tick.
                ctx.request_repaint();
            }
            Msg::Events(Err(e)) => {
                g.events_active = false;
                g.error = Some(format!("events: {e}"));
            }
            Msg::EventsClosed => {
                g.events_active = false;
            }
            Msg::AllLogs(lines) => {
                g.push_logs(lines);
                g.sort_all_logs();
            }
            Msg::AgentsScanned(sessions) => {
                g.agents = sessions;
            }
            Msg::PortsScanned(entries) => {
                g.ports = entries;
                // keep the selection pointing at the same socket
                if let Some(key) = g.selected_port.clone() {
                    g.selected_port = g
                        .ports
                        .iter()
                        .find(|p| {
                            (p.proto, p.addr.as_str(), p.port)
                                == (key.0.as_str(), key.1.as_str(), key.2)
                        })
                        .map(|p| (p.proto.to_string(), p.addr.clone(), p.port));
                }
            }
            Msg::ProcsScanned(rows) => {
                g.procs = rows;
                // keep the selection pointing at the same process
                if let Some(pid) = g.selected_proc {
                    if !g.procs.iter().any(|r| r.pid == pid) {
                        g.selected_proc = None;
                    }
                }
                self.queue_icon_decodes(ctx);
            }
            Msg::IconReady { name, epoch, rgba } => {
                if epoch != g.icons.epoch() {
                    return;
                }
                g.icon_requests.remove(&name);
                if let Some(rgba) = rgba {
                    let img = egui::ColorImage::from_rgba_unmultiplied(
                        [rgba.w as usize, rgba.h as usize],
                        &rgba.px,
                    );
                    let tex = ctx.load_texture(
                        format!("appicon:{name}:{epoch}"),
                        img,
                        egui::TextureOptions::LINEAR,
                    );
                    g.icons.insert(&name, tex);
                } else {
                    g.icons.mark_missing(&name);
                }
            }
            Msg::GpusScanned(gs) => {
                // async nvidia-smi pass: merge over the hwmon rows without
                // touching the sampler histories
                g.gpus = gs;
            }
            Msg::StrataHealth(res) => match res {
                Ok(h) => {
                    g.strata_health = Some(h);
                    g.strata_error = None;
                }
                Err(e) => {
                    g.strata_health = None;
                    g.strata_error = Some(e);
                }
            },
            Msg::StrataMetrics(res) => match res {
                Ok(m) => {
                    g.strata_metrics = Some(m);
                    g.strata_error = None;
                }
                Err(e) => g.strata_error = Some(e),
            },
        }
    }

    /// Left navigation rail: brand, engine status, sections, host footer.
    fn nav_rail(&mut self, ctx: &egui::Context) {
        #[cfg(test)]
        NAV_ROW_RECTS.lock().unwrap().clear();
        let mut new_tab: Option<Tab> = None;
        let mut refresh_now = false;
        let g = &self.gui;
        let n_run = g.containers.iter().filter(|c| c.state == "running").count();
        let items = [
            (Tab::Dashboard, "Dashboard", String::new()),
            (
                Tab::Containers,
                "Containers",
                format!("{n_run}/{}", g.containers.len()),
            ),
            (Tab::Images, "Images", g.images.len().to_string()),
            (Tab::Volumes, "Volumes", g.volumes.len().to_string()),
            (Tab::Agents, "Agents", g.agents.len().to_string()),
            (Tab::Hardware, "Hardware", String::new()),
            (Tab::Processes, "Processes", g.procs.len().to_string()),
            (Tab::Ports, "Ports", g.ports.len().to_string()),
            (Tab::Strata, "Strata", String::new()),
            (Tab::Logs, "Logs", String::new()),
        ];
        let sys_items = [(Tab::Settings, "Settings", String::new())];
        let version = g
            .info
            .as_ref()
            .map(|i| i.server_version.clone())
            .unwrap_or_default();
        let synced = self.last_sync.map(|t| t.elapsed().as_secs());
        egui::SidePanel::left("nav")
            .exact_width(216.0)
            .resizable(false)
            .frame(
                Frame::NONE
                    .fill(RAIL)
                    .inner_margin(Margin::symmetric(12, 18)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.add_space(4.0);
                    let (r, _) = ui.allocate_exact_size(Vec2::splat(32.0), Sense::hover());
                    logo_mark(ui.painter(), r);
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 0.0;
                        ui.label(RichText::new("SysKamling").font(bold(15.0)).color(TEXT));
                        ui.label(
                            RichText::new("Developer system monitor")
                                .size(11.5)
                                .color(FAINT),
                        );
                    });
                });
                ui.add_space(16.0);
                engine_card(ui, g.daemon_ok, &version);
                ui.add_space(20.0);
                section_label(ui, "WORKSPACE");
                ui.add_space(2.0);
                ui.spacing_mut().item_spacing.y = 2.0;
                // stable pick: the alphabetically-first decoded icon, so the
                // rail mark doesn't flicker as scans reorder the process list
                let proc_icon = g
                    .procs
                    .iter()
                    .map(|r| r.app_icon.as_str())
                    .filter(|n| !n.is_empty())
                    .min()
                    .and_then(|n| g.icons.get(n));
                for (tab, label, badge) in &items {
                    let real = (*tab == Tab::Processes).then_some(proc_icon).flatten();
                    if nav_item(ui, *tab, label, badge, g.tab == *tab, real) {
                        new_tab = Some(*tab);
                    }
                }
                ui.add_space(14.0);
                section_label(ui, "SYSTEM");
                ui.add_space(2.0);
                for (tab, label, badge) in &sys_items {
                    if nav_item(ui, *tab, label, badge, g.tab == *tab, None) {
                        new_tab = Some(*tab);
                    }
                }
                // Footer pinned to the bottom: host card, sync time, refresh.
                let footer_h = if g.info.is_some() { 156.0 } else { 72.0 };
                ui.add_space((ui.available_height() - footer_h).max(12.0));
                ui.spacing_mut().item_spacing.y = 8.0;
                if let Some(i) = &g.info {
                    Frame::NONE
                        .fill(SURFACE)
                        .stroke(Stroke::new(1.0_f32, BORDER))
                        .corner_radius(10)
                        .inner_margin(Margin::symmetric(12, 10))
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.spacing_mut().item_spacing.y = 2.0;
                            ui.label(RichText::new("HOST").size(10.5).color(FAINT));
                            ui.add(
                                egui::Label::new(
                                    RichText::new(&i.operating_system).size(12.5).color(TEXT),
                                )
                                .truncate(),
                            );
                            ui.label(
                                RichText::new(format!(
                                    "{} CPUs · {} RAM",
                                    i.ncpu,
                                    human_size(i.mem_total)
                                ))
                                .size(12.0)
                                .color(MUTED),
                            );
                        });
                }
                let sync = match synced {
                    Some(s) => format!("Synced {}", ago(s as i64)),
                    None => "Connecting…".to_string(),
                };
                ui.horizontal(|ui| {
                    ui.add_space(4.0);
                    ui.label(RichText::new(sync).size(11.5).color(FAINT));
                });
                let w = ui.available_width();
                if ghost_button(ui, "↻   Refresh now", w).clicked() {
                    refresh_now = true;
                }
            });
        if let Some(t) = new_tab {
            if t != self.gui.tab {
                self.on_tab_change(t);
            }
        }
        if refresh_now {
            spawn_refresh(&self.rt, &self.docker, &self.tx);
            self.spawn_dashboard_pulls();
            self.gui.status = "Refreshing…".to_string();
        }
    }

    /// Tab switch bookkeeping: opening the Logs tab loads the consolidated
    /// stream; opening the dashboard (re)starts stats streams.
    fn on_tab_change(&mut self, t: Tab) {
        self.gui.tab = t;
        // the poller gates on the visible tab; a forced pull means the
        // Strata page (and the dashboard's Strata card) is never blank on arrival
        if matches!(t, Tab::Strata | Tab::Dashboard) {
            self.last_strata_poll = Instant::now() - Duration::from_secs(999);
            self.spawn_strata_poll_inner(true);
        }
        match t {
            Tab::Logs => {
                if self.gui.logs_container.is_none() {
                    self.load_all_logs();
                }
            }
            Tab::Dashboard => {
                self.spawn_dashboard_pulls();
            }
            // same never-blank rule as Strata: seed a scan the moment the
            // Processes tab opens instead of waiting a full refresh tick
            Tab::Processes => {
                self.last_proc_scan = Instant::now() - Duration::from_secs(999);
                self.proc_seed_done = true;
                self.spawn_proc_scan();
            }
            _ => {}
        }
    }

    /// Rebuild the consolidated Logs tab: per-container timestamped tails
    /// merged into one page, then resume following each running container.
    fn load_all_logs(&mut self) {
        let g = &mut self.gui;
        g.logs_gen += 1;
        let gen = g.logs_gen;
        g.logs.clear();
        g.logs_container = Some("__all__".into());
        g.logs_container_name = "all containers".into();
        g.logs_follow = true;
        g.logs_active = true;
        self.logs_opened = Instant::now();
        self.prev_log_len = 0;
        let ids: Vec<String> = g.containers.iter().map(|c| c.id.clone()).collect();
        let names: Vec<(String, String)> = g
            .containers
            .iter()
            .map(|c| (c.id.clone(), c.short_name()))
            .collect();
        spawn_all_logs(&self.rt, &self.docker, &self.tx, ids);
        self.all_logs_follow.clear();
        for (id, name) in names {
            if self
                .gui
                .containers
                .iter()
                .any(|c| c.id == id && c.state == "running")
            {
                self.all_logs_follow.push(id.clone());
                spawn_logs(&self.rt, &self.docker, &self.tx, &id, gen, Some(name));
            }
        }
    }

    /// Second column on list tabs: title, search, state filter, rows.
    fn list_panel(&mut self, ctx: &egui::Context) {
        let tab = self.gui.tab;
        egui::SidePanel::left("list")
            .resizable(true)
            .default_width(360.0)
            .width_range(300.0..=560.0)
            .frame(Frame::NONE.fill(BG).inner_margin(Margin {
                left: 16,
                right: 14,
                top: 22,
                bottom: 8,
            }))
            .show(ctx, |ui| {
                let (title, count, hint) = match tab {
                    Tab::Containers => (
                        "Containers",
                        self.gui.containers.len(),
                        "Search name, image or state",
                    ),
                    Tab::Images => ("Images", self.gui.images.len(), "Search tags"),
                    Tab::Agents => ("Agents", self.gui.agents.len(), "Search agent or project"),
                    Tab::Ports => (
                        "Ports",
                        self.gui.ports.len(),
                        "Search port, process or address",
                    ),
                    _ => ("Volumes", self.gui.volumes.len(), "Search volumes"),
                };
                ui.horizontal(|ui| {
                    ui.label(RichText::new(title).font(bold(20.0)).color(TEXT));
                    ui.label(RichText::new(count.to_string()).size(14.0).color(FAINT));
                });
                ui.add_space(8.0);
                match tab {
                    Tab::Agents => search_box(ui, &mut self.gui.agent_filter, hint),
                    Tab::Ports => search_box(ui, &mut self.gui.port_filter, hint),
                    _ => search_box(ui, &mut self.gui.filter, hint),
                }
                if tab == Tab::Containers {
                    ui.add_space(4.0);
                    let n_all = self.gui.containers.len();
                    let n_run = self
                        .gui
                        .containers
                        .iter()
                        .filter(|c| c.state == "running")
                        .count();
                    let opts = [
                        (StateFilter::All, format!("All  {n_all}")),
                        (StateFilter::Running, format!("Running  {n_run}")),
                        (StateFilter::Stopped, format!("Stopped  {}", n_all - n_run)),
                    ];
                    if let Some(f) = segmented(ui, &opts, self.gui.state_filter) {
                        self.gui.state_filter = f;
                        self.gui.selected = None;
                    }
                }
                if tab == Tab::Ports {
                    ui.add_space(4.0);
                    let n_tcp = self.gui.ports.iter().filter(|p| p.proto == "tcp").count();
                    let n_udp = self.gui.ports.len() - n_tcp;
                    let opts = [
                        (ProtoFilter::All, format!("All  {}", self.gui.ports.len())),
                        (ProtoFilter::Tcp, format!("TCP  {n_tcp}")),
                        (ProtoFilter::Udp, format!("UDP  {n_udp}")),
                    ];
                    if let Some(f) = segmented(ui, &opts, self.gui.ports_proto) {
                        self.gui.ports_proto = f;
                    }
                    ui.add_space(4.0);
                    let g = &mut self.gui;
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = 6.0;
                        for fam in [4u8, 6] {
                            let on = g.ports_family == Some(fam);
                            if toggle_chip(ui, &format!("IPv{fam}"), on).clicked() {
                                g.ports_family = if on { None } else { Some(fam) };
                            }
                        }
                        if toggle_chip(ui, "Listening", g.ports_listening_only)
                            .on_hover_text("Hide established connections")
                            .clicked()
                        {
                            g.ports_listening_only = !g.ports_listening_only;
                        }
                        if toggle_chip(ui, "Exposed", g.ports_exposed_only)
                            .on_hover_text(
                                "Only sockets bound beyond loopback (reachable from the network)",
                            )
                            .clicked()
                        {
                            g.ports_exposed_only = !g.ports_exposed_only;
                        }
                        if toggle_chip(ui, "Has process", g.ports_owned_only)
                            .on_hover_text("Only sockets with a discoverable owning process")
                            .clicked()
                        {
                            g.ports_owned_only = !g.ports_owned_only;
                        }
                    });
                }
                ui.add_space(8.0);
                match tab {
                    Tab::Containers => self.container_list(ui),
                    Tab::Images => self.image_list(ui),
                    Tab::Agents => self.agent_list(ui),
                    Tab::Ports => self.port_list(ui),
                    _ => self.volume_list(ui),
                }
            });
    }

    fn container_list(&mut self, ui: &mut Ui) {
        // Snapshot what the rows need so the closure never borrows self.
        let rows = self.gui.filtered();
        if rows.is_empty() {
            empty_state(ui, "No containers match", "Try another search or filter.");
            return;
        }
        let sel = self.gui.selected;
        let busy_id = self
            .gui
            .pending_action
            .as_ref()
            .map(|(id, _)| id.clone())
            .unwrap_or_default();
        let dot_anim = &self.dot_anim;
        let stats = &self.gui.stats;
        let prev_stats = &self.gui.prev_stats;
        let hist = &self.gui.hist;
        let mut new_sel: Option<usize> = None;

        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 4.0;
                let time = ui.input(|i| i.time);
                for (i, c) in rows {
                    let (rect, resp) = ui
                        .allocate_exact_size(Vec2::new(ui.available_width(), 58.0), Sense::click());
                    if resp.clicked() {
                        new_sel = Some(i);
                    }
                    if !ui.is_rect_visible(rect) {
                        continue;
                    }
                    let running = c.state == "running";
                    let busy = !busy_id.is_empty() && busy_id == c.id;
                    let pulse = dot_anim.get(&c.id).copied().unwrap_or(0.0);
                    let p = ui.painter();
                    row_bg(p, rect, sel == Some(i), resp.hovered());

                    let dot_c = Pos2::new(rect.left() + 18.0, rect.center().y);
                    if busy {
                        spinner_at(p, time, dot_c, 6.0, ACCENT);
                        ui.ctx().request_repaint();
                    } else {
                        status_dot(p, dot_c, running, pulse);
                    }

                    let right_w = if running { 118.0 } else { 74.0 };
                    let left = rect.left() + 34.0;
                    let text_w = (rect.width() - 34.0 - right_w - 8.0).max(40.0);
                    let name = elide(p, c.short_name(), FontId::proportional(14.0), TEXT, text_w);
                    p.galley(Pos2::new(left, rect.top() + 10.0), name, TEXT);
                    let sub = elide(
                        p,
                        format!("{} · {}", c.image, c.status),
                        FontId::proportional(12.0),
                        MUTED,
                        text_w,
                    );
                    p.galley(Pos2::new(left, rect.top() + 32.0), sub, MUTED);

                    if running {
                        if let Some(h) = hist.get(&c.id) {
                            let r = Rect::from_min_size(
                                Pos2::new(rect.right() - 118.0, rect.top() + 17.0),
                                Vec2::new(52.0, 24.0),
                            );
                            let skip = h.cpu.len().saturating_sub(30);
                            sparkline(p, r, h.cpu.iter().skip(skip).copied(), ACCENT, 5.0);
                        }
                        if let Some(s) = stats.get(&c.id) {
                            let cpu = StatsSample::cpu_percent(prev_stats.get(&c.id), s)
                                .unwrap_or(0.0)
                                .max(0.0);
                            p.text(
                                Pos2::new(rect.right() - 12.0, rect.top() + 13.0),
                                egui::Align2::RIGHT_TOP,
                                format!("{cpu:.1}%"),
                                FontId::monospace(12.5),
                                TEXT,
                            );
                            p.text(
                                Pos2::new(rect.right() - 12.0, rect.top() + 32.0),
                                egui::Align2::RIGHT_TOP,
                                human_size(s.mem_usage()),
                                FontId::monospace(11.5),
                                FAINT,
                            );
                        }
                    } else {
                        let label = capitalize(&c.state);
                        pill_at(
                            p,
                            Pos2::new(rect.right() - 12.0, rect.center().y),
                            &label,
                            FAINT,
                            SURFACE_HI,
                        );
                    }
                    let ports = c.port_summary();
                    resp.on_hover_text(format!(
                        "{}\n{}{}",
                        c.image,
                        c.status,
                        if ports.is_empty() {
                            String::new()
                        } else {
                            format!("\n{ports}")
                        }
                    ));
                }
            });

        if let Some(i) = new_sel {
            self.gui.selected = Some(i);
            // The detail page shows this container's logs inline.
            if let Some(id) = self.gui.containers.get(i).map(|c| c.id.clone()) {
                if self.gui.logs_container.as_deref() != Some(id.as_str()) {
                    self.open_logs(&id);
                }
            }
        }
    }

    fn image_list(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        let f = g.filter.to_lowercase();
        let sel = g.selected_image.clone();
        let rows: Vec<&ImageSummary> = g
            .images
            .iter()
            .filter(|im| f.is_empty() || im.tag().to_lowercase().contains(&f))
            .collect();
        if rows.is_empty() {
            empty_state(ui, "No images match", "Try another search.");
            return;
        }
        let mut new_sel: Option<Option<String>> = None;
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 4.0;
                for im in rows {
                    let (rect, resp) = ui
                        .allocate_exact_size(Vec2::new(ui.available_width(), 58.0), Sense::click());
                    let is_sel = sel.as_deref() == Some(im.id.as_str());
                    if resp.clicked() {
                        new_sel = Some(if is_sel { None } else { Some(im.id.clone()) });
                    }
                    if !ui.is_rect_visible(rect) {
                        continue;
                    }
                    let p = ui.painter();
                    row_bg(p, rect, is_sel, resp.hovered());
                    let in_use = im.containers > 0;
                    icon_tile(p, rect, Tab::Images, if in_use { ACCENT } else { FAINT });
                    let left = rect.left() + 54.0;
                    let text_w = (rect.width() - 54.0 - 86.0).max(40.0);
                    let tag = im.tag();
                    let (repo, version) = tag.rsplit_once(':').unwrap_or((tag.as_str(), ""));
                    let name = elide(
                        p,
                        repo.to_string(),
                        FontId::proportional(14.0),
                        TEXT,
                        text_w,
                    );
                    p.galley(Pos2::new(left, rect.top() + 10.0), name, TEXT);
                    let usage = if in_use {
                        format!("{} in use", im.containers)
                    } else {
                        "unused".to_string()
                    };
                    let sub = if version.is_empty() {
                        format!("{} · {usage}", im.short_id())
                    } else {
                        format!("{version} · {} · {usage}", im.short_id())
                    };
                    let sub = elide(p, sub, FontId::proportional(12.0), MUTED, text_w);
                    p.galley(Pos2::new(left, rect.top() + 32.0), sub, MUTED);
                    p.text(
                        Pos2::new(rect.right() - 12.0, rect.center().y),
                        egui::Align2::RIGHT_CENTER,
                        human_size(im.size),
                        FontId::monospace(12.0),
                        MUTED,
                    );
                }
            });
        if let Some(s) = new_sel {
            self.gui.selected_image = s;
        }
    }

    fn agent_list(&mut self, ui: &mut Ui) {
        let mut rows: Vec<&AgentSession> = self
            .gui
            .filtered_agents()
            .into_iter()
            .map(|(_, a)| a)
            .collect();
        let sel = self.gui.selected_agent;
        if rows.is_empty() {
            if self.gui.agents.is_empty() {
                empty_state(
                    ui,
                    "No AI agents running",
                    "Claude Code, Codex, Cursor, omp, ollama… any recognized agent shows up here.",
                );
            } else {
                empty_state(ui, "No agents match", "Try another search.");
            }
            return;
        }
        // whatever needs you first, then whatever is working
        rows.sort_by_key(|a| (agent_state(a).rank, a.agent, a.lead_pid));
        let count = |rank: u8| {
            self.gui
                .agents
                .iter()
                .filter(|a| agent_state(a).rank == rank)
                .count()
        };
        let (needs, working) = (count(0), count(1));
        ui.horizontal(|ui| {
            ui.add_space(4.0);
            ui.label(
                RichText::new(format!("{} sessions", self.gui.agents.len()))
                    .size(12.0)
                    .color(MUTED),
            );
            if working > 0 {
                ui.label(
                    RichText::new(format!("· {working} working"))
                        .size(12.0)
                        .color(GREEN),
                );
            }
            if needs > 0 {
                ui.label(
                    RichText::new(format!("· {needs} need you"))
                        .size(12.0)
                        .color(AMBER),
                );
            }
        });
        ui.add_space(6.0);
        let now = epoch();
        let time = ui.input(|i| i.time);
        let mut new_sel: Option<Option<u32>> = None;
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 4.0;
                for a in rows {
                    let (rect, resp) = ui
                        .allocate_exact_size(Vec2::new(ui.available_width(), 62.0), Sense::click());
                    let is_sel = sel == Some(a.lead_pid);
                    if resp.clicked() {
                        new_sel = Some(if is_sel { None } else { Some(a.lead_pid) });
                    }
                    if !ui.is_rect_visible(rect) {
                        continue;
                    }
                    let p = ui.painter();
                    row_bg(p, rect, is_sel, resp.hovered());
                    icon_tile(p, rect, Tab::Agents, agent_color(a.agent));
                    let st = agent_state(a);
                    // status pill on the right, with how long it has held
                    let pill_c = Pos2::new(rect.right() - 10.0, rect.top() + 20.0);
                    pill_at(p, pill_c, st.label, st.color, st.color.gamma_multiply(0.14));
                    if let Some(since) = a.activity.as_ref().and_then(|x| x.status_since) {
                        p.text(
                            Pos2::new(rect.right() - 12.0, rect.top() + 44.0),
                            egui::Align2::RIGHT_CENTER,
                            uptime_str(now - since),
                            FontId::proportional(11.0),
                            FAINT,
                        );
                    }
                    let left = rect.left() + 54.0;
                    let text_w = (rect.width() - 54.0 - 104.0).max(40.0);
                    let title = elide(p, agent_title(a), FontId::proportional(14.0), TEXT, text_w);
                    p.galley(Pos2::new(left, rect.top() + 10.0), title, TEXT);
                    // second line: what it's doing right now, else where it runs
                    let doing = a
                        .activity
                        .as_ref()
                        .filter(|_| st.live)
                        .and_then(|x| x.steps.last());
                    let (sub, sub_color) = match doing {
                        Some(s) => (step_line(s), MUTED),
                        None => (agent_where(a), FAINT),
                    };
                    let mut x = left;
                    if doing.is_some() {
                        spinner_at(
                            p,
                            time,
                            Pos2::new(x + 5.0, rect.top() + 42.0),
                            4.5,
                            st.color,
                        );
                        x += 16.0;
                    }
                    let sub = elide(
                        p,
                        sub,
                        FontId::proportional(12.0),
                        sub_color,
                        text_w - (x - left),
                    );
                    p.galley(Pos2::new(x, rect.top() + 34.0), sub, sub_color);
                }
            });
        if let Some(s) = new_sel {
            self.gui.selected_agent = s;
        }
        if self.gui.agents.iter().any(|a| agent_state(a).live) {
            ui.ctx().request_repaint_after(Duration::from_millis(50));
        }
    }

    fn port_list(&mut self, ui: &mut Ui) {
        let rows = self.gui.filtered_ports();
        let sel = self.gui.selected_port.clone();
        if rows.is_empty() {
            if self.gui.ports.is_empty() {
                empty_state(
                    ui,
                    "No listening sockets",
                    "Nothing is bound on this machine — or /proc/net is unreadable.",
                );
            } else {
                empty_state(
                    ui,
                    "No sockets match",
                    "Try another search or turn off some filters.",
                );
            }
            return;
        }
        let mut new_sel: Option<Option<(String, String, u16)>> = None;
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 4.0;
                for (_i, p) in rows {
                    let key = (p.proto.to_string(), p.addr.clone(), p.port);
                    let rect_h = if p.proc_name.is_some() { 58.0 } else { 44.0 };
                    let (rect, resp) = ui.allocate_exact_size(
                        Vec2::new(ui.available_width(), rect_h),
                        Sense::click(),
                    );
                    let is_sel = sel.as_ref() == Some(&key);
                    if resp.clicked() {
                        new_sel = Some(if is_sel { None } else { Some(key.clone()) });
                    }
                    if !ui.is_rect_visible(rect) {
                        continue;
                    }
                    let paint = ui.painter();
                    row_bg(paint, rect, is_sel, resp.hovered());
                    let color = port_color(p);
                    icon_tile(paint, rect, Tab::Ports, color);
                    let left = rect.left() + 54.0;
                    let text_w = (rect.width() - 54.0 - 86.0).max(40.0);
                    let name = elide(
                        paint,
                        format!("{}   :{}", p.proto.to_uppercase(), p.port),
                        FontId::monospace(14.0),
                        TEXT,
                        text_w,
                    );
                    paint.galley(Pos2::new(left, rect.top() + 8.0), name, TEXT);
                    let sub = p
                        .proc_name
                        .clone()
                        .unwrap_or_else(|| format!("{} · kernel/other", p.addr));
                    let sub = elide(paint, sub, FontId::proportional(12.5), MUTED, text_w);
                    paint.galley(Pos2::new(left, rect.top() + 30.0), sub, MUTED);
                    paint.text(
                        Pos2::new(rect.right() - 12.0, rect.center().y),
                        egui::Align2::RIGHT_CENTER,
                        p.state,
                        FontId::monospace(11.0),
                        port_state_color(p.state),
                    );
                }
            });
        if let Some(s) = new_sel {
            self.gui.selected_port = s;
        }
    }

    /// Process list: Task-Manager style sortable table. Clicking a column
    /// header sorts by it; clicking again flips the direction.
    /// Search bar for the Processes tab, painted at the top of the center
    /// panel — wider than the sidebar box, with a live match count.
    fn proc_search(&mut self, ui: &mut Ui) {
        let n_all = self.gui.procs.len();
        let hint = "Search name, command, user, pid or cwd — click a column header to sort";
        search_box(ui, &mut self.gui.proc_filter, hint);
        let n = self
            .gui
            .procs
            .iter()
            .filter(|r| {
                let f = self.gui.proc_filter.to_lowercase();
                f.is_empty()
                    || r.name.to_lowercase().contains(&f)
                    || r.cmdline.to_lowercase().contains(&f)
                    || r.user.to_lowercase().contains(&f)
                    || r.cwd.to_lowercase().contains(&f)
                    || r.pid.to_string() == f
            })
            .count();
        if !self.gui.proc_filter.is_empty() {
            ui.label(
                RichText::new(format!("{n} of {n_all} processes match"))
                    .size(11.5)
                    .color(FAINT),
            );
        }
    }

    fn proc_list(&mut self, ui: &mut Ui) {
        if self.gui.procs.is_empty() {
            empty_state(
                ui,
                "No process data yet",
                "The /proc walk should land within a refresh tick — if this\
                 persists, the app may lack permission to read /proc.",
            );
            return;
        }
        // fixed column layout: name flexes, the rest are right-aligned numbers
        let w = ui.available_width();
        let icon_w = 30.0;
        let user_w = 84.0;
        let pid_w = 64.0;
        let cpu_w = 74.0;
        let mem_w = 92.0;
        let name_w = (w - icon_w - user_w - pid_w - cpu_w - mem_w - 30.0).max(80.0);

        // ---- filter first, then group ----
        let f = self.gui.proc_filter.to_lowercase();
        let rows: Vec<ProcInfo> = self
            .gui
            .procs
            .iter()
            .filter(|r| {
                f.is_empty()
                    || r.name.to_lowercase().contains(&f)
                    || r.cmdline.to_lowercase().contains(&f)
                    || r.user.to_lowercase().contains(&f)
                    || r.cwd.to_lowercase().contains(&f)
                    || r.pid.to_string() == f
            })
            .cloned()
            .collect();
        if rows.is_empty() {
            empty_state(ui, "No process matches", "Try another search.");
            return;
        }

        // ---- app groups ----
        // a process belongs to an app when its desktop icon name resolves;
        // everything else (kernel threads, daemons, bare binaries) stays a
        // row of its own. The table shows one row per app, and clicking the
        // row expands it to the processes behind it. The grouping itself is
        // shared with the TUI (procs::group_rows).
        let entries = procsmod::group_rows(&rows, self.gui.proc_sort, self.gui.proc_sort_desc);

        // ---- header ----
        let (hr, _) = ui.allocate_exact_size(Vec2::new(w, 24.0), Sense::hover());
        let p = ui.painter().clone();
        p.line_segment(
            [hr.left_bottom(), hr.right_bottom()],
            Stroke::new(1.0_f32, BORDER),
        );
        let mut new_sort: Option<SortCol> = None;
        // left x of every header cell, captured for the offscreen tests
        #[cfg(test)]
        PROC_HEADER_CELLS.lock().unwrap().clear();
        #[cfg(test)]
        let header_cells = &*PROC_HEADER_CELLS;
        let mut header = |ui: &mut Ui, label: &str, col: SortCol, x: f32, align: egui::Align2| {
            let p = ui.painter();
            // RIGHT-aligned labels sit at the cell's right edge; the Name
            // cell starts at x. Cells never overlap, so a click can only
            // ever hit one header.
            let (cx, cw) = match col {
                SortCol::Name => (hr.left(), icon_w + name_w + 12.0),
                SortCol::Pid => (x - pid_w - 6.0, pid_w + 12.0),
                SortCol::Cpu => (x - cpu_w - 6.0, cpu_w + 12.0),
                SortCol::Mem => (x - mem_w - 6.0, mem_w + 12.0),
            };
            let cell = Rect::from_min_size(Pos2::new(cx, hr.top()), Vec2::new(cw, 22.0));
            #[cfg(test)]
            header_cells.lock().unwrap().push((col, cell));
            let resp = ui.interact(cell, ui.id().with(("proc-sort", col as u8)), Sense::click());
            if resp.clicked() {
                new_sort = Some(col);
            }
            if resp.hovered() {
                p.rect_filled(cell, 4.0, SURFACE_HI);
            }
            let active = self.gui.proc_sort == col;
            let arrow = if active {
                if self.gui.proc_sort_desc {
                    " ▼"
                } else {
                    " ▲"
                }
            } else {
                ""
            };
            p.text(
                Pos2::new(x, hr.center().y),
                align,
                format!("{label}{arrow}"),
                FontId::proportional(12.0),
                if active {
                    ACCENT
                } else if resp.hovered() {
                    TEXT
                } else {
                    FAINT
                },
            );
        };
        header(
            ui,
            "Name",
            SortCol::Name,
            hr.left() + icon_w + 6.0,
            egui::Align2::LEFT_CENTER,
        );
        ui.painter().text(
            Pos2::new(hr.left() + icon_w + name_w + user_w - 6.0, hr.center().y),
            egui::Align2::RIGHT_CENTER,
            "User",
            FontId::proportional(12.0),
            FAINT,
        );
        header(
            ui,
            "PID",
            SortCol::Pid,
            hr.left() + icon_w + name_w + user_w + pid_w - 6.0,
            egui::Align2::RIGHT_CENTER,
        );
        header(
            ui,
            "CPU",
            SortCol::Cpu,
            hr.left() + icon_w + name_w + user_w + pid_w + cpu_w - 6.0,
            egui::Align2::RIGHT_CENTER,
        );
        header(
            ui,
            "Memory",
            SortCol::Mem,
            hr.right() - 6.0,
            egui::Align2::RIGHT_CENTER,
        );
        if let Some(col) = new_sort {
            if self.gui.proc_sort == col {
                self.gui.proc_sort_desc = !self.gui.proc_sort_desc;
            } else {
                self.gui.proc_sort = col;
                // text columns read best ascending, numbers descending
                self.gui.proc_sort_desc = !matches!(col, SortCol::Name | SortCol::Pid);
            }
        }

        // ---- rows ----
        let sel = self.gui.selected_proc;
        let mut new_sel: Option<Option<u32>> = None;
        let mut new_expand: Option<Option<String>> = None;
        let cur_expand = self.gui.proc_group.clone();
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for e in &entries {
                    // a resolved desktop icon means "this is an app": even a
                    // one-process app is a group row, so clicking it still
                    // reveals its process(es)
                    let is_app = !e.icon.is_empty();
                    let open = cur_expand.as_deref() == Some(e.icon.as_str());
                    let key: (&str, u64) = if is_app {
                        ("proc-group", egui::Id::new(&e.icon).value())
                    } else {
                        ("proc-row", e.members[0].pid as u64)
                    };
                    let rect = ui.allocate_exact_size(Vec2::new(w, 30.0), Sense::click()).0;
                    let clicked = ui
                        .interact(rect, ui.id().with(key), Sense::click())
                        .clicked();
                    let is_sel = !is_app && sel == Some(e.members[0].pid);
                    if clicked {
                        if is_app {
                            new_expand = Some(if open { None } else { Some(e.icon.clone()) });
                        } else {
                            new_sel = Some(if is_sel { None } else { Some(e.members[0].pid) });
                        }
                    }
                    if !ui.is_rect_visible(rect) {
                        if open {
                            for _ in &e.members {
                                ui.allocate_exact_size(Vec2::new(w, 26.0), Sense::hover());
                            }
                        }
                        continue;
                    }
                    row_bg(ui.painter(), rect, is_sel, false);
                    let color = if e.cpu > 80.0 {
                        RED
                    } else if e.cpu > 30.0 {
                        AMBER
                    } else {
                        TEXT
                    };
                    // app icon: the real application icon when the system
                    // theme has one, otherwise a tinted tile with a drawn
                    // glyph per process family (browser, terminal, …)
                    let tile = Rect::from_center_size(
                        rect.left_center() + Vec2::new(15.0, 0.0),
                        Vec2::splat(24.0),
                    );
                    let ic = proc_icon_color(e.kind);
                    ui.painter().rect_filled(tile, 7.0, ic.gamma_multiply(0.14));
                    match self.gui.icons.get(&e.icon) {
                        Some(tex) => {
                            let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
                            ui.painter().image(tex.id(), tile, uv, Color32::WHITE);
                        }
                        None => icon_glyph(ui.painter(), e.kind, tile.center(), 8.0, ic),
                    }
                    let x_name = rect.left() + icon_w + 6.0;
                    let mut title = e.title.clone();
                    if !e.sub.is_empty() {
                        title.push_str("  ·  ");
                        title.push_str(&e.sub);
                    }
                    let name = elide(
                        ui.painter(),
                        title,
                        FontId::proportional(13.0),
                        color,
                        name_w - 24.0,
                    );
                    let name_w_px = name.size().x;
                    ui.painter().galley(
                        Pos2::new(x_name, rect.center().y - name.size().y / 2.0),
                        name,
                        color,
                    );
                    if is_app {
                        // disclosure chevron in the name cell
                        let cx = x_name + name_w_px + 8.0;
                        let cy = rect.center().y;
                        let s = Stroke::new(1.4_f32, FAINT);
                        let (a, b, c2) = if open {
                            (
                                Pos2::new(cx, cy - 3.0),
                                Pos2::new(cx + 4.0, cy + 2.0),
                                Pos2::new(cx + 8.0, cy - 3.0),
                            )
                        } else {
                            (
                                Pos2::new(cx, cy - 4.0),
                                Pos2::new(cx + 4.0, cy),
                                Pos2::new(cx, cy + 4.0),
                            )
                        };
                        ui.painter().line_segment([a, b], s);
                        ui.painter().line_segment([b, c2], s);
                    }
                    ui.painter().text(
                        Pos2::new(
                            rect.left() + icon_w + name_w + user_w - 6.0,
                            rect.center().y,
                        ),
                        egui::Align2::RIGHT_CENTER,
                        &e.user,
                        FontId::proportional(11.5),
                        if e.user == "root" { AMBER } else { MUTED },
                    );
                    ui.painter().text(
                        Pos2::new(
                            rect.left() + icon_w + name_w + user_w + pid_w - 6.0,
                            rect.center().y,
                        ),
                        egui::Align2::RIGHT_CENTER,
                        &e.pid_label,
                        FontId::monospace(12.0),
                        MUTED,
                    );
                    ui.painter().text(
                        Pos2::new(
                            rect.left() + icon_w + name_w + user_w + pid_w + cpu_w - 6.0,
                            rect.center().y,
                        ),
                        egui::Align2::RIGHT_CENTER,
                        format!("{:.1}", e.cpu),
                        FontId::monospace(12.0),
                        color,
                    );
                    ui.painter().text(
                        Pos2::new(rect.right() - 6.0, rect.center().y),
                        egui::Align2::RIGHT_CENTER,
                        human_size(e.mem as i64),
                        FontId::monospace(12.0),
                        MUTED,
                    );
                    // expanded: the individual processes behind this app
                    if open {
                        for m in &e.members {
                            let rrect =
                                ui.allocate_exact_size(Vec2::new(w, 26.0), Sense::click()).0;
                            if !ui.is_rect_visible(rrect) {
                                continue;
                            }
                            ui.painter().line_segment(
                                [
                                    Pos2::new(rect.left() + 27.0, rrect.top()),
                                    Pos2::new(rect.left() + 27.0, rrect.bottom()),
                                ],
                                Stroke::new(1.0_f32, BORDER),
                            );
                            let msel = sel == Some(m.pid);
                            if msel {
                                ui.painter().rect_filled(
                                    Rect::from_min_max(
                                        Pos2::new(rect.left() + 28.0, rrect.top()),
                                        rrect.max,
                                    ),
                                    8.0,
                                    ACCENT.gamma_multiply(0.10),
                                );
                            }
                            if ui
                                .interact(
                                    rrect,
                                    ui.id().with(("proc-member", m.pid)),
                                    Sense::click(),
                                )
                                .clicked()
                            {
                                new_sel = Some(if msel { None } else { Some(m.pid) });
                            }
                            let x = rect.left() + 40.0;
                            ui.painter().circle_filled(
                                Pos2::new(x, rrect.center().y),
                                2.5,
                                proc_icon_color(m.icon),
                            );
                            let nm = elide(
                                ui.painter(),
                                m.name.clone(),
                                FontId::proportional(12.0),
                                MUTED,
                                name_w - 28.0,
                            );
                            ui.painter().galley(
                                Pos2::new(x + 9.0, rrect.center().y - nm.size().y / 2.0),
                                nm,
                                if msel { TEXT } else { MUTED },
                            );
                            ui.painter().text(
                                Pos2::new(
                                    rect.left() + icon_w + name_w + user_w + pid_w - 6.0,
                                    rrect.center().y,
                                ),
                                egui::Align2::RIGHT_CENTER,
                                m.pid.to_string(),
                                FontId::monospace(11.5),
                                MUTED,
                            );
                            ui.painter().text(
                                Pos2::new(
                                    rect.left() + icon_w + name_w + user_w + pid_w + cpu_w - 6.0,
                                    rrect.center().y,
                                ),
                                egui::Align2::RIGHT_CENTER,
                                format!("{:.1}", m.cpu_pct),
                                FontId::monospace(11.5),
                                if m.cpu_pct > 30.0 { AMBER } else { MUTED },
                            );
                            ui.painter().text(
                                Pos2::new(rect.right() - 6.0, rrect.center().y),
                                egui::Align2::RIGHT_CENTER,
                                human_size(m.rss_bytes as i64),
                                FontId::monospace(11.5),
                                MUTED,
                            );
                        }
                    }
                }
            });
        if let Some(s) = new_sel {
            self.gui.selected_proc = s;
        }
        if let Some(x) = new_expand {
            self.gui.proc_group = x;
        }
    }

    fn proc_detail(&mut self, ui: &mut Ui) {
        let Some(pid) = self.gui.selected_proc else {
            ui.label(
                RichText::new("Click a row above to inspect a process — full command line, threads, owner and process tree.")
                    .size(12.5)
                    .color(FAINT),
            );
            return;
        };
        let Some(r) = self.gui.procs.iter().find(|r| r.pid == pid).cloned() else {
            ui.label(
                RichText::new("That process exited; pick another row.")
                    .size(12.5)
                    .color(FAINT),
            );
            return;
        };
        let now = epoch();
        let up = (now - r.started).max(0);
        card(ui, |ui| {
            ui.horizontal(|ui| {
                // the app's real icon, bigger than in the table
                if let Some(tex) = self.gui.icons.get(&r.app_icon) {
                    let (rect, _) = ui.allocate_exact_size(Vec2::splat(28.0), Sense::hover());
                    let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
                    ui.painter().image(tex.id(), rect, uv, Color32::WHITE);
                    ui.add_space(4.0);
                }
                ui.label(RichText::new(&r.name).font(bold(16.0)).color(TEXT));
                let cpu_color = if r.cpu_pct > 80.0 {
                    RED
                } else if r.cpu_pct > 30.0 {
                    AMBER
                } else {
                    ACCENT
                };
                ui.label(
                    RichText::new(format!("{:.1}% cpu", r.cpu_pct))
                        .font(bold(13.0))
                        .color(cpu_color),
                );
                ui.label(
                    RichText::new(format!("· {}", human_size(r.rss_bytes as i64)))
                        .size(13.0)
                        .color(VIOLET),
                );
                let st: (&str, Color32) = match r.state {
                    'R' => ("running", GREEN),
                    'S' | 'I' => ("sleeping", MUTED),
                    'D' => ("uninterruptible", AMBER),
                    'Z' => ("zombie", RED),
                    'T' => ("stopped", AMBER),
                    _ => ("other", FAINT),
                };
                chip(ui, st.0, st.1);
                ui.label(
                    RichText::new(format!(
                        "pid {} · {} threads · user {} · up {}",
                        r.pid,
                        r.threads,
                        r.user,
                        uptime_str(up)
                    ))
                    .size(12.0)
                    .color(FAINT),
                );
            });
            if !r.cmdline.is_empty() {
                ui.add(
                    egui::Label::new(
                        RichText::new(&r.cmdline)
                            .font(FontId::monospace(12.0))
                            .color(MUTED),
                    )
                    .wrap(),
                );
            }
            ui.add_space(2.0);
            ui.label(
                RichText::new(format!(
                    "vm {} · resident {} · io {} read / {} written · cwd {}",
                    human_size(r.vsize_bytes as i64),
                    human_size(r.rss_bytes as i64),
                    human_size(r.io_read as i64),
                    human_size(r.io_write as i64),
                    if r.cwd.is_empty() { "—" } else { &r.cwd }
                ))
                .size(11.5)
                .color(FAINT),
            );
            // ancestors up to init, then direct children
            let mut chain = Vec::new();
            let mut cur = r.ppid;
            let mut guard = 0;
            while guard < 12 {
                match self.gui.procs.iter().find(|x| x.pid == cur) {
                    Some(a) => {
                        chain.push(format!("{} (pid {})", a.name, a.pid));
                        cur = a.ppid;
                        if cur == 0 || cur == a.pid {
                            break;
                        }
                    }
                    None => break,
                }
                guard += 1;
            }
            if !chain.is_empty() {
                ui.add_space(2.0);
                ui.label(RichText::new(chain.join(" ← ")).size(11.5).color(FAINT));
            }
            let kids: Vec<String> = self
                .gui
                .procs
                .iter()
                .filter(|x| x.ppid == r.pid)
                .map(|x| format!("{} ({})", x.name, x.pid))
                .collect();
            if !kids.is_empty() {
                ui.label(
                    RichText::new(format!("children: {}", kids.join(", ")))
                        .size(11.5)
                        .color(FAINT),
                );
            }
        });
    }

    fn volume_list(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        let f = g.filter.to_lowercase();
        let sel = g.selected_volume.clone();
        let rows: Vec<&docker::VolumeEntry> = g
            .volumes
            .iter()
            .filter(|v| f.is_empty() || v.name.to_lowercase().contains(&f))
            .collect();
        if rows.is_empty() {
            empty_state(ui, "No volumes match", "Try another search.");
            return;
        }
        let mut new_sel: Option<Option<String>> = None;
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 4.0;
                for v in rows {
                    let (rect, resp) = ui
                        .allocate_exact_size(Vec2::new(ui.available_width(), 58.0), Sense::click());
                    let is_sel = sel.as_deref() == Some(v.name.as_str());
                    if resp.clicked() {
                        new_sel = Some(if is_sel { None } else { Some(v.name.clone()) });
                    }
                    if !ui.is_rect_visible(rect) {
                        continue;
                    }
                    let p = ui.painter();
                    row_bg(p, rect, is_sel, resp.hovered());
                    let refs = v.usage_data.as_ref().map(|u| u.ref_count);
                    icon_tile(
                        p,
                        rect,
                        Tab::Volumes,
                        if refs != Some(0) { VIOLET } else { FAINT },
                    );
                    let left = rect.left() + 54.0;
                    let text_w = (rect.width() - 54.0 - 86.0).max(40.0);
                    let name = elide(p, v.name.clone(), FontId::proportional(14.0), TEXT, text_w);
                    p.galley(Pos2::new(left, rect.top() + 10.0), name, TEXT);
                    let sub = elide(
                        p,
                        match refs {
                            Some(1) => format!("{} · 1 ref", v.driver),
                            Some(n) => format!("{} · {n} refs", v.driver),
                            None => format!(
                                "{} · created {}",
                                v.driver,
                                v.created.get(..10).unwrap_or("?")
                            ),
                        },
                        FontId::proportional(12.0),
                        MUTED,
                        text_w,
                    );
                    p.galley(Pos2::new(left, rect.top() + 32.0), sub, MUTED);
                    let size = v
                        .usage_data
                        .as_ref()
                        .and_then(|u| u.size)
                        .filter(|s| *s >= 0)
                        .map(human_size)
                        .unwrap_or_default();
                    p.text(
                        Pos2::new(rect.right() - 12.0, rect.center().y),
                        egui::Align2::RIGHT_CENTER,
                        size,
                        FontId::monospace(12.0),
                        MUTED,
                    );
                }
            });
        if let Some(s) = new_sel {
            self.gui.selected_volume = s;
        }
    }

    fn center(&mut self, ctx: &egui::Context) {
        let tab = self.gui.tab;
        egui::CentralPanel::default()
            .frame(Frame::NONE.fill(BG).inner_margin(Margin {
                left: 28,
                right: 28,
                top: 22,
                bottom: 18,
            }))
            .show(ctx, |ui| {
                self.error_banner(ui);
                match tab {
                    Tab::Dashboard => self.dashboard_view(ui),
                    Tab::Logs => self.logs_view(ui),
                    Tab::Containers => self.container_detail(ui),
                    Tab::Images => self.image_detail(ui),
                    Tab::Volumes => self.volume_detail(ui),
                    Tab::Agents => self.agent_detail(ui),
                    Tab::Hardware => {
                        let usage = hwinv::disk_usage();
                        self.hardware_view(ui, &hwinv::INVENTORY, &usage)
                    }
                    Tab::Ports => self.port_detail(ui),
                    Tab::Processes => {
                        self.proc_search(ui);
                        ui.add_space(8.0);
                        self.proc_list(ui);
                        ui.add_space(10.0);
                        self.proc_detail(ui);
                    }
                    Tab::Strata => self.strata_view(ui),
                    Tab::Settings => self.settings_view(ui),
                }
            });
    }

    fn error_banner(&mut self, ui: &mut Ui) {
        let Some(e) = self.gui.error.clone() else {
            return;
        };
        let mut dismiss = false;
        Frame::NONE
            .fill(RED.gamma_multiply(0.10))
            .stroke(Stroke::new(1.0_f32, RED.gamma_multiply(0.45)))
            .corner_radius(10)
            .inner_margin(Margin::symmetric(14, 10))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(RichText::new("⚠").size(15.0).color(RED));
                    let w = ui.available_width() - 80.0;
                    ui.allocate_ui(Vec2::new(w, 0.0), |ui| {
                        ui.add(egui::Label::new(RichText::new(&e).size(13.0).color(TEXT)).wrap());
                    });
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add(
                                egui::Button::new(RichText::new("Dismiss").size(12.5).color(MUTED))
                                    .frame(false),
                            )
                            .clicked()
                        {
                            dismiss = true;
                        }
                    });
                });
            });
        ui.add_space(14.0);
        if dismiss {
            self.gui.error = None;
        }
    }

    /// Short-lived status toast in the bottom-right corner.
    fn toast(&mut self, ctx: &egui::Context) {
        let (text, at) = &self.toast;
        let age = at.elapsed().as_secs_f32();
        if text.is_empty() || age > 4.0 {
            return;
        }
        let alpha = ((4.0 - age) / 0.4).clamp(0.0, 1.0) * (age / 0.15).clamp(0.0, 1.0);
        let lift = 8.0 * (1.0 - (age / 0.2).clamp(0.0, 1.0));
        egui::Area::new(egui::Id::new("toast"))
            .anchor(egui::Align2::RIGHT_BOTTOM, Vec2::new(-24.0, -20.0 + lift))
            .interactable(false)
            .show(ctx, |ui| {
                ui.set_opacity(alpha);
                Frame::NONE
                    .fill(ELEVATED)
                    .stroke(Stroke::new(1.0_f32, Color32::from_rgb(56, 64, 80)))
                    .corner_radius(10)
                    .shadow(ctx.style().visuals.popup_shadow)
                    .inner_margin(Margin::symmetric(14, 10))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let (r, _) = ui.allocate_exact_size(Vec2::splat(8.0), Sense::hover());
                            ui.painter().circle_filled(r.center(), 4.0, GREEN);
                            ui.label(RichText::new(capitalize(text)).size(13.0).color(TEXT));
                        });
                    });
            });
        ctx.request_repaint_after(Duration::from_millis(33));
    }

    // ------------------------------------------------------------------
    // Dashboard: one high-level page across every feature. Health first
    // (only what needs a look), then the machine's vitals, then one card
    // per workload — containers, agents, ports, Strata — each opening its
    // tab for the details.
    // ------------------------------------------------------------------

    fn dashboard_view(&mut self, ui: &mut Ui) {
        let sub = match &self.gui.hw {
            Some(h) => format!(
                "{} · up {}",
                hwinv::INVENTORY.cpu.trim_end_matches(" Processor"),
                uptime_str(h.uptime as i64)
            ),
            None => "Collecting host vitals…".to_string(),
        };
        ui.horizontal(|ui| {
            ui.vertical(|ui| page_header(ui, "Dashboard", &sub));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                live_pill(ui, true);
            });
        });
        ui.add_space(16.0);
        let mut nav = None;
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                nav = self.health_strip(ui);
                ui.add_space(14.0);
                self.vital_tiles(ui);
                ui.add_space(14.0);
                nav = nav.or(self.workload_cards(ui));
                ui.add_space(8.0);
            });
        if let Some(t) = nav {
            self.on_tab_change(t);
        }
    }

    /// Everything that needs a look, worst first, each row opening the tab
    /// that explains it; a single green line when nothing does.
    fn health_strip(&self, ui: &mut Ui) -> Option<Tab> {
        let g = &self.gui;
        let level = |v: f64, warn: f64, bad: f64| {
            if v >= bad {
                Some(RED)
            } else if v >= warn {
                Some(AMBER)
            } else {
                None
            }
        };
        let hottest = |it: &mut dyn Iterator<Item = f64>| {
            it.fold(None::<f64>, |a, b| Some(a.map_or(b, |a| a.max(b))))
        };
        let mut issues: Vec<(Color32, String, Tab)> = Vec::new();
        if let Some(h) = &g.hw {
            let mem = (h.mem_total - h.mem_avail) as f64 / h.mem_total.max(1) as f64 * 100.0;
            if let Some(c) = level(mem, 80.0, 92.0) {
                issues.push((
                    c,
                    format!(
                        "Memory {mem:.0}% used · {} free",
                        human_size(h.mem_avail as i64)
                    ),
                    Tab::Hardware,
                ));
            }
            if h.swap_total > 0 {
                let swap =
                    h.swap_total.saturating_sub(h.swap_free) as f64 / h.swap_total as f64 * 100.0;
                if let Some(c) = level(swap, 50.0, 85.0) {
                    issues.push((c, format!("Swap {swap:.0}% used"), Tab::Hardware));
                }
            }
            if h.load5 > h.ncpu as f64 {
                issues.push((
                    AMBER,
                    format!("Load {:.1} is above {} CPUs", h.load5, h.ncpu),
                    Tab::Hardware,
                ));
            }
            let gpu_t = h
                .gpu_temp_c
                .or_else(|| hottest(&mut g.gpus.iter().filter_map(|x| x.temp_c)));
            let ssd_t = hottest(&mut h.disks.iter().filter_map(|d| d.temp_c));
            for (what, t, warn, bad) in [
                ("CPU", h.cpu_temp_c.or(h.temp_c), 80.0, 90.0),
                ("GPU", gpu_t, 80.0, 90.0),
                ("SSD", ssd_t, 60.0, 70.0),
            ] {
                if let Some((c, t)) = t.and_then(|t| level(t, warn, bad).map(|c| (c, t))) {
                    issues.push((
                        c,
                        format!("{what} running hot at {} °C", temp_str(t)),
                        Tab::Hardware,
                    ));
                }
            }
        }
        let restarting = g
            .containers
            .iter()
            .filter(|c| c.state == "restarting")
            .count();
        if restarting > 0 {
            issues.push((
                RED,
                format!("{restarting} container(s) stuck restarting"),
                Tab::Containers,
            ));
        }
        // exited non-zero recently ("Exited (1) 3 minutes ago"); old corpses
        // from last week are not news
        let crashed = g
            .containers
            .iter()
            .filter(|c| c.state == "exited" && !c.status.contains("(0)"))
            .filter(|c| c.status.contains("second") || c.status.contains("minute"))
            .count();
        if crashed > 0 {
            issues.push((
                AMBER,
                format!("{crashed} container(s) exited with an error in the last hour"),
                Tab::Containers,
            ));
        }
        let waiting = g.agents.iter().filter(|a| agent_state(a).rank == 0).count();
        if waiting > 0 {
            issues.push((
                AMBER,
                format!("{waiting} agent session(s) waiting for your input"),
                Tab::Agents,
            ));
        }
        if !g.strata_cfg.api_key.trim().is_empty() && g.strata_error.is_some() {
            issues.push((AMBER, "Strata server is not responding".into(), Tab::Strata));
        }
        if let Some(df) = &g.df {
            let reclaim: i64 = [
                &df.image_usage,
                &df.container_usage,
                &df.volume_usage,
                &df.build_cache_usage,
            ]
            .into_iter()
            .flatten()
            .map(|u| u.reclaimable)
            .sum();
            if reclaim >= 10 << 30 {
                issues.push((
                    ACCENT,
                    format!("{} of Docker data can be reclaimed", human_size(reclaim)),
                    Tab::Images,
                ));
            }
        }
        issues.sort_by_key(|(c, _, _)| [RED, AMBER, ACCENT].iter().position(|x| x == c));

        let mut nav = None;
        card(ui, |ui| {
            if issues.is_empty() {
                ui.horizontal(|ui| {
                    let (r, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
                    ui.painter().circle_filled(r.center(), 4.0, GREEN);
                    ui.label(
                        RichText::new("All systems normal")
                            .font(bold(13.5))
                            .color(TEXT),
                    );
                    ui.label(
                        RichText::new("memory, temperatures, containers and Strata look healthy")
                            .size(12.5)
                            .color(FAINT),
                    );
                });
                return;
            }
            card_title(
                ui,
                "Needs attention",
                &format!(
                    "{} issue{}",
                    issues.len(),
                    if issues.len() == 1 { "" } else { "s" }
                ),
            );
            ui.spacing_mut().item_spacing.y = 0.0;
            for (color, text, tab) in &issues {
                let (rect, resp) =
                    ui.allocate_exact_size(Vec2::new(ui.available_width(), 30.0), Sense::click());
                let p = ui.painter();
                if resp.hovered() {
                    p.rect_filled(rect, 8.0, SURFACE_HI);
                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                }
                p.circle_filled(Pos2::new(rect.left() + 10.0, rect.center().y), 4.0, *color);
                p.text(
                    Pos2::new(rect.left() + 24.0, rect.center().y),
                    egui::Align2::LEFT_CENTER,
                    text,
                    FontId::proportional(13.0),
                    TEXT,
                );
                p.text(
                    Pos2::new(rect.right() - 8.0, rect.center().y),
                    egui::Align2::RIGHT_CENTER,
                    format!("{} ›", tab_label(*tab)),
                    FontId::proportional(12.0),
                    if resp.hovered() { ACCENT } else { FAINT },
                );
                if resp.clicked() {
                    nav = Some(*tab);
                }
            }
        });
        nav
    }

    /// Machine vitals: each tile is the live number, its context and its
    /// recent history — CPU, memory (+swap), GPU (disk when no GPU
    /// reports), network.
    fn vital_tiles(&self, ui: &mut Ui) {
        let g = &self.gui;
        let hist = |q: &VecDeque<f64>| q.iter().copied().collect::<Vec<f64>>();
        let sum = |a: &VecDeque<f64>, b: &VecDeque<f64>| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| x + y)
                .collect::<Vec<f64>>()
        };
        let Some(h) = &g.hw else {
            card(ui, |ui| {
                ui.label(RichText::new("Collecting host vitals…").color(FAINT));
            });
            return;
        };
        let mem_used = h.mem_total - h.mem_avail;
        let swap_used = h.swap_total.saturating_sub(h.swap_free);
        let mut cpu_sub = vec![format!("load {:.2}", h.load1)];
        if let Some(t) = h.cpu_temp_c.or(h.temp_c) {
            cpu_sub.insert(0, format!("{} °C", temp_str(t)));
        }
        cpu_sub.push(format!("{} threads", h.ncpu));
        let mut tiles = vec![
            Kpi {
                label: "CPU",
                device: String::new(),
                value: format!("{:.0}%", h.cpu_pct),
                sub: cpu_sub.join(" · "),
                color: ACCENT,
                viz: Viz::Spark(hist(&g.hw_cpu_hist)),
                glyph: Some("cpu"),
            },
            Kpi {
                label: "Memory",
                device: String::new(),
                value: format!(
                    "{:.0}%",
                    mem_used as f64 / h.mem_total.max(1) as f64 * 100.0
                ),
                sub: format!(
                    "{} of {}{}",
                    human_size(mem_used as i64),
                    human_size(h.mem_total as i64),
                    if swap_used > 0 {
                        format!(" · swap {}", human_size(swap_used as i64))
                    } else {
                        String::new()
                    }
                ),
                color: VIOLET,
                viz: Viz::Spark(hist(&g.hw_mem_hist)),
                glyph: Some("ram"),
            },
        ];
        let gpu = g
            .gpus
            .iter()
            .filter(|x| x.busy_pct.is_some())
            .max_by(|a, b| {
                a.busy_pct
                    .partial_cmp(&b.busy_pct)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        tiles.push(match gpu {
            Some(gp) => {
                let mut sub = Vec::new();
                if let Some(t) = gp.temp_c {
                    sub.push(format!("{} °C", temp_str(t)));
                }
                if gp.vram_total > 0 {
                    sub.push(format!(
                        "VRAM {} of {}",
                        human_size(gp.vram_used as i64),
                        human_size(gp.vram_total as i64)
                    ));
                }
                if sub.is_empty() {
                    sub.push(gp.name.clone());
                }
                Kpi {
                    label: "GPU",
                    device: String::new(),
                    value: format!("{:.0}%", gp.busy_pct.unwrap_or(0.0)),
                    sub: sub.join(" · "),
                    color: CYAN,
                    viz: Viz::Spark(hist(&g.hw_gpu_busy_hist)),
                    glyph: Some("gpu"),
                }
            }
            None => Kpi {
                label: "Disk I/O",
                device: String::new(),
                value: human_rate(h.disk_read_bps + h.disk_write_bps),
                sub: format!(
                    "read {} · write {}",
                    human_rate(h.disk_read_bps),
                    human_rate(h.disk_write_bps)
                ),
                color: AMBER,
                viz: Viz::Spark(sum(&g.hw_disk_r_hist, &g.hw_disk_w_hist)),
                glyph: Some("disk"),
            },
        });
        tiles.push(Kpi {
            label: "Network",
            device: String::new(),
            value: human_rate(h.net_rx_bps + h.net_tx_bps),
            sub: format!(
                "↓ {} · ↑ {}",
                human_rate(h.net_rx_bps),
                human_rate(h.net_tx_bps)
            ),
            color: GREEN,
            viz: Viz::Spark(sum(&g.hw_net_rx_hist, &g.hw_net_tx_hist)),
            glyph: Some("net"),
        });
        let per_row = if ui.available_width() > 860.0 { 4 } else { 2 };
        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.x = 14.0;
            for (n, row) in tiles.chunks(per_row).enumerate() {
                if n > 0 {
                    ui.add_space(6.0);
                }
                ui.columns(per_row, |cols| {
                    for (col, k) in cols.iter_mut().zip(row) {
                        kpi_card(col, k);
                    }
                });
            }
        });
    }

    /// One card per workload, two per row; returns the tab to open when a
    /// card is clicked.
    fn workload_cards(&self, ui: &mut Ui) -> Option<Tab> {
        let mut nav = None;
        let pick = |nav: &mut Option<Tab>, t: Tab| {
            if nav.is_none() {
                *nav = Some(t);
            }
        };
        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.x = 14.0;
            ui.columns(2, |cols| {
                if overview_card(&mut cols[0], Tab::Containers, "Containers", |ui| {
                    self.containers_overview(ui)
                }) {
                    pick(&mut nav, Tab::Containers);
                }
                if overview_card(&mut cols[1], Tab::Strata, "Strata", |ui| {
                    self.strata_overview(ui)
                }) {
                    // nothing to look at until it's configured
                    pick(
                        &mut nav,
                        if self.gui.strata_cfg.api_key.trim().is_empty() {
                            Tab::Settings
                        } else {
                            Tab::Strata
                        },
                    );
                }
            });
            ui.add_space(6.0);
            ui.columns(2, |cols| {
                if overview_card(&mut cols[0], Tab::Agents, "Agents", |ui| {
                    self.agents_overview(ui)
                }) {
                    pick(&mut nav, Tab::Agents);
                }
                if overview_card(&mut cols[1], Tab::Ports, "Ports", |ui| {
                    self.ports_overview(ui)
                }) {
                    pick(&mut nav, Tab::Ports);
                }
            });
            ui.add_space(6.0);
            ui.columns(2, |cols| {
                if overview_card(&mut cols[0], Tab::Processes, "Processes", |ui| {
                    self.processes_overview(ui)
                }) {
                    pick(&mut nav, Tab::Processes);
                }
                if overview_card(&mut cols[1], Tab::Hardware, "Hardware", |ui| {
                    self.hardware_overview(ui)
                }) {
                    pick(&mut nav, Tab::Hardware);
                }
            });
        });
        nav
    }

    fn containers_overview(&self, ui: &mut Ui) {
        let g = &self.gui;
        if !g.daemon_ok && g.error.is_some() {
            headline(ui, "Offline", "Docker engine unreachable", RED);
            return;
        }
        let n_run = g.containers.iter().filter(|c| c.state == "running").count();
        headline(
            ui,
            &n_run.to_string(),
            &format!("running of {}", g.containers.len()),
            if n_run > 0 { GREEN } else { TEXT },
        );
        // share of the whole host, like the CPU tile above (Docker reports
        // 100% per core)
        let ncpu = g.hw.as_ref().map(|h| h.ncpu.max(1)).unwrap_or(1) as f64;
        let mut rows: Vec<(String, f64, i64)> = g
            .containers
            .iter()
            .filter(|c| c.state == "running")
            .filter_map(|c| {
                let s = g.stats.get(&c.id)?;
                let cpu = StatsSample::cpu_percent(g.prev_stats.get(&c.id), s).unwrap_or(0.0);
                Some((c.short_name(), cpu.max(0.0) / ncpu, s.mem_usage()))
            })
            .collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let cpu: f64 = rows.iter().map(|r| r.1).sum();
        let mem: i64 = rows.iter().map(|r| r.2).sum();
        let disk = g.df.as_ref().map(|d| {
            [
                &d.image_usage,
                &d.container_usage,
                &d.volume_usage,
                &d.build_cache_usage,
            ]
            .into_iter()
            .flatten()
            .map(|u| u.total_size)
            .sum::<i64>()
        });
        let mut meta = vec![format!("{} images", g.images.len())];
        if !rows.is_empty() {
            meta.insert(0, format!("{cpu:.1}% CPU · {}", human_size(mem)));
        }
        if let Some(d) = disk {
            meta.push(format!("{} on disk", human_size(d)));
        }
        meta_line(ui, &meta.join(" · "));
        for (name, cpu, mem) in rows.iter().take(3) {
            mini_row(
                ui,
                GREEN,
                name,
                &format!("{cpu:.1}% · {}", human_size(*mem)),
            );
        }
        if rows.is_empty() {
            meta_line(
                ui,
                if n_run > 0 {
                    "Collecting container stats…"
                } else {
                    "No running containers."
                },
            );
        }
        if let Some(e) = g.events.last() {
            let action = e.action.split(':').next().unwrap_or(&e.action);
            let color = match action {
                "start" | "create" | "unpause" => GREEN,
                "die" | "kill" | "stop" | "oom" => RED,
                _ => FAINT,
            };
            mini_row(
                ui,
                color,
                &format!("{} {action}", e.container_name()),
                &ago(epoch() - e.time),
            );
        }
    }

    fn agents_overview(&self, ui: &mut Ui) {
        let g = &self.gui;
        let n = g.agents.len();
        headline(
            ui,
            &n.to_string(),
            "AI coding sessions",
            if n > 0 { GREEN } else { TEXT },
        );
        if n == 0 {
            meta_line(ui, "No agents running.");
            return;
        }
        let mut top: Vec<(&AgentSession, AgentState)> =
            g.agents.iter().map(|a| (a, agent_state(a))).collect();
        top.sort_by_key(|(a, st)| (st.rank, a.lead_pid));
        let count = |rank: u8| top.iter().filter(|(_, st)| st.rank == rank).count();
        let mut meta = Vec::new();
        for (rank, what) in [(0, "need you"), (1, "working"), (2, "idle")] {
            let c = count(rank);
            if c > 0 {
                meta.push(format!("{c} {what}"));
            }
        }
        meta_line(ui, &meta.join(" · "));
        for (a, st) in top.iter().take(3) {
            mini_row(ui, st.color, &agent_title(a), st.label);
        }
    }

    fn ports_overview(&self, ui: &mut Ui) {
        let g = &self.gui;
        let listen: Vec<&PortEntry> = g.ports.iter().filter(|p| p.state == "LISTEN").collect();
        let exposed = listen
            .iter()
            .filter(|p| p.addr == "0.0.0.0" || p.addr == "[::]")
            .count();
        headline(ui, &listen.len().to_string(), "listening ports", CYAN);
        meta_line(ui, &format!("{exposed} open on all interfaces"));
        // dev servers first: sockets whose owner runs in a project directory
        let mut picks: Vec<&PortEntry> =
            listen.iter().copied().filter(|p| p.cwd.is_some()).collect();
        if picks.is_empty() {
            picks = listen
                .iter()
                .copied()
                .filter(|p| p.proc_name.is_some())
                .collect();
        }
        picks.sort_by_key(|p| p.port);
        picks.dedup_by_key(|p| p.port);
        let home = std::env::var("HOME").unwrap_or_default();
        for p in picks.iter().take(3) {
            let name = format!(":{} {}", p.port, p.proc_name.as_deref().unwrap_or("?"));
            let dir = match p.cwd.as_deref() {
                Some(d) if d.trim_end_matches('/') == home => "~".to_string(),
                Some(d) => basename(d),
                None => String::new(),
            };
            mini_row(ui, port_color(p), &name, &dir);
        }
    }

    /// Dashboard card for the process list: how many run, and the hogs.
    fn processes_overview(&self, ui: &mut Ui) {
        let g = &self.gui;
        if g.procs.is_empty() {
            headline(ui, "—", "processes", FAINT);
            meta_line(ui, "The /proc walk has not landed yet.");
            return;
        }
        headline(
            ui,
            &g.procs.len().to_string(),
            "processes running",
            if !g.procs.is_empty() { TEXT } else { FAINT },
        );
        let mut cpu: Vec<ProcInfo> = g.procs.clone();
        procsmod::sort_rows(&mut cpu, SortCol::Cpu, true);
        let mut mem: Vec<ProcInfo> = g.procs.clone();
        procsmod::sort_rows(&mut mem, SortCol::Mem, true);
        // top three by CPU, then the single biggest memory holder, so the
        // card answers "what is eating my machine right now"
        let mut picks: Vec<(&ProcInfo, String)> = cpu
            .iter()
            .take(3)
            .map(|r| (r, format!("{:.0}% cpu", r.cpu_pct)))
            .collect();
        if !mem.is_empty() && !picks.iter().any(|(r, _)| r.pid == mem[0].pid) {
            picks.push((&mem[0], human_size(mem[0].rss_bytes as i64)));
        }
        for (r, v) in picks.into_iter().take(4) {
            // the icon's own average colour when we already decoded it,
            // otherwise the family colour
            let dot = iconsmod::tint_cached(&r.app_icon).unwrap_or_else(|| proc_icon_color(r.icon));
            mini_row(ui, dot, &format!("{} · pid {}", r.name, r.pid), &v);
        }
    }

    /// Dashboard card for the Hardware tab: what the machine is doing right
    /// now, so the vitals are readable without leaving the dashboard.
    fn hardware_overview(&self, ui: &mut Ui) {
        let g = &self.gui;
        let Some(h) = &g.hw else {
            headline(ui, "—", "host vitals", FAINT);
            meta_line(ui, "Collecting host vitals…");
            return;
        };
        let mem = h.mem_total - h.mem_avail;
        headline(
            ui,
            &format!("{:.0}%", h.cpu_pct),
            &format!("cpu · {} cores", h.ncpu),
            if h.cpu_pct > 85.0 {
                RED
            } else if h.cpu_pct > 60.0 {
                AMBER
            } else {
                ACCENT
            },
        );
        meta_line(
            ui,
            &format!("load {} · up {}", h.load1, uptime_str(h.uptime as i64)),
        );
        mini_row(
            ui,
            VIOLET,
            "Memory",
            &format!(
                "{} of {}",
                human_size(mem as i64),
                human_size(h.mem_total as i64)
            ),
        );
        mini_row(
            ui,
            GREEN,
            "Disk I/O",
            &format!(
                "↓ {} ↑ {}",
                human_rate(h.disk_read_bps),
                human_rate(h.disk_write_bps)
            ),
        );
        mini_row(
            ui,
            AMBER,
            "Network",
            &format!(
                "↓ {} ↑ {}",
                human_rate(h.net_rx_bps),
                human_rate(h.net_tx_bps)
            ),
        );
    }

    fn strata_overview(&self, ui: &mut Ui) {
        let g = &self.gui;
        let Some(m) = &g.strata_metrics else {
            if g.strata_cfg.api_key.trim().is_empty() {
                headline(ui, "Not set up", "", FAINT);
                meta_line(ui, "Add the endpoint and API key in Settings.");
            } else if let Some(e) = &g.strata_error {
                headline(ui, "Offline", "", RED);
                meta_line(ui, e);
            } else {
                headline(ui, "Connecting…", "", FAINT);
            }
            return;
        };
        let (state, color) = match m.state.as_str() {
            "generating" => ("Generating", GREEN),
            "reading" => ("Reading", ACCENT),
            _ => ("Idle", TEXT),
        };
        let queued = if m.queued > 0 {
            format!(" · {} queued", m.queued)
        } else {
            String::new()
        };
        headline(
            ui,
            state,
            &format!(
                "{}{queued}",
                if m.model.is_empty() {
                    "strata"
                } else {
                    &m.model
                }
            ),
            color,
        );
        let mut meta = vec![format!("{} requests served", fmt_tokens(m.tot_requests))];
        if let (Some(u), Some(t)) = (m.gpu_mem_used, m.gpu_mem_total) {
            meta.push(format!("VRAM {:.1} of {:.0} GB", u / 1e9, t / 1e9));
        }
        meta_line(ui, &meta.join(" · "));
        let last = m.requests.first();
        let dash = || "–".to_string();
        mini_row(
            ui,
            ACCENT,
            "Decode speed",
            &strata::decode_now(m, last)
                .map(|v| format!("{v:.1} tok/s"))
                .unwrap_or_else(dash),
        );
        mini_row(
            ui,
            VIOLET,
            "First token (p50)",
            &m.ttft.p50.map(|v| format!("{v:.2} s")).unwrap_or_else(dash),
        );
        let used = strata::context_used(m, last);
        mini_row(
            ui,
            CYAN,
            "Context",
            &if m.max_context > 0 {
                format!(
                    "{}% of {}",
                    (used * 100) / m.max_context,
                    fmt_tokens(m.max_context)
                )
            } else {
                dash()
            },
        );
    }

    // ------------------------------------------------------------------
    // Container detail: identity, live stats, inline logs
    // ------------------------------------------------------------------

    fn container_detail(&mut self, ui: &mut Ui) {
        let Some(c) = self.gui.selected_container().cloned() else {
            empty_state(
                ui,
                "No container selected",
                "Pick a container on the left to see live stats, details and logs.",
            );
            return;
        };
        let running = c.state == "running";
        let busy = self
            .gui
            .pending_action
            .as_ref()
            .is_some_and(|(i, _)| *i == c.id);

        let mut action = None;
        let mut remove = false;
        ui.horizontal(|ui| {
            let title_w = (ui.available_width() - 330.0).max(160.0);
            ui.vertical(|ui| {
                ui.set_max_width(title_w);
                ui.spacing_mut().item_spacing.y = 6.0;
                ui.add(
                    egui::Label::new(RichText::new(c.short_name()).font(bold(22.0)).color(TEXT))
                        .truncate(),
                );
                ui.horizontal(|ui| {
                    state_badge(ui, &c.state);
                    if let Some(p) = c.project() {
                        chip(ui, p, MUTED);
                    }
                    ui.add(
                        egui::Label::new(
                            RichText::new(&c.image)
                                .font(FontId::monospace(12.5))
                                .color(MUTED),
                        )
                        .truncate(),
                    );
                });
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if button(ui, "✕  Remove", Btn::Danger, !busy).clicked() {
                    remove = true;
                }
                if button(ui, "↻  Restart", Btn::Secondary, !busy).clicked() {
                    action = Some(Pending::Restart);
                }
                let clicked = if running {
                    button(ui, "■  Stop", Btn::Secondary, !busy).clicked()
                } else {
                    button(ui, "▶  Start", Btn::Primary, !busy).clicked()
                };
                if clicked {
                    action = Some(if running {
                        Pending::Stop
                    } else {
                        Pending::Start
                    });
                }
                if busy {
                    ui.add(egui::Spinner::new().size(18.0).color(ACCENT));
                }
            });
        });
        if remove {
            self.gui.confirm_remove = Some(c.id.clone());
        }
        if let Some(p) = action {
            self.gui.pending_action = Some((c.id.clone(), p));
            self.spin = Some((c.id.clone(), Instant::now()));
            spawn_action(&self.rt, &self.docker, &self.tx, &c, p);
        }
        ui.add_space(16.0);

        self.detail_stats(ui, &c);
        ui.add_space(14.0);

        card(ui, |ui| {
            card_title(ui, "Details", "");
            let ports = c.port_summary();
            let facts: [(&str, String, bool); 6] = [
                ("Status", c.status.clone(), false),
                ("Container ID", c.id[..12.min(c.id.len())].to_string(), true),
                (
                    "Created",
                    if c.created > 0 {
                        fmt_epoch(c.created)
                    } else {
                        "—".into()
                    },
                    false,
                ),
                ("Image", c.image.clone(), true),
                (
                    "Ports",
                    if ports.is_empty() {
                        "—".into()
                    } else {
                        ports
                    },
                    true,
                ),
                (
                    "Writable layer",
                    c.size_rw.map(human_size).unwrap_or_else(|| "—".into()),
                    false,
                ),
            ];
            let per_row = if ui.available_width() > 520.0 { 3 } else { 2 };
            ui.scope(|ui| {
                ui.spacing_mut().item_spacing.x = 20.0;
                for row in facts.chunks(per_row) {
                    ui.columns(per_row, |cols| {
                        for (col, (k, v, mono)) in cols.iter_mut().zip(row) {
                            kv(col, k, v, *mono);
                        }
                    });
                }
            });
        });
        ui.add_space(14.0);
        self.logs_card(ui, "Logs", false);
    }

    /// Live metric tiles for the selected container (CPU, memory, net, disk).
    fn detail_stats(&mut self, ui: &mut Ui, c: &ContainerSummary) {
        // Start a stats stream for the selected running container even when
        // the dashboard tab isn't active, so selection shows live numbers.
        let (missing, stats_gen) = {
            let g = &self.gui;
            (
                !g.stats.contains_key(&c.id)
                    && !self.streaming_stats.contains(&c.id)
                    && c.state == "running",
                g.stats_gen,
            )
        };
        if missing {
            self.streaming_stats.push(c.id.clone());
            self.stats_spawned.insert(c.id.clone(), Instant::now());
            spawn_stats(&self.rt, &self.docker, &self.tx, &c.id, stats_gen);
        }
        let g = &self.gui;
        let Some(s) = g.stats.get(&c.id) else {
            card(ui, |ui| {
                ui.horizontal(|ui| {
                    if c.state == "running" {
                        ui.add(egui::Spinner::new().size(14.0).color(ACCENT));
                        ui.label(RichText::new("Collecting live stats…").color(MUTED));
                    } else {
                        ui.label(
                            RichText::new("This container isn't running. Start it to see live CPU, memory and network.")
                                .color(MUTED),
                        );
                    }
                });
            });
            return;
        };
        let cpu = StatsSample::cpu_percent(g.prev_stats.get(&c.id), s)
            .unwrap_or(0.0)
            .max(0.0);
        let h = g.hist.get(&c.id);
        let series = |f: &dyn Fn(&StatHist) -> Vec<f64>| h.map(f).unwrap_or_default();
        let last = |q: Option<&VecDeque<i64>>| q.and_then(|q| q.back().copied()).unwrap_or(0);
        let rx = last(h.map(|h| &h.rx));
        let txb = last(h.map(|h| &h.tx));
        let br = last(h.map(|h| &h.blk_r));
        let bw = last(h.map(|h| &h.blk_w));
        let kpis = [
            Kpi {
                label: "CPU",
                device: String::new(),
                value: format!("{cpu:.1}%"),
                sub: "of one core = 100%".to_string(),
                color: ACCENT,
                viz: Viz::Spark(series(&|h| h.cpu.iter().copied().collect())),
                glyph: Some("cpu"),
            },
            Kpi {
                label: "Memory",
                device: String::new(),
                value: human_size(s.mem_usage()),
                sub: format!(
                    "of {} · {:.0}%",
                    human_size(s.mem_limit()),
                    s.mem_percent().unwrap_or(0.0)
                ),
                color: VIOLET,
                viz: Viz::Spark(series(&|h| h.mem.iter().map(|v| *v as f64).collect())),
                glyph: Some("ram"),
            },
            Kpi {
                label: "Network",
                device: String::new(),
                value: format!("↓ {}/s", human_size(rx)),
                sub: format!("↑ {}/s · {} pids", human_size(txb), s.pids_stats.current),
                color: CYAN,
                viz: Viz::Spark(series(&|h| {
                    h.rx.iter()
                        .zip(&h.tx)
                        .map(|(a, b)| (a + b) as f64)
                        .collect()
                })),
                glyph: Some("net"),
            },
            Kpi {
                label: "Disk I/O",
                device: String::new(),
                value: format!("{}/s", human_size(br + bw)),
                sub: format!(
                    "read {} · write {}",
                    human_size(s.blk_read()),
                    human_size(s.blk_write())
                ),
                color: AMBER,
                viz: Viz::Spark(series(&|h| {
                    h.blk_r
                        .iter()
                        .zip(&h.blk_w)
                        .map(|(a, b)| (a + b) as f64)
                        .collect()
                })),
                glyph: Some("disk"),
            },
        ];
        let per_row = if ui.available_width() > 560.0 { 4 } else { 2 };
        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.x = 12.0;
            for (n, row) in kpis.chunks(per_row).enumerate() {
                if n > 0 {
                    ui.add_space(6.0);
                }
                ui.columns(per_row, |cols| {
                    for (col, k) in cols.iter_mut().zip(row) {
                        kpi_card(col, k);
                    }
                });
            }
        });
    }

    // ------------------------------------------------------------------
    // Images / volumes detail
    // ------------------------------------------------------------------

    fn image_detail(&mut self, ui: &mut Ui) {
        let Some(id) = self.gui.selected_image.clone() else {
            empty_state(
                ui,
                "No image selected",
                "Pick an image on the left to see its size, tags and users.",
            );
            return;
        };
        let Some(im) = self.gui.images.iter().find(|i| i.id == id).cloned() else {
            empty_state(ui, "Image not found", "That image no longer exists.");
            return;
        };
        let users: Vec<String> = self
            .gui
            .containers
            .iter()
            .filter(|c| c.image == im.tag() || c.image.contains(im.short_id()))
            .map(|c| c.short_name())
            .collect();
        let total = self
            .gui
            .df
            .as_ref()
            .map(|d| d.layers_size)
            .filter(|v| *v > 0)
            .unwrap_or_else(|| self.gui.images.iter().map(|i| i.size).sum::<i64>().max(1));

        ui.horizontal(|ui| {
            ui.label(RichText::new(im.tag()).font(bold(22.0)).color(TEXT));
            chip(
                ui,
                if im.containers > 0 {
                    "in use"
                } else {
                    "unused"
                },
                if im.containers > 0 { GREEN } else { FAINT },
            );
        });
        ui.label(
            RichText::new(format!("sha256:{}", im.short_id()))
                .font(FontId::monospace(12.5))
                .color(MUTED),
        );
        ui.add_space(16.0);
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                let kpis = [
                    Kpi {
                        label: "Size",
                        device: String::new(),
                        value: human_size(im.size),
                        sub: format!(
                            "{:.1}% of all layers",
                            im.size as f64 / total.max(1) as f64 * 100.0
                        ),
                        color: ACCENT,
                        viz: Viz::Ratio(im.size as f32 / total.max(1) as f32),
                        glyph: None,
                    },
                    Kpi {
                        label: "Unique",
                        device: String::new(),
                        value: human_size((im.size - im.shared_size.max(0)).max(0)),
                        sub: format!(
                            "{} shared with other images",
                            human_size(im.shared_size.max(0))
                        ),
                        color: VIOLET,
                        viz: Viz::Ratio(
                            (im.size - im.shared_size.max(0)).max(0) as f32 / im.size.max(1) as f32,
                        ),
                        glyph: None,
                    },
                    Kpi {
                        label: "Containers",
                        device: String::new(),
                        value: im.containers.max(0).to_string(),
                        sub: if im.created > 0 {
                            format!("created {}", fmt_epoch(im.created))
                        } else {
                            String::new()
                        },
                        color: GREEN,
                        viz: Viz::None,
                        glyph: None,
                    },
                ];
                ui.scope(|ui| {
                    ui.spacing_mut().item_spacing.x = 14.0;
                    ui.columns(3, |cols| {
                        for (col, k) in cols.iter_mut().zip(&kpis) {
                            kpi_card(col, k);
                        }
                    });
                });
                ui.add_space(14.0);
                ui.scope(|ui| {
                    ui.spacing_mut().item_spacing.x = 14.0;
                    ui.columns(2, |cols| {
                        card(&mut cols[0], |ui| {
                            card_title(ui, "Tags", &im.repo_tags.len().to_string());
                            if im.repo_tags.is_empty() {
                                ui.label(
                                    RichText::new("<none>")
                                        .font(FontId::monospace(12.5))
                                        .color(FAINT),
                                );
                            }
                            for t in &im.repo_tags {
                                ui.add(
                                    egui::Label::new(
                                        RichText::new(t).font(FontId::monospace(12.5)).color(TEXT),
                                    )
                                    .truncate(),
                                );
                            }
                        });
                        card(&mut cols[1], |ui| {
                            card_title(ui, "Used by", &users.len().to_string());
                            if users.is_empty() {
                                ui.label(
                                    RichText::new("No containers use this image.").color(FAINT),
                                );
                            }
                            for u in users.iter().take(20) {
                                ui.horizontal(|ui| {
                                    let (r, _) =
                                        ui.allocate_exact_size(Vec2::splat(8.0), Sense::hover());
                                    ui.painter().circle_filled(r.center(), 3.0, GREEN);
                                    ui.add(
                                        egui::Label::new(RichText::new(u).size(13.0).color(TEXT))
                                            .truncate(),
                                    );
                                });
                            }
                        });
                    });
                });
            });
    }

    fn agent_detail(&mut self, ui: &mut Ui) {
        let Some(pid) = self.gui.selected_agent else {
            empty_state(
                ui,
                "No agent selected",
                "Pick an agent session on the left to see what it's doing.",
            );
            return;
        };
        let Some(a) = self.gui.agents.iter().find(|a| a.lead_pid == pid).cloned() else {
            empty_state(
                ui,
                "Session ended",
                "That agent session is no longer running.",
            );
            return;
        };
        let st = agent_state(&a);
        let now = epoch();
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.add(
                    egui::Label::new(RichText::new(agent_title(&a)).font(bold(22.0)).color(TEXT))
                        .truncate(),
                );
                let mut sub = vec![a.agent.to_string()];
                if let Some(c) = &a.cwd {
                    sub.push(c.clone());
                }
                if let Some(b) = a.activity.as_ref().and_then(|x| x.branch.clone()) {
                    sub.push(format!("⎇ {b}"));
                }
                ui.label(RichText::new(sub.join(" · ")).size(12.5).color(MUTED));
            });
        });
        ui.add_space(14.0);
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| match &a.activity {
                Some(act) => self.agent_activity(ui, &a, act, &st, now),
                None => {
                    card(ui, |ui| {
                        ui.label(
                            RichText::new(format!(
                                "Live activity is read from Claude Code's session files. For {}, SysKamling sees the processes only.",
                                a.agent
                            ))
                            .size(12.5)
                            .color(MUTED),
                        );
                    });
                    ui.add_space(12.0);
                    agent_process_card(ui, &a, now);
                }
            });
    }

    /// Status, task, numbers and recent steps of an agent that reports them.
    fn agent_activity(
        &self,
        ui: &mut Ui,
        a: &AgentSession,
        act: &Activity,
        st: &AgentState,
        now: i64,
    ) {
        let time = ui.input(|i| i.time);
        // ---- now: status, what it's on, the task ----
        card(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(st.label).font(bold(18.0)).color(st.color));
                if let Some(since) = act.status_since {
                    ui.label(
                        RichText::new(format!("for {}", uptime_str(now - since)))
                            .size(12.5)
                            .color(FAINT),
                    );
                }
            });
            ui.add_space(2.0);
            let current = act.steps.last().filter(|_| st.live);
            match (act.status.as_str(), current) {
                (_, Some(s)) => {
                    let (r, _) = ui
                        .allocate_exact_size(Vec2::new(ui.available_width(), 22.0), Sense::hover());
                    let p = ui.painter();
                    spinner_at(
                        p,
                        time,
                        Pos2::new(r.left() + 6.0, r.center().y),
                        5.0,
                        st.color,
                    );
                    let line = elide(
                        p,
                        step_line(s),
                        FontId::proportional(13.5),
                        TEXT,
                        r.width() - 22.0,
                    );
                    p.galley(
                        Pos2::new(r.left() + 20.0, r.center().y - line.size().y / 2.0),
                        line,
                        TEXT,
                    );
                    ui.ctx().request_repaint_after(Duration::from_millis(50));
                }
                ("waiting", _) => {
                    ui.label(RichText::new("Waiting for you: approve a tool call or answer a question in its terminal.").size(13.0).color(TEXT));
                }
                ("idle", _) => {
                    ui.label(
                        RichText::new("Done. Waiting for the next prompt.")
                            .size(13.0)
                            .color(MUTED),
                    );
                }
                _ => {}
            }
            if let Some(p) = &act.last_prompt {
                ui.add_space(8.0);
                ui.label(RichText::new("TASK").size(10.5).color(FAINT));
                ui.add(egui::Label::new(RichText::new(p).size(13.0).color(TEXT)).wrap());
            }
        });
        ui.add_space(12.0);
        // ---- numbers: one compact strip ----
        let stats = [
            (
                "Last activity",
                act.last_active
                    .map(|t| ago(now - t))
                    .unwrap_or_else(|| "–".into()),
                format!("{} steps in view", act.steps.len()),
            ),
            (
                "Context",
                act.context_tokens
                    .map(|t| format!("{} tokens", fmt_tokens(t as i64)))
                    .unwrap_or_else(|| "–".into()),
                act.model.clone().unwrap_or_default(),
            ),
            (
                "CPU",
                format!("{:.1}%", a.cpu_pct),
                format!(
                    "{} process{}",
                    a.pids.len(),
                    if a.pids.len() == 1 { "" } else { "es" }
                ),
            ),
            (
                "Memory",
                human_size(a.mem_bytes as i64),
                "resident".to_string(),
            ),
        ];
        card(ui, |ui| {
            ui.columns(4, |cols| {
                for (col, (label, value, sub)) in cols.iter_mut().zip(&stats) {
                    col.spacing_mut().item_spacing.y = 1.0;
                    col.label(RichText::new(*label).size(11.5).color(FAINT));
                    col.add(
                        egui::Label::new(RichText::new(value).font(bold(17.0)).color(TEXT))
                            .truncate(),
                    );
                    col.add(
                        egui::Label::new(RichText::new(sub).size(11.5).color(FAINT)).truncate(),
                    );
                }
            });
        });
        ui.add_space(12.0);
        // ---- recent steps, newest first ----
        card(ui, |ui| {
            card_title(ui, "Recent activity", "newest first");
            if act.steps.is_empty() {
                ui.label(RichText::new("No steps recorded yet.").color(FAINT));
                return;
            }
            ui.spacing_mut().item_spacing.y = 0.0;
            for s in act.steps.iter().rev() {
                let (r, _) =
                    ui.allocate_exact_size(Vec2::new(ui.available_width(), 28.0), Sense::hover());
                if !ui.is_rect_visible(r) {
                    continue;
                }
                let p = ui.painter();
                let (tag, color) = match s.kind {
                    StepKind::Prompt => ("You".to_string(), ACCENT),
                    StepKind::Reply => ("Reply".to_string(), MUTED),
                    StepKind::Tool if s.failed => (s.tool.clone(), RED),
                    StepKind::Tool => (s.tool.clone(), CYAN),
                };
                let y = r.center().y;
                if s.pending && st.live {
                    spinner_at(p, time, Pos2::new(r.left() + 5.0, y), 4.0, color);
                } else {
                    p.circle_filled(Pos2::new(r.left() + 5.0, y), 3.0, color);
                }
                let when = s.at.map(|t| ago(now - t)).unwrap_or_default();
                p.text(
                    Pos2::new(r.right() - 4.0, y),
                    egui::Align2::RIGHT_CENTER,
                    when,
                    FontId::proportional(11.5),
                    FAINT,
                );
                let tag_g = elide(p, tag, FontId::proportional(12.0), color, 150.0);
                let tag_w = tag_g.size().x;
                p.galley(
                    Pos2::new(r.left() + 18.0, y - tag_g.size().y / 2.0),
                    tag_g,
                    color,
                );
                let text_color = if s.kind == StepKind::Reply {
                    MUTED
                } else {
                    TEXT
                };
                let text = elide(
                    p,
                    s.text.clone(),
                    FontId::proportional(13.0),
                    text_color,
                    r.width() - tag_w - 120.0,
                );
                p.galley(
                    Pos2::new(r.left() + 28.0 + tag_w, y - text.size().y / 2.0),
                    text,
                    text_color,
                );
            }
        });
        ui.add_space(12.0);
        agent_process_card(ui, a, now);
    }

    // ------------------------------------------------------------------
    // Hardware tab: host vitals, Task-Manager style
    // ------------------------------------------------------------------

    fn hardware_view(&mut self, ui: &mut Ui, inv: &hwinv::Inventory, usage: &[hwinv::DiskUsage]) {
        let Some(s) = self.gui.hw.clone() else {
            empty_state(
                ui,
                "Collecting…",
                "The first hardware sample lands in a moment.",
            );
            return;
        };
        page_header(
            ui,
            "Hardware",
            &format!(
                "{} · {} cores · {} RAM · up {}",
                if inv.board.is_empty() {
                    inv.cpu.as_str()
                } else {
                    inv.board.as_str()
                },
                s.ncpu,
                human_size(s.mem_total as i64),
                ago(s.uptime as i64)
            ),
        );
        // The page is taller than most windows: KPI cards, charts, the
        // machine inventory and per-disk detail all stack, so the body
        // scrolls instead of clipping.
        // regression: the body must actually scroll, not clip. The binding
        // and the probe call are test-only; the release build renders the
        // very same scroll area without it.
        let hw_body = ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add_space(16.0);

                let mem_used = s.mem_total - s.mem_avail;
                let swap_used = s.swap_total.saturating_sub(s.swap_free);
                let cpu_hist: Vec<f64> = self.gui.hw_cpu_hist.iter().copied().collect();
                let gpu_busy_hist: Vec<f64> = self.gui.hw_gpu_busy_hist.iter().copied().collect();
                let dr_hist: Vec<f64> = self.gui.hw_disk_r_hist.iter().copied().collect();
                let dw_hist: Vec<f64> = self.gui.hw_disk_w_hist.iter().copied().collect();
                let rx_hist: Vec<f64> = self.gui.hw_net_rx_hist.iter().copied().collect();
                let tx_hist: Vec<f64> = self.gui.hw_net_tx_hist.iter().copied().collect();
                let gpus = self.gui.gpus.clone();

                // per-card device identity from the inventory: the cards themselves
                // say WHICH cpu/ram/ssd/nic this is — no separate machine section
                let cpu_name = inv.cpu.clone();
                let mem_name = inv
                    .devices
                    .iter()
                    .find(|d| d.kind == "memory")
                    .map(|d| format!("{} RAM", d.detail))
                    .unwrap_or_default();
                let disk_name = inv
                    .disk_kerns()
                    .first()
                    .and_then(|k| inv.disk_name(k))
                    .unwrap_or_default()
                    .to_string();
                let nic_name = inv
                    .devices
                    .iter()
                    .find(|d| d.kind == "nic")
                    .map(|d| d.name.clone())
                    .unwrap_or_default();

                let mut kpis = vec![
                    Kpi {
                        label: "CPU",
                        device: cpu_name,
                        value: format!("{:.1}%", s.cpu_pct),
                        sub: match s.cpu_temp_c {
                            Some(t) => format!("{} °C · load {}", temp_str(t), s.load1),
                            None => format!("load {} · {} · {}", s.load1, s.load5, s.load15),
                        },
                        color: ACCENT,
                        viz: Viz::Spark(cpu_hist),
                        glyph: Some("cpu"),
                    },
                    Kpi {
                        label: "Memory",
                        device: mem_name,
                        value: human_size(mem_used as i64),
                        sub: format!(
                            "of {} ({:.0}%)",
                            human_size(s.mem_total as i64),
                            mem_used as f64 / s.mem_total.max(1) as f64 * 100.0
                        ),
                        color: VIOLET,
                        viz: Viz::Ratio(mem_used as f32 / s.mem_total.max(1) as f32),
                        glyph: Some("ram"),
                    },
                    Kpi {
                        label: "Disk I/O",
                        device: disk_name,
                        value: format!(
                            "{} · {}",
                            human_rate(s.disk_read_bps),
                            human_rate(s.disk_write_bps)
                        ),
                        sub: "read · write per second".into(),
                        color: GREEN,
                        viz: Viz::Spark(
                            dr_hist
                                .into_iter()
                                .zip(dw_hist)
                                .map(|(r, w)| r + w)
                                .collect(),
                        ),
                        glyph: Some("disk"),
                    },
                    Kpi {
                        label: "Network",
                        device: nic_name,
                        value: format!(
                            "{} · {}",
                            human_rate(s.net_rx_bps),
                            human_rate(s.net_tx_bps)
                        ),
                        sub: "down · up per second".into(),
                        color: AMBER,
                        viz: Viz::Spark(
                            rx_hist
                                .into_iter()
                                .zip(tx_hist)
                                .map(|(r, t)| r + t)
                                .collect(),
                        ),
                        glyph: Some("net"),
                    },
                ];
                if let Some(g) = gpus.iter().max_by(|a, b| {
                    a.busy_pct
                        .unwrap_or(0.0)
                        .partial_cmp(&b.busy_pct.unwrap_or(0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                }) {
                    kpis.push(Kpi {
                        label: "GPU",
                        device: g.name.clone(),
                        value: g
                            .busy_pct
                            .map(|b| format!("{b:.0}%"))
                            .unwrap_or_else(|| "—".into()),
                        sub: match (g.temp_c, g.power_mw) {
                            (Some(t), Some(p)) => format!(
                                "{} · {} °C · {:.0} W",
                                g.name,
                                temp_str(t),
                                p as f64 / 1000.0
                            ),
                            (Some(t), None) => format!("{} · {} °C", g.name, temp_str(t)),
                            _ => g.name.clone(),
                        },
                        color: CYAN,
                        viz: Viz::Spark(gpu_busy_hist.clone()),
                        glyph: Some("gpu"),
                    });
                }
                if let Some((dname, t)) = s
                    .disks
                    .iter()
                    .filter_map(|d| d.temp_c.map(|t| (d.name.clone(), t)))
                    .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                {
                    kpis.push(Kpi {
                        label: "SSD",
                        device: inv.disk_name(&dname).map(String::from).unwrap_or(dname),
                        value: format!("{} °C", temp_str(t)),
                        sub: "drive temperature".into(),
                        color: if t > 70.0 {
                            RED
                        } else if t > 60.0 {
                            AMBER
                        } else {
                            GREEN
                        },
                        viz: Viz::Spark(self.gui.hw_disk_temp_hist.iter().copied().collect()),
                        glyph: Some("disk"),
                    });
                }
                ui.scope(|ui| {
                    ui.spacing_mut().item_spacing.x = 14.0;
                    let per_row = if kpis.len() >= 6 { 3 } else { 4 };
                    for (n, row) in kpis.chunks(per_row).enumerate() {
                        if n > 0 {
                            ui.add_space(6.0);
                        }
                        ui.columns(per_row, |cols| {
                            for (col, k) in cols.iter_mut().zip(row) {
                                kpi_card(col, k);
                            }
                        });
                    }
                });
                ui.add_space(14.0);

                let swap_hist: Vec<f64> = self.gui.hw_swap_hist.iter().copied().collect();
                ui.scope(|ui| {
                    ui.spacing_mut().item_spacing.x = 14.0;
                    ui.columns(2, |cols| {
                        card(&mut cols[0], |ui| {
                            // memory already has its KPI card above; this one is
                            // dedicated to swap pressure
                            card_title(
                                ui,
                                "Swap",
                                &format!(
                                    "{} of {}",
                                    human_size(swap_used as i64),
                                    human_size(s.swap_total as i64)
                                ),
                            );
                            bar_row(ui, "Used", swap_used, s.swap_total, AMBER);
                            let w = ui.available_width();
                            area_chart(ui, &swap_hist, AMBER, Vec2::new(w, 54.0), |v| {
                                human_size(v as i64)
                            });
                        });
                        card(&mut cols[1], |ui| {
                            card_title(ui, "Sensors", "");
                            if s.sensors.is_empty() {
                                ui.label(
                                    RichText::new("no hwmon temperature sensors found")
                                        .color(FAINT),
                                );
                            } else {
                                for x in &s.sensors {
                                    let c = if x.c > 85.0 {
                                        RED
                                    } else if x.c > 70.0 {
                                        AMBER
                                    } else {
                                        GREEN
                                    };
                                    ui.horizontal(|ui| {
                                        ui.label(RichText::new(x.group).size(11.5).color(FAINT));
                                        ui.label(RichText::new(&x.label).size(12.5).color(MUTED));
                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                ui.label(
                                                    RichText::new(format!("{} °C", temp_str(x.c)))
                                                        .font(FontId::monospace(12.5))
                                                        .color(c),
                                                );
                                            },
                                        );
                                    });
                                }
                            }
                            ui.add_space(4.0);
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("Disk").size(12.5).color(MUTED).weak());
                                ui.label(
                                    RichText::new(format!("↓ {}", human_rate(s.disk_read_bps)))
                                        .font(FontId::monospace(12.5))
                                        .color(TEXT),
                                );
                                ui.label(
                                    RichText::new(format!("↑ {}", human_rate(s.disk_write_bps)))
                                        .font(FontId::monospace(12.5))
                                        .color(TEXT),
                                );
                            });
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("Net  ").size(12.5).color(MUTED).weak());
                                ui.label(
                                    RichText::new(format!("↓ {}", human_rate(s.net_rx_bps)))
                                        .font(FontId::monospace(12.5))
                                        .color(TEXT),
                                );
                                ui.label(
                                    RichText::new(format!("↑ {}", human_rate(s.net_tx_bps)))
                                        .font(FontId::monospace(12.5))
                                        .color(TEXT),
                                );
                            });
                        });
                    });
                });

                // ---- GPUs: one card per display adapter ----
                if !gpus.is_empty() {
                    ui.add_space(14.0);
                    section_heading(ui, "Graphics");
                    ui.add_space(6.0);
                    for g in &gpus {
                        let busy = g.busy_pct.unwrap_or(0.0);
                        let color = if busy > 85.0 {
                            RED
                        } else if busy > 60.0 {
                            AMBER
                        } else {
                            CYAN
                        };
                        card(ui, |ui| {
                            card_title(
                                ui,
                                &g.name,
                                &match (g.temp_c, g.power_mw) {
                                    (Some(t), Some(p)) => {
                                        format!("{} °C · {:.0} W", temp_str(t), p as f64 / 1000.0)
                                    }
                                    (Some(t), None) => format!("{} °C", temp_str(t)),
                                    (None, Some(p)) => format!("{:.0} W", p as f64 / 1000.0),
                                    (None, None) => String::new(),
                                },
                            );
                            if g.busy_pct.is_some() {
                                bar_row(ui, "Busy", busy as u64, 100, color);
                            } else {
                                ui.label(
                                    RichText::new("utilization not exposed by this driver")
                                        .size(12.0)
                                        .color(FAINT),
                                );
                            }
                            if g.vram_total > 0 {
                                bar_row(ui, "VRAM", g.vram_used, g.vram_total, VIOLET);
                            }
                            let w = ui.available_width();
                            area_chart(ui, &gpu_busy_hist, color, Vec2::new(w, 60.0), |v| {
                                format!("{v:.0}%")
                            });
                        });
                        ui.add_space(8.0);
                    }
                }

                // ---- Disks: per-device stats, Task-Manager style ----
                if !s.disks.is_empty() {
                    ui.add_space(6.0);
                    section_heading(ui, "Storage");
                    ui.add_space(6.0);
                    // only drives we actually inventoried: internal SATA/OEM disks
                    // (sda…) and anything not named by the inventory are skipped
                    let inv_disks: Vec<&hwinv::HwDevice> =
                        inv.devices.iter().filter(|d| d.kind == "disk").collect();
                    let named: Vec<(&crate::hw::DiskStat, Option<&hwinv::HwDevice>)> = s
                        .disks
                        .iter()
                        .filter(|d| {
                            inv_disks
                                .iter()
                                .any(|x| x.kern.as_deref() == Some(d.name.as_str()))
                        })
                        .map(|d| {
                            let inv = inv_disks
                                .iter()
                                .find(|x| x.kern.as_deref() == Some(d.name.as_str()))
                                .copied();
                            (d, inv)
                        })
                        .collect();
                    if named.is_empty() && !s.disks.is_empty() {
                        // no inventory match at all: fall back to everything except
                        // obvious internal/OEM names so the section isn't blank
                        let fallback: Vec<&crate::hw::DiskStat> = s
                            .disks
                            .iter()
                            .filter(|d| !(d.name == "sda" || d.name == "sdb"))
                            .collect();
                        for d in fallback {
                            disk_card(ui, d, None, usage);
                            ui.add_space(8.0);
                        }
                    }
                    for (d, inv) in &named {
                        disk_card(ui, d, *inv, usage);
                        ui.add_space(8.0);
                    }
                }
            });
        // regression: the body must actually scroll, not clip
        #[cfg(test)]
        hw_scroll_probe::record(hw_body.inner_rect.height(), hw_body.content_size.y);
        #[cfg(not(test))]
        let _ = hw_body;
    }
    // ------------------------------------------------------------------
    // Ports tab: the socket detail page
    // ------------------------------------------------------------------

    fn port_detail(&mut self, ui: &mut Ui) {
        let Some(key) = self.gui.selected_port.clone() else {
            empty_state(
                ui,
                "No socket selected",
                "Pick a listening socket on the left to see its owner and links.",
            );
            return;
        };
        let Some(p) = self
            .gui
            .ports
            .iter()
            .find(|e| (e.proto, e.addr.as_str(), e.port) == (key.0.as_str(), key.1.as_str(), key.2))
            .cloned()
        else {
            empty_state(ui, "Socket closed", "That socket is no longer bound.");
            return;
        };
        let color = port_color(&p);
        ui.horizontal(|ui| {
            ui.label(
                RichText::new(format!("{}:{}", p.addr, p.port))
                    .font(FontId::monospace(22.0))
                    .color(TEXT),
            );
            chip(
                ui,
                &format!("{}{}", p.proto, if p.family == 6 { "6" } else { "" }),
                color,
            );
            if p.state != "-" {
                chip(ui, p.state, port_state_color(p.state));
            }
        });
        ui.label(
            RichText::new(format!("inode {}", p.inode))
                .font(FontId::monospace(12.0))
                .color(FAINT),
        );
        ui.add_space(16.0);

        let kpis = [
            Kpi {
                label: "Process",
                device: String::new(),
                value: p.proc_name.clone().unwrap_or_else(|| "—".into()),
                sub: p
                    .pid
                    .map(|x| format!("pid {x}"))
                    .unwrap_or_else(|| "owner not visible".into()),
                color: if p.pid.is_some() { GREEN } else { FAINT },
                viz: Viz::None,
                glyph: None,
            },
            Kpi {
                label: "Protocol",
                device: String::new(),
                value: p.proto.to_uppercase(),
                sub: format!("IPv{}", p.family),
                color: ACCENT,
                viz: Viz::None,
                glyph: None,
            },
            Kpi {
                label: "Bind",
                device: String::new(),
                value: if p.addr == "0.0.0.0" || p.addr == "[::]" {
                    "public"
                } else if p.addr.starts_with("127.") || p.addr == "[::1]" {
                    "localhost"
                } else {
                    "interface"
                }
                .into(),
                sub: p.addr.clone(),
                color: if p.addr == "0.0.0.0" || p.addr == "[::]" {
                    AMBER
                } else {
                    GREEN
                },
                viz: Viz::None,
                glyph: None,
            },
        ];
        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.x = 14.0;
            ui.columns(3, |cols| {
                for (col, k) in cols.iter_mut().zip(&kpis) {
                    kpi_card(col, k);
                }
            });
        });
        ui.add_space(14.0);

        card(ui, |ui| {
            card_title(ui, "Working directory", "");
            match &p.cwd {
                Some(c) => ui.label(RichText::new(c).font(FontId::monospace(12.5)).color(TEXT)),
                None => ui.label(
                    RichText::new(if p.pid.is_some() {
                        "not readable (different user?)"
                    } else {
                        "no owning process found"
                    })
                    .color(FAINT),
                ),
            };
        });
        ui.add_space(10.0);

        // Cross-link: which container publishes this host port?
        let linked = self.gui.containers_on_port(p.port);
        card(ui, |ui| {
            card_title(ui, "Linked containers", &linked.len().to_string());
            if linked.is_empty() {
                ui.label(
                    RichText::new(if p.pid.is_some() {
                        "No container publishes this host port."
                    } else {
                        "If a container publishes this port, its process lives in another namespace and shows up here as an anonymous socket forwarded by dockerd."
                    })
                    .size(12.5)
                    .color(FAINT),
                );
            }
            for name in linked {
                ui.horizontal(|ui| {
                    let (r, _) = ui.allocate_exact_size(Vec2::splat(8.0), Sense::hover());
                    ui.painter().circle_filled(r.center(), 3.0, GREEN);
                    ui.label(RichText::new(name).size(13.0).color(TEXT));
                });
            }
        });
    }

    // ------------------------------------------------------------------
    // Settings: every persisted knob lives here
    // ------------------------------------------------------------------

    fn settings_view(&mut self, ui: &mut Ui) {
        page_header(
            ui,
            "Settings",
            "how SysKamling behaves and which servers it watches",
        );
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.add_space(16.0);
                let mut general_dirty = false;

                // ---- General ------------------------------------------------
                card(ui, |ui| {
                    card_title(ui, "General", "");
                    ui.label(
                        RichText::new(
                            "changes apply the moment you touch them · saved automatically",
                        )
                        .size(12.0)
                        .color(FAINT),
                    );
                    ui.add_space(4.0);

                    // refresh interval: stepper + quick presets
                    let presets = [1.0_f64, 2.0, 5.0, 10.0];
                    setting_row(
                        ui,
                        "Refresh interval",
                        "seconds between live pulls · 0 disables polling",
                        |ui| {
                            let mut hit = None;
                            for p in presets {
                                let on = (self.gui.refresh_edit - p).abs() < 0.01;
                                let b = ui.add_sized(
                                    [40.0, 26.0],
                                    egui::Button::new(
                                        RichText::new(format!("{p:.0}s")).size(12.0).color(if on {
                                            ACCENT
                                        } else {
                                            MUTED
                                        }),
                                    )
                                    .fill(if on {
                                        ACCENT.gamma_multiply(0.18)
                                    } else {
                                        SURFACE_HI
                                    })
                                    .stroke(Stroke::new(
                                        1.0_f32,
                                        if on {
                                            ACCENT.gamma_multiply(0.6)
                                        } else {
                                            BORDER
                                        },
                                    )),
                                );
                                if b.clicked() {
                                    hit = Some(p);
                                }
                            }
                            let dv = egui::DragValue::new(&mut self.gui.refresh_edit)
                                .range(0.0..=60.0)
                                .speed(0.25)
                                .suffix(" s")
                                .max_decimals(1);
                            let r = ui.add_sized([92.0, 26.0], dv);
                            if r.lost_focus() || ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                general_dirty = true;
                            }
                            if let Some(p) = hit {
                                self.gui.refresh_edit = p;
                                general_dirty = true;
                            }
                        },
                    );
                    // apply live so the stepper feels responsive...
                    self.refresh = self.gui.refresh_edit.max(0.0);
                    if general_dirty {
                        // ...and persist once the edit settles (Enter / focus loss)
                        if let Err(e) = strata::patch(|c| {
                            c.general = strata::GeneralConfig {
                                refresh_secs: self.refresh,
                                autoscroll_logs: self.gui.autoscroll_logs,
                            };
                        }) {
                            self.gui.status = format!("could not save settings: {e}");
                        } else {
                            let v = self.refresh;
                            let vs = if v.fract() == 0.0 {
                                format!("{}", v as i64)
                            } else {
                                format!("{v}")
                            };
                            self.gui.status = format!("refresh every {vs} s");
                        }
                        general_dirty = false;
                    }

                    setting_row(
                        ui,
                        "Follow log output",
                        "stick to the bottom as new lines land",
                        |ui| {
                            let on = self.gui.autoscroll_logs;
                            if switch(ui, on).clicked() {
                                self.gui.autoscroll_logs = !on;
                                self.gui.logs_follow = !on;
                                general_dirty = true;
                            }
                        },
                    );
                    if general_dirty {
                        if let Err(e) = strata::patch(|c| {
                            c.general = strata::GeneralConfig {
                                refresh_secs: self.refresh,
                                autoscroll_logs: self.gui.autoscroll_logs,
                            };
                        }) {
                            self.gui.status = format!("could not save settings: {e}");
                        } else {
                            self.gui.status = if self.gui.autoscroll_logs {
                                "log follow on".into()
                            } else {
                                "log follow off".into()
                            };
                        }
                        general_dirty = false;
                    }
                });
                ui.add_space(12.0);

                // ---- Strata server ------------------------------------------
                // edits are staged in the fields; nothing touches the config
                // file until Test & save, so half-typed values can't break the
                // live connection
                let mut test_now = false;
                card(ui, |ui| {
                    let connected = self.gui.strata_health.is_some();
                    let dot = if connected {
                        GREEN
                    } else if self.gui.strata_error.is_some() {
                        RED
                    } else {
                        FAINT
                    };
                    card_title(ui, "Strata server", "");
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("●").size(13.0).color(dot));
                        let sub = if let Some(h) = &self.gui.strata_health {
                            let svc = if h.service.is_empty() {
                                "strata"
                            } else {
                                h.service.as_str()
                            };
                            let model = if h.model.is_empty() {
                                "no model"
                            } else {
                                &h.model
                            };
                            format!(
                                "{svc} · {} · {} ctx · {model} · {}",
                                h.status,
                                h.max_context,
                                if h.loaded { "loaded" } else { "not loaded" }
                            )
                        } else if let Some(e) = &self.gui.strata_error {
                            e.clone()
                        } else {
                            "not connected yet — set the endpoint and key, then Test & save".into()
                        };
                        ui.label(RichText::new(sub).size(12.0).color(if connected {
                            GREEN
                        } else {
                            MUTED
                        }));
                    });
                    ui.add_space(6.0);

                    let ep_resp = field(
                        ui,
                        &mut self.gui.strata_endpoint_edit,
                        "http://127.0.0.1:9191",
                        true,
                        false,
                        None,
                    );
                    #[cfg(test)]
                    {
                        self.gui.probe_endpoint_rect = Some(ep_resp.rect);
                    }
                    #[cfg(not(test))]
                    let _ = ep_resp;
                    ui.add_space(8.0);

                    let masked = self.gui.strata_key_masked;
                    let key_resp = field(
                        ui,
                        &mut self.gui.strata_key_edit,
                        "sk-…",
                        true,
                        masked,
                        Some((
                            if masked { "eye" } else { "eye_off" },
                            masked,
                            &mut |on: bool| {
                                self.gui.strata_key_masked = !on;
                            },
                        )),
                    );
                    // pull the save row up flush under the fields instead of
                    // leaving a full spacing gap
                    let c = ui.cursor();
                    ui.scope_builder(
                        UiBuilder::new().max_rect(Rect::from_min_max(
                            Pos2::new(c.min.x, key_resp.rect.bottom() + 10.0),
                            ui.max_rect().max,
                        )),
                        |ui| {
                            ui.horizontal(|ui| {
                                let btn = ui.add_sized(
                                    [118.0, 30.0],
                                    egui::Button::new(
                                        RichText::new("⟳  Test & save")
                                            .font(bold(12.5))
                                            .color(ACCENT),
                                    )
                                    .fill(ACCENT.gamma_multiply(0.14))
                                    .stroke(Stroke::new(1.0_f32, ACCENT.gamma_multiply(0.55))),
                                );
                                #[cfg(test)]
                                {
                                    self.gui.probe_save_rect = Some(btn.rect);
                                }
                                if btn.clicked() {
                                    test_now = true;
                                    // commit the trimmed values right away so
                                    // the status line can't echo a half-typed
                                    // edit before the save block runs
                                    self.gui.strata_cfg.endpoint =
                                        self.gui.strata_endpoint_edit.trim().to_string();
                                    self.gui.strata_cfg.api_key =
                                        self.gui.strata_key_edit.trim().to_string();
                                }
                                if let Some(h) = &self.gui.strata_health {
                                    let svc = if h.service.is_empty() {
                                        "strata"
                                    } else {
                                        h.service.as_str()
                                    };
                                    ui.label(
                                        RichText::new(format!(
                                            "● {svc} · {} · {} ctx · images {} · key {}",
                                            h.status,
                                            h.max_context,
                                            if h.images { "on" } else { "off" },
                                            if h.api_key { "set" } else { "none" }
                                        ))
                                        .size(12.0)
                                        .color(GREEN),
                                    );
                                } else if let Some(e) = &self.gui.strata_error {
                                    ui.label(RichText::new(format!("● {e}")).size(12.0).color(RED));
                                }
                            });
                        },
                    );

                    // save row: button flush against the fields, live status
                    // to its right — no dead gap between them
                });

                if test_now {
                    self.gui.strata_cfg.endpoint = self.gui.strata_endpoint_edit.trim().to_string();
                    self.gui.strata_cfg.api_key = self.gui.strata_key_edit.trim().to_string();
                    match strata::patch(|c| c.strata = self.gui.strata_cfg.clone()) {
                        Ok(_) => {
                            self.strata_client =
                                Arc::new(strata::HttpClient::new(&self.gui.strata_cfg));
                            self.last_strata_poll = Instant::now() - Duration::from_secs(999);
                            self.spawn_strata_poll_inner(true);
                            self.gui.status = "strata settings saved".into();
                        }
                        Err(e) => self.gui.status = format!("could not save settings: {e}"),
                    }
                }
                ui.add_space(12.0);

                // ---- config file ---------------------------------------------
                card(ui, |ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Config file").font(bold(14.5)).color(TEXT));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let path = strata::config_path()
                                .map(|p| p.display().to_string())
                                .unwrap_or_else(|| "unavailable".into());
                            ui.label(
                                RichText::new(path)
                                    .font(FontId::monospace(12.0))
                                    .color(MUTED),
                            );
                        });
                    });
                });
                ui.add_space(8.0);
            });
    }

    // ------------------------------------------------------------------
    // Strata: a remote Strata inference server's monitor, statistics only
    // ------------------------------------------------------------------

    fn strata_view(&mut self, ui: &mut Ui) {
        // The poller only runs while the tab is visible; pulling the instant
        // it opens means the first metrics land without waiting a whole
        // refresh interval for the next tick.
        if self.gui.strata_metrics.is_none() {
            self.last_strata_poll = Instant::now() - Duration::from_secs(999);
            // poll_now bypasses the visible-tab gate, so the first pull
            // lands even when the tick order puts the fetch before the view
            self.spawn_strata_poll();
        }
        // ---- connection status (settings live on the Settings tab) ----
        card(ui, |ui| {
            ui.horizontal(|ui| {
                if let Some(h) = &self.gui.strata_health {
                    let svc = if h.service.is_empty() {
                        "strata"
                    } else {
                        h.service.as_str()
                    };
                    let model = if h.model.is_empty() {
                        "no model"
                    } else {
                        &h.model
                    };
                    ui.label(
                        RichText::new(format!(
                            "● {svc} · {} · {} ctx · {} · {} · images {} · key {}",
                            h.status,
                            h.max_context,
                            model,
                            if h.loaded { "loaded" } else { "not loaded" },
                            if h.images { "on" } else { "off" },
                            if h.api_key { "set" } else { "none" }
                        ))
                        .size(12.0)
                        .color(GREEN),
                    );
                } else if let Some(e) = &self.gui.strata_error {
                    ui.label(RichText::new(format!("● {e}")).size(12.0).color(RED));
                } else {
                    ui.label(RichText::new("● connecting…").size(12.0).color(FAINT));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui
                        .button(RichText::new("⚙ Settings").font(bold(12.5)).color(ACCENT))
                        .clicked()
                    {
                        self.on_tab_change(Tab::Settings);
                    }
                });
            });
        });
        ui.add_space(12.0);

        // ---- the statistics themselves ----
        let Some(m) = self.gui.strata_metrics.clone() else {
            if let Some(e) = &self.gui.strata_error {
                empty_state(ui, "Server not reachable", e);
            } else if self.gui.strata_cfg.api_key.trim().is_empty() {
                empty_state(
                    ui,
                    "No API key set",
                    "This server keeps /metrics behind a key — open ⚙ Settings, fill in endpoint and key, and press ⟳ Test & save.",
                );
            } else {
                empty_state(
                    ui,
                    "Collecting…",
                    "The first /metrics pull lands in a moment.",
                );
            }
            return;
        };
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| self.strata_stats(ui, &m));
    }

    fn strata_stats(&mut self, ui: &mut Ui, m: &strata::Metrics) {
        let last = m.requests.first();
        // header: model state, like the web monitor's pill + progress bar
        let (state, state_color, detail, frac) = match m.state.as_str() {
            "reading" => {
                let pct = match (m.prompt_read, m.prompt_total) {
                    (Some(r), Some(t)) if t > 0 => format!(" · {}%", (100 * r) / t),
                    _ => String::new(),
                };
                (
                    format!("Reading prompt{pct}"),
                    ACCENT,
                    fmt_tokens(m.prompt_read.unwrap_or(0))
                        + &m.prompt_total
                            .map(|t| format!(" / {} tokens read", fmt_tokens(t)))
                            .unwrap_or_default(),
                    m.prompt_total
                        .filter(|t| *t > 0)
                        .and_then(|t| m.prompt_read.map(|r| r as f32 / t as f32))
                        .unwrap_or(0.0),
                )
            }
            "generating" => {
                let phase = m.phase.clone().unwrap_or_else(|| "Generating".into());
                (
                    capitalize(&phase),
                    GREEN,
                    format!(
                        "{} tokens · {} tok/s",
                        fmt_tokens(m.generated.unwrap_or(0)),
                        m.tok_s
                            .map(|v| format!("{v:.1}"))
                            .unwrap_or_else(|| "–".into())
                    ),
                    m.max_tokens
                        .filter(|t| *t > 0)
                        .and_then(|t| m.generated.map(|g| (g as f32 / t as f32).min(1.0)))
                        .unwrap_or(0.0),
                )
            }
            _ => (
                "Idle".into(),
                FAINT,
                last.map(|r| {
                    format!(
                        "last: {} tokens{}",
                        fmt_tokens(r.output_tokens.unwrap_or(0)),
                        r.decode_tok_s
                            .map(|d| format!(" at {d:.1} tok/s"))
                            .unwrap_or_default()
                    )
                })
                .unwrap_or_else(|| "Waiting for a request".into()),
                0.0,
            ),
        };
        let queued = if m.queued > 0 {
            format!(" · {} queued", m.queued)
        } else {
            String::new()
        };
        page_header(ui, "Strata", "");
        ui.horizontal(|ui| {
            ui.label(RichText::new("MODEL STATE").size(11.0).color(FAINT));
            chip(ui, &format!("{state}{queued}"), state_color);
            ui.label(RichText::new(detail).size(12.5).color(MUTED));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(
                    RichText::new(format!(
                        "{} · {} tok/s",
                        if m.model.is_empty() {
                            "strata"
                        } else {
                            &m.model
                        },
                        m.hist
                            .get("tok_s")
                            .and_then(|h| h.last())
                            .map(|v| format!("{v:.1}"))
                            .unwrap_or_else(|| "–".into())
                    ))
                    .size(12.5)
                    .color(MUTED),
                );
            });
        });
        let (r, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 5.0), Sense::hover());
        progress(ui.painter(), r, frac, state_color);
        ui.add_space(14.0);

        // ---- the eight tiles ----
        let hist = |k: &str| -> Vec<f64> { m.hist.get(k).cloned().unwrap_or_default() };
        let speed = strata::decode_now(m, last);
        let prefill = strata::prefill_now(m, last);
        let multi_gpu = m
            .gpu_name
            .as_deref()
            .map(|n| n.contains(" + "))
            .unwrap_or(false);
        let gpu_sub = if multi_gpu {
            m.gpu_name.clone().unwrap_or_default()
        } else {
            m.gpu_name
                .clone()
                .or_else(|| m.gpu_note.clone())
                .unwrap_or_default()
        };
        let gen = m.gpu_pcie_gen_max.or(m.gpu_pcie_gen);
        let ttft = &m.ttft;
        let kpis = vec![
            Kpi {
                label: "GPU SPEED",
                device: String::new(),
                value: speed
                    .map(|v| format!("{v:.1}"))
                    .unwrap_or_else(|| "–".into()),
                sub: if m.state == "generating" {
                    "tok/s · decode now".into()
                } else {
                    "tok/s · decode last request".into()
                },
                color: ACCENT,
                viz: Viz::Spark(hist("tok_s")),
                glyph: Some("gauge"),
            },
            Kpi {
                label: "GPU PREFILL",
                device: String::new(),
                value: prefill.map(fmt_tokens_f).unwrap_or_else(|| "–".into()),
                sub: if m.state == "idle" {
                    "t/s · prefill last request".into()
                } else {
                    "t/s · prefill this request".into()
                },
                color: CYAN,
                viz: Viz::Spark(hist("prefill_tok_s_mean")),
                glyph: Some("gauge"),
            },
            Kpi {
                label: "GPU LOAD",
                device: String::new(),
                value: m
                    .gpu_util
                    .map(|v| format!("{v:.0}"))
                    .unwrap_or_else(|| "–".into()),
                sub: format!("% · {}", gpu_sub),
                color: GREEN,
                viz: Viz::Spark(hist("gpu_util")),
                glyph: Some("gauge"),
            },
            Kpi {
                label: "GPU VRAM",
                device: String::new(),
                value: m
                    .gpu_mem_used
                    .map(|v| format!("{:.1}", v / 1e9))
                    .unwrap_or_else(|| "–".into()),
                sub: m
                    .gpu_mem_total
                    .map(|t| {
                        format!(
                            "of {:.0} GB · {}",
                            t / 1e9,
                            fmt_tokens(m.expert_slots.unwrap_or(0))
                        )
                    })
                    .unwrap_or_else(|| "GB".into()),
                color: VIOLET,
                viz: Viz::Spark(hist("gpu_mem_used")),
                glyph: Some("gpu"),
            },
            Kpi {
                label: "GPU TEMP",
                device: String::new(),
                value: m
                    .gpu_temp
                    .map(|v| format!("{v:.0}"))
                    .unwrap_or_else(|| "–".into()),
                sub: "°C".into(),
                color: temp_color(m.gpu_temp.unwrap_or(0.0), 85.0),
                viz: Viz::Spark(hist("gpu_temp")),
                glyph: Some("thermo"),
            },
            Kpi {
                label: "GPU POWER",
                device: String::new(),
                value: m
                    .gpu_power
                    .map(|v| format!("{v:.0}"))
                    .unwrap_or_else(|| "–".into()),
                sub: m
                    .gpu_power_limit
                    .map(|l| format!("W · of {:.0} W limit", l))
                    .unwrap_or_else(|| "W".into()),
                color: AMBER,
                viz: Viz::Spark(hist("gpu_power")),
                glyph: Some("gauge"),
            },
            Kpi {
                label: "GPU PCIe",
                device: String::new(),
                value: gen.map(|g| format!("Gen{g}")).unwrap_or_else(|| "–".into()),
                sub: format!(
                    "{} · to GPU {}",
                    m.gpu_pcie_width
                        .map(|w| format!("x{w}"))
                        .unwrap_or_else(|| "x?".into()),
                    m.gpu_pcie_rx_mb
                        .map(|v| format!("{v:.1} MB/s"))
                        .unwrap_or_else(|| "–".into())
                ),
                color: ORANGE,
                viz: Viz::Spark(hist("gpu_pcie_rx_mb")),
                glyph: Some("net"),
            },
            Kpi {
                label: "CPU",
                device: String::new(),
                value: m
                    .cpu
                    .map(|v| format!("{v:.0}"))
                    .unwrap_or_else(|| "–".into()),
                sub: format!(
                    "% · {}",
                    match (m.cores, m.threads) {
                        (Some(c), Some(t)) => format!("{} cores · {} threads", c, t),
                        (None, Some(t)) => format!("{t} threads"),
                        _ => m.cpu_name.clone().unwrap_or_default(),
                    }
                ),
                color: ACCENT,
                viz: Viz::Spark(hist("cpu")),
                glyph: Some("cpu"),
            },
            Kpi {
                label: "DISK READ",
                device: String::new(),
                value: m
                    .disk_read_mb
                    .map(|v| {
                        if v >= 1000.0 {
                            format!("{:.2}", v / 1024.0)
                        } else {
                            format!("{v:.1}")
                        }
                    })
                    .unwrap_or_else(|| "–".into()),
                sub: m
                    .disk_write_mb
                    .map(|w| {
                        format!(
                            "{} · write {:.1} MB/s",
                            if m.disk_read_mb.unwrap_or(0.0) >= 1000.0 {
                                "GB/s"
                            } else {
                                "MB/s"
                            },
                            w
                        )
                    })
                    .unwrap_or_else(|| "MB/s".into()),
                color: GREEN,
                viz: Viz::Spark(hist("disk_read_mb")),
                glyph: Some("disk"),
            },
            Kpi {
                label: "FIRST TOKEN",
                device: String::new(),
                value: ttft
                    .p50
                    .map(|v| format!("{v:.2}"))
                    .unwrap_or_else(|| "–".into()),
                sub: if ttft.count > 0 {
                    format!(
                        "s · p90 {:.2} s · {} requests{}",
                        ttft.p90.unwrap_or(0.0),
                        ttft.count,
                        m.itl
                            .p50
                            .map(|v| format!(" · {:.0} ms between tokens", v * 1000.0))
                            .unwrap_or_default()
                    )
                } else {
                    "s · waiting for a request".into()
                },
                color: VIOLET,
                viz: Viz::Spark(
                    m.requests
                        .iter()
                        .rev()
                        .filter_map(|r| r.hit_rate.map(|h| h * 100.0))
                        .collect(),
                ),
                glyph: Some("gauge"),
            },
            Kpi {
                label: "BATCH SLOTS",
                device: String::new(),
                value: if m.parallel > 0 {
                    let busy = if m.running > 0 {
                        m.running
                    } else if m.state != "idle" {
                        1
                    } else {
                        0
                    };
                    format!("{busy}")
                } else {
                    (if m.queued > 0 {
                        m.queued
                    } else if m.state != "idle" {
                        1
                    } else {
                        0
                    })
                    .to_string()
                },
                sub: if m.parallel > 0 {
                    let mut parts = vec![format!("/ {} slots", m.parallel)];
                    if m.waiting > 0 {
                        parts.push(format!("{} waiting", m.waiting));
                    }
                    if m.queued > 0 {
                        parts.push(format!("{} queued", m.queued));
                    }
                    if m.batch_groups > 1 {
                        parts.push(format!("{} groups", m.batch_groups));
                    }
                    parts.join(" · ")
                } else {
                    "running · single slot · --batch runs more".into()
                },
                color: CYAN,
                viz: Viz::None,
                glyph: Some("gauge"),
            },
        ];
        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.x = 14.0;
            for (n, row) in kpis.chunks(4).enumerate() {
                if n > 0 {
                    ui.add_space(6.0);
                }
                ui.columns(4, |cols| {
                    for (col, k) in cols.iter_mut().zip(row) {
                        kpi_card(col, k);
                    }
                });
            }
        });
        ui.add_space(14.0);

        // ---- context fill + engine facts ----
        let used = strata::context_used(m, last);
        let frac_ctx = if m.max_context > 0 {
            ((used as f64 / m.max_context as f64).min(1.0)) as f32
        } else {
            0.0
        };
        ui.columns(2, |cols| {
            card(&mut cols[0], |ui| {
                card_title(ui, "Context fill", "");
                ui.horizontal(|ui| {
                    let (r, _) = ui.allocate_exact_size(Vec2::splat(72.0), Sense::hover());
                    ring(ui.painter(), r, frac_ctx, ACCENT);
                    ui.label(
                        RichText::new(format!("{}%", (frac_ctx * 100.0).round()))
                            .font(bold(20.0))
                            .color(TEXT),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!(
                                "{} / {}",
                                fmt_tokens(used),
                                fmt_tokens(m.max_context)
                            ))
                            .size(12.5)
                            .color(MUTED),
                        );
                    });
                });
                ui.add_space(4.0);
                let cache_bytes = m.expert_cache_mib.unwrap_or(0) * 1_048_576;
                fact_row(
                    ui,
                    "Experts in VRAM",
                    &match (m.expert_slots, m.gpu_mem_total) {
                        (Some(s), Some(_)) => {
                            format!("{} · {}", fmt_tokens(s), human_size(cache_bytes))
                        }
                        (Some(s), None) => fmt_tokens(s),
                        _ => "–".into(),
                    },
                    cache_bytes as f64 / m.gpu_mem_total.unwrap_or(0.0).max(1.0),
                );
                fact_row(
                    ui,
                    "System RAM",
                    &match (m.ram_used, m.ram_total) {
                        (Some(u), Some(t)) => {
                            let g = format!("{:.1}", u / 1e9);
                            format!("{g} / {:.0} GB", t / 1e9)
                        }
                        _ => "–".into(),
                    },
                    m.ram_used.unwrap_or(0.0) / m.ram_total.unwrap_or(1.0).max(1.0),
                );
                fact_row(
                    ui,
                    "GPU temperature",
                    &m.gpu_temp
                        .map(|t| format!("{t:.0} °C"))
                        .unwrap_or_else(|| "–".into()),
                    m.gpu_temp.unwrap_or(0.0) / 100.0,
                );
            });
            card(&mut cols[1], |ui| {
                card_title(ui, "Engine", "");
                let kv = [
                    ("Model", m.model.clone()),
                    ("Context", fmt_tokens(m.max_context)),
                    ("Images", if m.images { "on" } else { "off" }.into()),
                    (
                        "CPU",
                        m.cpu_name
                            .clone()
                            .unwrap_or_else(|| match (m.cores, m.threads) {
                                (Some(c), Some(t)) => format!("{c} cores · {t} threads"),
                                _ => String::new(),
                            }),
                    ),
                    ("GPU", m.gpu_name.clone().unwrap_or_else(|| "–".into())),
                ];
                for (k, v) in kv {
                    if !v.is_empty() {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(k).size(12.5).color(MUTED));
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.label(RichText::new(v).size(12.5).color(TEXT));
                                },
                            );
                        });
                    }
                }
            });
        });
        ui.add_space(14.0);

        // ---- recent requests ----
        let show_all = self.gui.strata_show_all;
        card(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new("Recent requests")
                        .font(bold(14.5))
                        .color(TEXT),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if m.requests_kept > 12
                        && ui
                            .button(
                                RichText::new(if show_all {
                                    "show fewer".to_string()
                                } else {
                                    format!("show all ({})", m.requests_kept)
                                })
                                .size(11.5)
                                .color(ACCENT),
                            )
                            .clicked()
                    {
                        self.gui.strata_show_all = !show_all;
                    }
                });
            });
            ui.add_space(4.0);
            if m.requests.is_empty() {
                ui.label(RichText::new("No requests yet").size(12.5).color(FAINT));
            } else {
                let cols: [(&str, f32, bool); 9] = [
                    ("TIME", 74.0, false),
                    ("STATUS", 88.0, false),
                    ("PROMPT", 64.0, true),
                    ("REUSED", 64.0, true),
                    ("OUTPUT", 64.0, true),
                    ("PREFILL t/s", 84.0, true),
                    ("DECODE t/s", 84.0, true),
                    ("HIT RATE", 96.0, true),
                    ("DURATION", 70.0, true),
                ];
                let rows: Vec<&strata::RequestRow> = if show_all {
                    m.requests.iter().collect()
                } else {
                    m.requests.iter().take(12).collect()
                };
                let line = |ui: &mut Ui, cells: Vec<(String, Color32, bool)>| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 0.0;
                        for ((_, w, right), (text, color, mono)) in cols.iter().zip(cells) {
                            let cell = Rect::from_min_size(ui.cursor().min, Vec2::new(*w, 18.0));
                            ui.allocate_exact_size(Vec2::new(*w, 18.0), Sense::hover());
                            let align = if *right {
                                egui::Align2::RIGHT_CENTER
                            } else {
                                egui::Align2::LEFT_CENTER
                            };
                            let font = if mono {
                                FontId::monospace(11.5)
                            } else {
                                FontId::proportional(11.5)
                            };
                            ui.painter().text(
                                Pos2::new(
                                    if *right {
                                        cell.right() - 6.0
                                    } else {
                                        cell.left()
                                    },
                                    cell.center().y,
                                ),
                                align,
                                text,
                                font,
                                color,
                            );
                        }
                    });
                };
                line(
                    ui,
                    cols.iter()
                        .map(|(h, _, _)| (h.to_string(), FAINT, false))
                        .collect(),
                );
                ui.painter().line_segment(
                    [
                        Pos2::new(ui.cursor().min.x, ui.cursor().min.y),
                        Pos2::new(ui.max_rect().right() - 18.0, ui.cursor().min.y),
                    ],
                    Stroke::new(1.0_f32, BORDER),
                );
                for r in rows {
                    let t = epoch_of(r.time);
                    let (h, mnt, s) = ((t / 3600) % 24, (t / 60) % 60, t % 60);
                    let pre = strata::prefill_rate(r);
                    let hit = r
                        .hit_rate
                        .map(|h| {
                            format!("{:.1}%", h * 100.0)
                                + &r.pcie_share
                                    .map(|p| format!(" +{:.1}% PCIe", p * 100.0))
                                    .unwrap_or_default()
                        })
                        .unwrap_or_else(|| "–".into());
                    line(
                        ui,
                        vec![
                            (format!("{h:02}:{mnt:02}:{s:02}"), MUTED, true),
                            (finish_label(&r.finish), finish_color(&r.finish), false),
                            (
                                r.prompt_tokens
                                    .map(fmt_tokens)
                                    .unwrap_or_else(|| "–".into()),
                                TEXT,
                                true,
                            ),
                            (
                                r.reused.map(fmt_tokens).unwrap_or_else(|| "0".into()),
                                MUTED,
                                true,
                            ),
                            (
                                r.output_tokens
                                    .map(fmt_tokens)
                                    .unwrap_or_else(|| "–".into()),
                                TEXT,
                                true,
                            ),
                            (
                                pre.map(fmt_tokens_f).unwrap_or_else(|| "–".into()),
                                MUTED,
                                true,
                            ),
                            (
                                r.decode_tok_s
                                    .map(|v| format!("{v:.1}"))
                                    .unwrap_or_else(|| "–".into()),
                                MUTED,
                                true,
                            ),
                            (hit, MUTED, true),
                            (
                                r.duration_s
                                    .map(|v| format!("{v:.1} s"))
                                    .unwrap_or_else(|| "–".into()),
                                MUTED,
                                true,
                            ),
                        ],
                    );
                }
            }
            if m.tot_requests > 0 {
                let read = m.tot_prompt_tokens - m.tot_reused;
                let p_speed = if m.tot_prompt_ms > 0.0 && read > 0 {
                    format!(
                        " at {} tok/s",
                        fmt_tokens_f(read as f64 / (m.tot_prompt_ms / 1000.0))
                    )
                } else {
                    String::new()
                };
                let o_speed = if m.tot_decode_ms > 0.0 && m.tot_output_tokens > 0 {
                    format!(
                        " at {:.1} tok/s",
                        m.tot_output_tokens as f64 / (m.tot_decode_ms / 1000.0)
                    )
                } else {
                    String::new()
                };
                let since = m
                    .totals_since
                    .map(|t| {
                        let t = epoch_of(t);
                        let (h, mnt, _s) = ((t / 3600) % 24, (t / 60) % 60, t % 60);
                        format!("{h:02}:{mnt:02}")
                    })
                    .unwrap_or_else(|| "start".into());
                ui.add_space(6.0);
                ui.label(
                    RichText::new(format!(
                        "Since {since}: {} requests · {} prompt tokens read{p_speed} ({} reused) · {} written{o_speed}",
                        fmt_tokens(m.tot_requests),
                        fmt_tokens(read),
                        fmt_tokens(m.tot_reused),
                        fmt_tokens(m.tot_output_tokens),
                    ))
                    .size(12.0)
                    .color(FAINT),
                );
            }
        });
        ui.add_space(14.0);

        // ---- conversation cache ----
        if let Some(c) = &m.conv {
            card(ui, |ui| {
                card_title(
                    ui,
                    "Conversation cache",
                    &format!(
                        "{} of {} requests reused part of their prompt",
                        fmt_tokens(c.requests_reused),
                        fmt_tokens(c.requests)
                    ),
                );
                if c.enabled {
                    bar_row(
                        ui,
                        "Slots",
                        c.parked.unwrap_or(0) as u64,
                        c.slots.unwrap_or(0) as u64,
                        ACCENT,
                    );
                    if let (Some(b), Some(bud)) = (c.bytes, c.budget_mib.map(|mib| mib * 1_048_576))
                    {
                        bar_row(ui, "Memory", b as u64, bud as u64, CYAN);
                    }
                    let switches = format!(
                        "{} parked · {} restored{}",
                        fmt_tokens(c.parks.unwrap_or(0)),
                        fmt_tokens(c.restores.unwrap_or(0)),
                        c.evictions
                            .filter(|e| *e > 0)
                            .map(|e| format!(" · {} evicted", fmt_tokens(e)))
                            .unwrap_or_default()
                    );
                    ui.label(RichText::new(switches).size(12.0).color(MUTED));
                }
                if c.last_prompt.is_some() {
                    ui.label(
                        RichText::new(format!(
                            "Last request: {} of {} prompt tokens reused",
                            fmt_tokens(c.last_reused.unwrap_or(0)),
                            fmt_tokens(c.last_prompt.unwrap_or(0))
                        ))
                        .size(12.0)
                        .color(MUTED),
                    );
                }
                let share = if c.prompt_tokens > 0 {
                    format!(
                        " ({:.1}% of all prompt tokens)",
                        100.0 * c.reused_tokens as f64 / c.prompt_tokens as f64
                    )
                } else {
                    String::new()
                };
                ui.label(
                    RichText::new(format!(
                        "Reused since start: {} tokens{share}",
                        fmt_tokens(c.reused_tokens)
                    ))
                    .size(12.0)
                    .color(MUTED),
                );
                ui.label(
                    RichText::new(if c.enabled {
                        "A request that continues a parked conversation gets its state back instead of reading it again; the oldest goes when the slots or the memory are full."
                    } else {
                        "The engine keeps the last conversation's state, so a follow-up reads only what is new. To keep several conversations, add \"--conversation-cache-mib\" to the run config's args."
                    })
                    .size(11.5)
                    .color(FAINT),
                );
            });
        }
    }

    fn volume_detail(&mut self, ui: &mut Ui) {
        let Some(name) = self.gui.selected_volume.clone() else {
            empty_state(
                ui,
                "No volume selected",
                "Pick a volume on the left to see its details.",
            );
            return;
        };
        let Some(v) = self.gui.volumes.iter().find(|v| v.name == name).cloned() else {
            empty_state(ui, "Volume not found", "That volume no longer exists.");
            return;
        };
        let refs = v.usage_data.as_ref().map(|u| u.ref_count);
        ui.horizontal(|ui| {
            ui.add(
                egui::Label::new(RichText::new(&v.name).font(bold(22.0)).color(TEXT)).truncate(),
            );
        });
        ui.horizontal(|ui| {
            chip(ui, &v.driver, MUTED);
            match refs {
                Some(0) => chip(ui, "unused", FAINT),
                Some(_) => chip(ui, "in use", GREEN),
                None => {}
            }
        });
        ui.add_space(16.0);
        let size = v
            .usage_data
            .as_ref()
            .and_then(|u| u.size)
            .filter(|s| *s >= 0);
        let kpis = [
            Kpi {
                label: "Size",
                device: String::new(),
                value: size.map(human_size).unwrap_or_else(|| "—".into()),
                sub: if size.is_some() {
                    "on disk".into()
                } else {
                    "not reported by driver".into()
                },
                color: VIOLET,
                viz: Viz::None,
                glyph: None,
            },
            Kpi {
                label: "References",
                device: String::new(),
                value: refs.map(|n| n.to_string()).unwrap_or_else(|| "—".into()),
                sub: if refs.is_some() {
                    "containers mounting it".into()
                } else {
                    "not reported by the daemon".into()
                },
                color: GREEN,
                viz: Viz::None,
                glyph: None,
            },
            Kpi {
                label: "Created",
                device: String::new(),
                value: v.created.get(..10).unwrap_or("—").to_string(),
                sub: v.created.get(11..19).unwrap_or("").to_string(),
                color: ACCENT,
                viz: Viz::None,
                glyph: None,
            },
        ];
        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.x = 14.0;
            ui.columns(3, |cols| {
                for (col, k) in cols.iter_mut().zip(&kpis) {
                    kpi_card(col, k);
                }
            });
        });
        ui.add_space(14.0);
        // The list endpoint omits mounts, so match on container names.
        let users: Vec<String> = self
            .gui
            .containers
            .iter()
            .filter(|c| c.names.iter().any(|n| n.contains(&v.name)))
            .map(|c| c.short_name())
            .collect();
        card(ui, |ui| {
            card_title(ui, "In use by", "");
            if users.is_empty() {
                ui.label(
                    RichText::new("No container names match this volume. The list API omits mounts; `docker volume inspect` shows exact users.")
                        .color(FAINT),
                );
            }
            for u in users {
                ui.label(RichText::new(u).font(FontId::monospace(12.5)).color(TEXT));
            }
        });
    }

    // ------------------------------------------------------------------
    // Logs
    // ------------------------------------------------------------------

    fn logs_view(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        let is_all = g.logs_container.as_deref() == Some("__all__");
        let sub = if g.logs_container.is_none() {
            "Loading…".to_string()
        } else if is_all {
            format!("All running containers · {} lines", g.logs.len())
        } else {
            format!("{} · {} lines", g.logs_container_name, g.logs.len())
        };
        let mut back = false;
        let mut to_all = false;
        let mut reopen = false;
        ui.horizontal(|ui| {
            ui.vertical(|ui| page_header(ui, "Logs", &sub));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if is_all {
                    if button(ui, "↻  Reload", Btn::Secondary, true).clicked() {
                        to_all = true;
                    }
                } else if g.logs_container.is_some() {
                    if button(ui, "All containers", Btn::Secondary, true).clicked() {
                        to_all = true;
                    }
                    if button(ui, "↻  Reopen", Btn::Secondary, true).clicked() {
                        reopen = true;
                    }
                    if button(ui, "← Container", Btn::Secondary, true).clicked() {
                        back = true;
                    }
                }
            });
        });
        ui.add_space(16.0);
        if back {
            self.gui.tab = Tab::Containers;
        } else if to_all {
            self.load_all_logs();
        } else if reopen {
            if let Some(id) = self.gui.logs_container.clone() {
                self.open_logs(&id);
            }
        }
        if self.gui.logs_container.is_none() {
            empty_state(
                ui,
                "No logs yet",
                "Pick a container, or reopen this tab to stream all containers.",
            );
            return;
        }
        self.logs_card(ui, "Output", true);
    }

    /// Log card: toolbar (search, follow) over a virtualized log well.
    fn logs_card(&mut self, ui: &mut Ui, title: &str, show_source: bool) {
        let mut follow = self.gui.logs_follow;
        let streaming = self.gui.logs_active;
        card(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(title).font(bold(14.5)).color(TEXT));
                if streaming {
                    live_pill(ui, true);
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let label = if follow { "Following" } else { "Follow" };
                    if toggle_chip(ui, label, follow).clicked() {
                        follow = !follow;
                    }
                    ui.add(
                        egui::TextEdit::singleline(&mut self.gui.log_filter)
                            .hint_text(RichText::new("Filter lines").color(FAINT))
                            .desired_width(200.0)
                            .margin(Margin::symmetric(10, 5)),
                    );
                });
            });
            ui.add_space(6.0);
            self.logs_page(ui, show_source);
        });
        self.gui.logs_follow = follow;
    }

    /// Shared log renderer with fade-in and fresh-line flash; only the rows
    /// in view are laid out, so long pages stay cheap.
    fn logs_page(&mut self, ui: &mut Ui, show_source: bool) {
        let g = &self.gui;
        let age = self.logs_opened.elapsed().as_secs_f32();
        let page_alpha = (age / 0.45).clamp(0.0, 1.0);
        let fresh_start = self.prev_log_len.min(g.logs.len());
        let fresh_age = age.min(2.0);
        let all = show_source && g.logs_container.as_deref() == Some("__all__");
        let needle = g.log_filter.to_lowercase();
        let idx: Vec<usize> = if needle.is_empty() {
            (0..g.logs.len()).collect()
        } else {
            g.logs
                .iter()
                .enumerate()
                .filter(|(_, l)| l.text.to_lowercase().contains(&needle))
                .map(|(i, _)| i)
                .collect()
        };
        let height = ui.available_height().max(220.0);
        let font = FontId::monospace(12.5);
        let row_h = ui.fonts_mut(|f| f.row_height(&font)) + 3.0;
        Frame::NONE
            .fill(WELL)
            .corner_radius(10)
            .inner_margin(Margin::symmetric(12, 10))
            .show(ui, |ui| {
                ui.set_min_height(height - 20.0);
                ui.set_width(ui.available_width());
                if idx.is_empty() {
                    ui.label(
                        RichText::new(if g.logs.is_empty() {
                            "Waiting for output…"
                        } else {
                            "No lines match the filter."
                        })
                        .font(font.clone())
                        .color(FAINT),
                    );
                    return;
                }
                ui.spacing_mut().item_spacing.y = 3.0;
                ScrollArea::both()
                    .auto_shrink([false, false])
                    .max_height(height - 20.0)
                    .stick_to_bottom(g.logs_follow)
                    .show_rows(ui, row_h - 3.0, idx.len(), |ui, range| {
                        for &i in &idx[range] {
                            let l = &g.logs[i];
                            let mut a = page_alpha;
                            if i >= fresh_start {
                                let t = (fresh_age / 0.35).clamp(0.0, 1.0);
                                a = a.max(0.35 + 0.65 * t);
                            }
                            let job = log_line_job(
                                l,
                                if all { l.source.as_deref() } else { None },
                                &font,
                                a,
                            );
                            ui.add(egui::Label::new(job).extend());
                        }
                    });
            });
        self.prev_log_len = g.logs.len();
    }

    fn remove_dialog(&mut self, ctx: &egui::Context, id: &str) {
        let name = self
            .gui
            .containers
            .iter()
            .find(|c| c.id == id)
            .map(|c| c.short_name())
            .unwrap_or_else(|| id.to_string());
        let mut confirm = false;
        let mut cancel = false;
        let resp = egui::Modal::new(egui::Id::new("remove-modal"))
            .backdrop_color(Color32::from_black_alpha(150))
            .frame(
                Frame::NONE
                    .fill(SURFACE)
                    .stroke(Stroke::new(1.0_f32, BORDER))
                    .corner_radius(14)
                    .shadow(ctx.style().visuals.window_shadow)
                    .inner_margin(Margin::same(22)),
            )
            .show(ctx, |ui| {
                ui.set_width(380.0);
                ui.label(
                    RichText::new("Remove container?")
                        .font(bold(17.0))
                        .color(TEXT),
                );
                ui.add_space(4.0);
                ui.label(
                    RichText::new(format!(
                        "{name} will be deleted. Its writable layer is lost; volumes are kept."
                    ))
                    .color(MUTED),
                );
                ui.add_space(14.0);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if button(ui, "Remove", Btn::Danger, true).clicked() {
                        confirm = true;
                    }
                    if button(ui, "Cancel", Btn::Secondary, true).clicked() {
                        cancel = true;
                    }
                });
            });
        if resp.should_close() {
            cancel = true;
        }
        if confirm {
            if let Some(c) = self.gui.containers.iter().find(|c| c.id == id).cloned() {
                self.gui.pending_action = Some((id.to_string(), Pending::Remove));
                spawn_action(&self.rt, &self.docker, &self.tx, &c, Pending::Remove);
            }
            self.gui.confirm_remove = None;
        } else if cancel {
            self.gui.confirm_remove = None;
            self.gui.status = "remove cancelled".to_string();
        }
    }

    fn open_logs(&mut self, id: &str) {
        let g = &mut self.gui;
        let Some(c) = g.containers.iter().find(|c| c.id == id).cloned() else {
            return;
        };
        // Closing the consolidated page: drop its follower tasks.
        if g.logs_container.as_deref() == Some("__all__") {
            g.logs_gen += 1;
            self.all_logs_follow.clear();
        }
        g.logs_gen += 1;
        g.pending_logs_gen = Some(g.logs_gen);
        g.logs_container = Some(id.to_string());
        g.logs_container_name = c.short_name();
        g.logs.clear();
        g.logs_follow = true;
        g.logs_active = false;
        self.logs_opened = Instant::now();
        self.prev_log_len = 0;
        spawn_logs(&self.rt, &self.docker, &self.tx, id, g.logs_gen, None);
    }
}

// ---------------------------------------------------------------------------
// Small widgets
// ---------------------------------------------------------------------------

/// Circular gauge ring (the Strata context-fill dial).
fn ring(p: &egui::Painter, r: Rect, frac: f32, color: Color32) {
    let stroke = Stroke::new(6.0_f32, SURFACE_HI);
    p.circle(
        r.center(),
        r.width() / 2.0 - 3.0,
        Color32::TRANSPARENT,
        stroke,
    );
    if frac > 0.001 {
        let frac = frac.clamp(0.0, 1.0);
        let mut pts = Vec::new();
        let steps = (64.0 * frac).max(2.0) as usize;
        let rad = r.width() / 2.0 - 3.0;
        for i in 0..=steps {
            let a = std::f32::consts::TAU * (i as f32 / steps as f32) - std::f32::consts::FRAC_PI_2;
            pts.push(Pos2::new(
                r.center().x + rad * a.cos(),
                r.center().y + rad * a.sin(),
            ));
        }
        p.add(egui::Shape::line(pts, Stroke::new(6.0_f32, color)));
    }
}

/// A "label …… value" row with a thin progress bar under it.
fn fact_row(ui: &mut Ui, label: &str, value: &str, frac: f64) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(label).size(12.5).color(MUTED));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(value).size(12.5).color(TEXT));
        });
    });
    let (r, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 4.0), Sense::hover());
    progress(ui.painter(), r, frac as f32, ACCENT);
}

/// Thousands-grouped integer, like the web monitor's kfmt.
fn fmt_tokens(n: i64) -> String {
    let digits = n.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}

/// Thousands-grouped rate, e.g. 117,7 t/s stays "117.7" but 1177 -> "1,177".
fn fmt_tokens_f(v: f64) -> String {
    if v >= 100.0 {
        fmt_tokens(v.round() as i64)
    } else {
        format!("{v:.1}")
    }
}

/// Epoch seconds to local-ish wall clock (same shift as now_string).
fn epoch_of(t: f64) -> i64 {
    t as i64
}

fn finish_label(finish: &str) -> String {
    match finish {
        "stop" => "Done",
        "length" => "Max tokens",
        "cancel" => "Stopped",
        "disconnect" => "Closed",
        "error" => "Error",
        "" => "–",
        other => other,
    }
    .to_string()
}

fn finish_color(finish: &str) -> Color32 {
    match finish {
        "stop" => GREEN,
        "length" => ACCENT,
        "cancel" | "disconnect" => AMBER,
        "error" => RED,
        _ => FAINT,
    }
}

/// Warm color for a temperature: green under 60% of `hot`, amber to 85%,
/// red beyond.
fn temp_color(t: f64, hot: f64) -> Color32 {
    if t >= hot {
        RED
    } else if t >= hot * 0.7 {
        AMBER
    } else {
        GREEN
    }
}

/// Dashboard / detail metric tile.
struct Kpi {
    label: &'static str,
    /// device identity shown above the number, e.g. "AMD Ryzen 7 7800X3D"
    device: String,
    value: String,
    sub: String,
    color: Color32,
    viz: Viz,
    /// optional line-icon drawn in the tile's top-right corner
    glyph: Option<&'static str>,
}

/// What a metric tile draws under its number.
enum Viz {
    None,
    Ratio(f32),
    Spark(Vec<f64>),
}

fn kpi_card(ui: &mut Ui, k: &Kpi) {
    Frame::NONE
        .fill(SURFACE)
        .stroke(Stroke::new(1.0_f32, BORDER))
        .corner_radius(14)
        .inner_margin(Margin::symmetric(16, 14))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing.y = 2.0;
            if !k.device.is_empty() {
                ui.add(
                    egui::Label::new(RichText::new(&k.device).font(bold(13.0)).color(TEXT))
                        .truncate(),
                );
            }
            ui.horizontal(|ui| {
                let (r, _) = ui.allocate_exact_size(Vec2::splat(8.0), Sense::hover());
                ui.painter().circle_filled(r.center(), 3.5, k.color);
                ui.label(RichText::new(k.label).size(12.5).color(MUTED));
                if let Some(g) = k.glyph {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let (gr, _) = ui.allocate_exact_size(Vec2::splat(16.0), Sense::hover());
                        icon_glyph(
                            ui.painter(),
                            g,
                            gr.center(),
                            7.0,
                            k.color.gamma_multiply(0.75),
                        );
                    });
                }
            });
            ui.add(
                egui::Label::new(RichText::new(&k.value).font(bold(24.0)).color(TEXT)).truncate(),
            );
            ui.add(egui::Label::new(RichText::new(&k.sub).size(12.0).color(FAINT)).truncate());
            ui.add_space(6.0);
            let (r, _) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 28.0), Sense::hover());
            let p = ui.painter();
            match &k.viz {
                Viz::None => {}
                Viz::Ratio(f) => {
                    let bar = Rect::from_center_size(r.center(), Vec2::new(r.width(), 6.0));
                    progress(p, bar, *f, k.color);
                }
                Viz::Spark(s) if s.len() >= 2 => {
                    let mut pts = area_points(r, s, series_max(s, 1.0));
                    fill_under(p, &pts, r.bottom(), k.color, 60);
                    pts.dedup();
                    p.add(egui::Shape::line(pts, Stroke::new(1.5_f32, k.color)));
                }
                Viz::Spark(_) => {
                    p.text(
                        r.left_center(),
                        egui::Align2::LEFT_CENTER,
                        "collecting…",
                        FontId::proportional(11.5),
                        FAINT,
                    );
                }
            }
        });
}

/// Dashboard card for one feature: icon + title, then `add`'s summary.
/// The whole card is the click target; true when clicked.
fn overview_card(ui: &mut Ui, tab: Tab, title: &str, add: impl FnOnce(&mut Ui)) -> bool {
    let inner = Frame::NONE
        .fill(SURFACE)
        .stroke(Stroke::new(1.0_f32, BORDER))
        .corner_radius(14)
        .inner_margin(Margin::symmetric(16, 14))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.set_min_height(176.0);
            ui.spacing_mut().item_spacing.y = 2.0;
            ui.horizontal(|ui| {
                let (r, _) = ui.allocate_exact_size(Vec2::splat(16.0), Sense::hover());
                nav_icon(ui.painter(), tab, r, MUTED);
                ui.label(RichText::new(title).font(bold(14.0)).color(TEXT));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new("›").size(16.0).color(FAINT));
                });
            });
            ui.add_space(6.0);
            add(ui);
        });
    let rect = inner.response.rect;
    let resp = ui.interact(rect, ui.id().with(("overview", title)), Sense::click());
    #[cfg(test)]
    DASH_CARD_RECTS
        .lock()
        .unwrap()
        .push((tab, rect, title.to_string()));
    if resp.hovered() {
        ui.painter().rect_stroke(
            rect,
            14.0,
            Stroke::new(1.0_f32, ACCENT.gamma_multiply(0.6)),
            egui::epaint::StrokeKind::Inside,
        );
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    resp.clicked()
}

/// Big number (or state word) with a caption beside it.
fn headline(ui: &mut Ui, value: &str, caption: &str, color: Color32) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(value).font(bold(24.0)).color(color));
        ui.label(RichText::new(caption).size(12.5).color(MUTED));
    });
}

fn meta_line(ui: &mut Ui, text: &str) {
    ui.add(egui::Label::new(RichText::new(text).size(12.0).color(FAINT)).truncate());
    ui.add_space(4.0);
}

/// Compact dot · name ……… value row for the dashboard cards.
fn mini_row(ui: &mut Ui, dot: Color32, name: &str, value: &str) {
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 24.0), Sense::hover());
    let p = ui.painter();
    p.circle_filled(Pos2::new(rect.left() + 4.0, rect.center().y), 3.0, dot);
    let v = p.layout_no_wrap(value.to_string(), FontId::monospace(12.0), MUTED);
    let vw = v.size().x;
    p.galley(
        Pos2::new(rect.right() - vw, rect.center().y - v.size().y / 2.0),
        v,
        MUTED,
    );
    let n = elide(
        p,
        name.to_string(),
        FontId::proportional(13.0),
        TEXT,
        rect.width() - vw - 28.0,
    );
    p.galley(
        Pos2::new(rect.left() + 14.0, rect.center().y - n.size().y / 2.0),
        n,
        TEXT,
    );
}

fn tab_label(tab: Tab) -> &'static str {
    match tab {
        Tab::Dashboard => "Dashboard",
        Tab::Containers => "Containers",
        Tab::Images => "Images",
        Tab::Volumes => "Volumes",
        Tab::Agents => "Agents",
        Tab::Hardware => "Hardware",
        Tab::Ports => "Ports",
        Tab::Processes => "Processes",
        Tab::Strata => "Strata",
        Tab::Logs => "Logs",
        Tab::Settings => "Settings",
    }
}

/// Surface card with standard padding; fills the available width.
/// One physical drive's stats card: named title, rates, lifetime counters.
fn disk_card(
    ui: &mut Ui,
    d: &crate::hw::DiskStat,
    inv: Option<&hwinv::HwDevice>,
    usage: &[hwinv::DiskUsage],
) {
    let title = inv.map(|x| x.name.as_str()).unwrap_or(&d.name);
    let temp = d
        .temp_c
        .map(|t| format!("{} °C", temp_str(t)))
        .unwrap_or_default();
    card(ui, |ui| {
        card_title(ui, title, &temp);
        ui.horizontal(|ui| {
            ui.label(RichText::new("Rate").size(12.5).color(MUTED).weak());
            ui.label(
                RichText::new(format!("↓ {}", human_rate(d.read_bps)))
                    .font(FontId::monospace(12.5))
                    .color(TEXT),
            );
            ui.label(
                RichText::new(format!("↑ {}", human_rate(d.write_bps)))
                    .font(FontId::monospace(12.5))
                    .color(TEXT),
            );
            if let Some(inv) = inv {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(RichText::new(&inv.detail).size(11.5).color(FAINT));
                });
            }
        });
        // filesystem capacity on this drive (root, home, …), from /proc/mounts
        for u in usage.iter().filter(|u| u.kern == d.name) {
            let frac = u.used as f32 / u.total.max(1) as f32;
            let col = if frac > 0.9 {
                RED
            } else if frac > 0.75 {
                AMBER
            } else {
                GREEN
            };
            bar_row(ui, &u.mount, u.used, u.total, col);
        }
        ui.horizontal(|ui| {
            ui.label(RichText::new("Lifetime").size(12.5).color(MUTED).weak());
            ui.label(
                RichText::new(format!("{} read", human_size(d.total_read as i64)))
                    .font(FontId::monospace(12.0))
                    .color(MUTED),
            );
            ui.label(
                RichText::new(format!("{} written", human_size(d.total_write as i64)))
                    .font(FontId::monospace(12.0))
                    .color(MUTED),
            );
            ui.label(
                RichText::new(format!("{} reads · {} writes", d.reads, d.writes))
                    .size(11.5)
                    .color(FAINT),
            );
            // average queue depth = weighted busy ms / elapsed ms
            let queue = if d.uptime > 0.0 {
                d.ms_weighted as f64 / (d.uptime * 1000.0)
            } else {
                0.0
            };
            ui.label(
                RichText::new(format!("avg queue {queue:.2}"))
                    .size(11.5)
                    .color(FAINT),
            );
        });
        if let Some(t) = d.temp_c {
            let c = if t > 70.0 {
                RED
            } else if t > 60.0 {
                AMBER
            } else {
                GREEN
            };
            ui.label(
                RichText::new(format!("drive temperature {} °C", temp_str(t)))
                    .size(11.5)
                    .color(c),
            );
        }
    });
}

fn card<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    Frame::NONE
        .fill(SURFACE)
        .stroke(Stroke::new(1.0_f32, BORDER))
        .corner_radius(14)
        .inner_margin(Margin::same(18))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.spacing_mut().item_spacing = Vec2::new(8.0, 8.0);
            add(ui)
        })
        .inner
}

fn card_title(ui: &mut Ui, title: &str, right: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(title).font(bold(14.5)).color(TEXT));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(RichText::new(right).size(12.0).color(FAINT));
        });
    });
    ui.add_space(4.0);
}

/// Small line icons for inline buttons (eye, refresh) drawn with the
/// painter, matching the nav-rail icon language.
/// Per-family accent for the process-row icon tiles; families that share a
/// colour family (browsers, terminals) stay visually distinct.
fn proc_icon_color(kind: &str) -> Color32 {
    match kind {
        "browser" => ACCENT,
        "terminal" => GREEN,
        "service" | "kernel" => FAINT,
        "container" => CYAN,
        "database" => ORANGE,
        "runtime" => VIOLET,
        "chat" => GREEN,
        "music" => ORANGE,
        "media" => VIOLET,
        "files" => AMBER,
        "game" => RED,
        "agent" => ACCENT,
        "audio" => ORANGE,
        "desktop" => CYAN,
        "settings" => MUTED,
        _ => MUTED,
    }
}

fn icon_glyph(p: &egui::Painter, kind: &str, c: Pos2, r: f32, col: Color32) {
    let s = Stroke::new(1.5_f32, col);
    match kind {
        "cpu" => {
            // chip: square, inner die, four pins per side
            let b = Rect::from_center_size(c, Vec2::splat(r * 1.35));
            p.rect_stroke(
                b,
                2.0,
                Stroke::new(1.4_f32, col),
                egui::epaint::StrokeKind::Inside,
            );
            p.rect_stroke(
                b.shrink(2.4),
                1.0,
                Stroke::new(1.1_f32, col),
                egui::epaint::StrokeKind::Inside,
            );
            for i in 0..4 {
                let t = b.left() + b.width() * (i as f32 + 0.5) / 4.0;
                p.line_segment([Pos2::new(t, b.top()), Pos2::new(t, b.top() - r * 0.3)], s);
                p.line_segment(
                    [Pos2::new(t, b.bottom()), Pos2::new(t, b.bottom() + r * 0.3)],
                    s,
                );
                let u = b.top() + b.height() * (i as f32 + 0.5) / 4.0;
                p.line_segment(
                    [Pos2::new(b.left(), u), Pos2::new(b.left() - r * 0.3, u)],
                    s,
                );
                p.line_segment(
                    [Pos2::new(b.right(), u), Pos2::new(b.right() + r * 0.3, u)],
                    s,
                );
            }
        }
        "ram" => {
            // memory module: body, notch, three cells, pins below
            let b = Rect::from_center_size(c, Vec2::new(r * 2.3, r * 1.3));
            p.rect_stroke(
                b,
                1.5,
                Stroke::new(1.4_f32, col),
                egui::epaint::StrokeKind::Inside,
            );
            for i in 1..=3 {
                let x = b.left() + b.width() * i as f32 / 4.0;
                p.line_segment(
                    [Pos2::new(x, b.top() + 1.5), Pos2::new(x, b.bottom() - 1.5)],
                    Stroke::new(1.1_f32, col),
                );
            }
            for i in 0..4 {
                let x = b.left() + b.width() * (i as f32 + 0.5) / 4.0;
                p.line_segment(
                    [Pos2::new(x, b.bottom()), Pos2::new(x, b.bottom() + r * 0.3)],
                    s,
                );
            }
        }
        "disk" => {
            // platter: rounded body, spindle dot, actuator arm
            let b = Rect::from_center_size(c, Vec2::new(r * 2.1, r * 1.7));
            p.rect_stroke(
                b,
                2.5,
                Stroke::new(1.4_f32, col),
                egui::epaint::StrokeKind::Inside,
            );
            p.circle_filled(Pos2::new(b.left() + r * 0.62, b.center().y), r * 0.28, col);
            p.line_segment(
                [
                    Pos2::new(b.left() + r * 0.62, b.center().y),
                    Pos2::new(b.right() - r * 0.55, b.top() + r * 0.42),
                ],
                s,
            );
        }
        "gauge" => {
            // speedometer: arc + needle
            let arc = |a: f32| c + Vec2::new(a.cos(), a.sin()) * r;
            let a0 = std::f32::consts::PI * 0.85;
            let a1 = std::f32::consts::PI * 0.15;
            let mut prev = arc(a0);
            for k in 1..=10 {
                let a = a0 + (std::f32::consts::TAU - (a0 - a1)) * k as f32 / 10.0;
                let pt = arc(a);
                p.line_segment([prev, pt], s);
                prev = pt;
            }
            let na = std::f32::consts::PI * 0.72;
            p.line_segment([c, c + (arc(na) - c) * 0.8], Stroke::new(1.6_f32, col));
        }
        "gpu" => {
            // card: bracket on the left, long body, two fan circles
            let b =
                Rect::from_center_size(c + Vec2::new(r * 0.25, 0.0), Vec2::new(r * 2.1, r * 1.35));
            p.rect_stroke(
                b,
                2.0,
                Stroke::new(1.4_f32, col),
                egui::epaint::StrokeKind::Inside,
            );
            p.line_segment(
                [
                    Pos2::new(b.left() - r * 0.45, b.top() - r * 0.2),
                    Pos2::new(b.left() - r * 0.45, b.bottom() + r * 0.2),
                ],
                Stroke::new(1.8_f32, col),
            );
            p.circle_stroke(
                Pos2::new(b.left() + b.width() * 0.33, b.center().y),
                r * 0.42,
                Stroke::new(1.1_f32, col),
            );
            p.circle_stroke(
                Pos2::new(b.left() + b.width() * 0.72, b.center().y),
                r * 0.42,
                Stroke::new(1.1_f32, col),
            );
        }
        "thermo" => {
            // thermometer: stem + bulb
            p.circle_stroke(Pos2::new(c.x, c.y + r * 0.62), r * 0.5, s);
            p.circle_filled(Pos2::new(c.x, c.y + r * 0.62), r * 0.24, col);
            p.line_segment(
                [
                    Pos2::new(c.x - r * 0.28, c.y - r * 0.9),
                    Pos2::new(c.x - r * 0.28, c.y + r * 0.35),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(c.x + r * 0.28, c.y - r * 0.9),
                    Pos2::new(c.x + r * 0.28, c.y + r * 0.35),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(c.x - r * 0.28, c.y - r * 0.9),
                    Pos2::new(c.x + r * 0.28, c.y - r * 0.9),
                ],
                s,
            );
        }
        "net" => {
            // up/down arrows side by side
            let x1 = c.x - r * 0.5;
            let x2 = c.x + r * 0.5;
            p.line_segment(
                [Pos2::new(x1, c.y + r * 0.8), Pos2::new(x1, c.y - r * 0.8)],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(x1 - r * 0.35, c.y - r * 0.4),
                    Pos2::new(x1, c.y - r * 0.85),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(x1 + r * 0.35, c.y - r * 0.4),
                    Pos2::new(x1, c.y - r * 0.85),
                ],
                s,
            );
            p.line_segment(
                [Pos2::new(x2, c.y - r * 0.8), Pos2::new(x2, c.y + r * 0.8)],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(x2 - r * 0.35, c.y + r * 0.4),
                    Pos2::new(x2, c.y + r * 0.85),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(x2 + r * 0.35, c.y + r * 0.4),
                    Pos2::new(x2, c.y + r * 0.85),
                ],
                s,
            );
        }
        "eye" => {
            // almond outline + pupil
            let w = r * 1.35;
            let h = r * 0.78;
            p.add(egui::epaint::CubicBezierShape::from_points_stroke(
                [
                    c + Vec2::new(-w, 0.0),
                    c + Vec2::new(-w * 0.42, -h),
                    c + Vec2::new(w * 0.42, -h),
                    c + Vec2::new(w, 0.0),
                ],
                false,
                Color32::TRANSPARENT,
                s,
            ));
            p.add(egui::epaint::CubicBezierShape::from_points_stroke(
                [
                    c + Vec2::new(w, 0.0),
                    c + Vec2::new(w * 0.42, h),
                    c + Vec2::new(-w * 0.42, h),
                    c + Vec2::new(-w, 0.0),
                ],
                false,
                Color32::TRANSPARENT,
                s,
            ));
            p.circle_filled(c, r * 0.34, col);
        }
        "eye_off" => {
            // almond with the pupil shrunk to a ring, plus the slash
            let w = r * 1.35;
            let h = r * 0.78;
            p.add(egui::epaint::CubicBezierShape::from_points_stroke(
                [
                    c + Vec2::new(-w, 0.0),
                    c + Vec2::new(-w * 0.42, -h),
                    c + Vec2::new(w * 0.42, -h),
                    c + Vec2::new(w, 0.0),
                ],
                false,
                Color32::TRANSPARENT,
                s,
            ));
            p.add(egui::epaint::CubicBezierShape::from_points_stroke(
                [
                    c + Vec2::new(w, 0.0),
                    c + Vec2::new(w * 0.42, h),
                    c + Vec2::new(-w * 0.42, h),
                    c + Vec2::new(-w, 0.0),
                ],
                false,
                Color32::TRANSPARENT,
                s,
            ));
            p.circle_stroke(c, r * 0.34, s);
            p.line_segment(
                [
                    c + Vec2::new(-r * 1.15, r * 1.05),
                    c + Vec2::new(r * 1.15, -r * 1.05),
                ],
                Stroke::new(1.6_f32, col),
            );
        }
        "refresh" => {
            // open arc + arrowhead
            let arc = |a: f32| c + Vec2::new(a.cos(), a.sin()) * r;
            let a0 = -2.4_f32;
            let a1 = 1.9_f32;
            let mut prev = arc(a0);
            for k in 1..=14 {
                let a = a0 + (a1 - a0) * k as f32 / 14.0;
                let pt = arc(a);
                p.line_segment([prev, pt], s);
                prev = pt;
            }
            let tip = arc(a1);
            let tang = Vec2::new(
                (a1 + std::f32::consts::FRAC_2_PI).cos(),
                (a1 + std::f32::consts::FRAC_2_PI).sin(),
            );
            let nrm = Vec2::new(-tang.y, tang.x);
            p.line_segment([tip, tip - tang * r * 0.55 + nrm * r * 0.38], s);
            p.line_segment([tip, tip - tang * r * 0.55 - nrm * r * 0.38], s);
        }
        "browser" => {
            // globe: circle, equator, one meridian ellipse
            p.circle_stroke(c, r, Stroke::new(1.4_f32, col));
            p.line_segment([Pos2::new(c.x - r, c.y), Pos2::new(c.x + r, c.y)], s);
            p.add(egui::Shape::ellipse_stroke(
                c,
                Vec2::new(r * 0.45, r),
                Stroke::new(1.2_f32, col),
            ));
        }
        "terminal" => {
            // window with a prompt caret
            let b = Rect::from_center_size(c, Vec2::new(r * 2.0, r * 1.6));
            p.rect_stroke(
                b,
                3.0,
                Stroke::new(1.4_f32, col),
                egui::epaint::StrokeKind::Inside,
            );
            p.line_segment(
                [
                    Pos2::new(b.left() + 3.0, b.top() + 4.0),
                    Pos2::new(b.right() - 3.0, b.top() + 4.0),
                ],
                Stroke::new(1.1_f32, col),
            );
            let x = b.left() + 4.0;
            let y = b.center().y + 1.0;
            p.line_segment([Pos2::new(x, y - 2.5), Pos2::new(x + 2.5, y)], s);
            p.line_segment([Pos2::new(x + 2.5, y), Pos2::new(x, y + 2.5)], s);
            p.line_segment(
                [Pos2::new(x + 4.0, y + 2.5), Pos2::new(x + 8.0, y + 2.5)],
                s,
            );
        }
        "service" => {
            // gear: hub + six teeth
            p.circle_stroke(c, r * 0.55, Stroke::new(1.3_f32, col));
            for i in 0..6 {
                let a = std::f32::consts::TAU * i as f32 / 6.0;
                let u = Vec2::new(a.cos(), a.sin());
                p.line_segment([c + u * r * 0.7, c + u * r], Stroke::new(1.6_f32, col));
            }
        }
        "kernel" => {
            // hexagon node: the kernel mark
            let mut pts: Vec<Pos2> = (0..6)
                .map(|i| {
                    let a = std::f32::consts::TAU * i as f32 / 6.0 - std::f32::consts::FRAC_PI_2;
                    c + Vec2::new(a.cos(), a.sin()) * r
                })
                .collect();
            pts.push(pts[0]);
            p.add(egui::Shape::closed_line(pts, Stroke::new(1.4_f32, col)));
            p.circle_filled(c, r * 0.22, col);
        }
        "container" => {
            // docker-ish: rounded hull, two portholes, mast line
            let b =
                Rect::from_center_size(c + Vec2::new(0.0, r * 0.15), Vec2::new(r * 1.9, r * 1.2));
            p.rect_stroke(
                b,
                4.0,
                Stroke::new(1.4_f32, col),
                egui::epaint::StrokeKind::Inside,
            );
            for f in [0.35, 0.65] {
                p.circle_stroke(
                    Pos2::new(b.left() + b.width() * f, b.center().y + 1.0),
                    r * 0.18,
                    Stroke::new(1.1_f32, col),
                );
            }
            p.line_segment(
                [
                    Pos2::new(b.left() + b.width() * 0.25, b.top() - r * 0.35),
                    Pos2::new(b.left() + b.width() * 0.75, b.top() - r * 0.35),
                ],
                s,
            );
        }
        "database" => {
            // cylinder: top ellipse + sides + bottom arc
            let rx = r * 0.85;
            let top = Pos2::new(c.x, c.y - r * 0.7);
            p.add(egui::Shape::ellipse_stroke(
                top,
                Vec2::new(rx, r * 0.32),
                Stroke::new(1.3_f32, col),
            ));
            p.line_segment(
                [
                    Pos2::new(c.x - rx, top.y),
                    Pos2::new(c.x - rx, c.y + r * 0.55),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(c.x + rx, top.y),
                    Pos2::new(c.x + rx, c.y + r * 0.55),
                ],
                s,
            );
            let arc: Vec<Pos2> = (0..=12)
                .map(|i| {
                    let a = std::f32::consts::PI * i as f32 / 12.0;
                    Pos2::new(c.x + rx * a.cos(), c.y + r * 0.55 + r * 0.32 * a.sin())
                })
                .collect();
            p.add(egui::Shape::line(arc, s));
        }
        "runtime" => {
            // curly braces { } around a dot
            let (x, y) = (c.x, c.y);
            p.line_segment(
                [
                    Pos2::new(x - r * 0.5, y - r),
                    Pos2::new(x - r * 0.9, y - r * 0.5),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(x - r * 0.9, y - r * 0.5),
                    Pos2::new(x - r * 0.9, y + r * 0.5),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(x - r * 0.9, y + r * 0.5),
                    Pos2::new(x - r * 0.5, y + r),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(x + r * 0.5, y - r),
                    Pos2::new(x + r * 0.9, y - r * 0.5),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(x + r * 0.9, y - r * 0.5),
                    Pos2::new(x + r * 0.9, y + r * 0.5),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(x + r * 0.9, y + r * 0.5),
                    Pos2::new(x + r * 0.5, y + r),
                ],
                s,
            );
            p.circle_filled(c, r * 0.18, col);
        }
        "chat" => {
            // speech bubble with tail
            let b =
                Rect::from_center_size(c + Vec2::new(0.0, -r * 0.15), Vec2::new(r * 1.9, r * 1.35));
            p.rect_stroke(
                b,
                4.0,
                Stroke::new(1.4_f32, col),
                egui::epaint::StrokeKind::Inside,
            );
            p.line_segment(
                [
                    Pos2::new(b.left() + r * 0.5, b.bottom()),
                    Pos2::new(b.left() + r * 0.2, b.bottom() + r * 0.45),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(b.left() + r * 0.2, b.bottom() + r * 0.45),
                    Pos2::new(b.left() + r * 0.95, b.bottom()),
                ],
                s,
            );
        }
        "music" => {
            // eighth note: head + stem + flag
            p.add(egui::Shape::ellipse_stroke(
                Pos2::new(c.x - r * 0.35, c.y + r * 0.55),
                Vec2::new(r * 0.42, r * 0.32),
                Stroke::new(1.3_f32, col),
            ));
            p.line_segment(
                [
                    Pos2::new(c.x + r * 0.05, c.y + r * 0.5),
                    Pos2::new(c.x + r * 0.05, c.y - r),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(c.x + r * 0.05, c.y - r),
                    Pos2::new(c.x + r * 0.75, c.y - r * 0.55),
                ],
                s,
            );
        }
        "media" => {
            // play triangle inside a frame: image/video tools
            let b = Rect::from_center_size(c, Vec2::new(r * 1.9, r * 1.6));
            p.rect_stroke(
                b,
                3.0,
                Stroke::new(1.3_f32, col),
                egui::epaint::StrokeKind::Inside,
            );
            let t = b.center();
            p.add(egui::Shape::closed_line(
                vec![
                    Pos2::new(t.x - r * 0.3, t.y - r * 0.45),
                    Pos2::new(t.x + r * 0.5, t.y),
                    Pos2::new(t.x - r * 0.3, t.y + r * 0.45),
                ],
                Stroke::new(1.3_f32, col),
            ));
        }
        "files" => {
            // folder: tab + body
            let b =
                Rect::from_center_size(c + Vec2::new(0.0, r * 0.18), Vec2::new(r * 1.9, r * 1.25));
            p.rect_stroke(
                b,
                3.0,
                Stroke::new(1.4_f32, col),
                egui::epaint::StrokeKind::Inside,
            );
            p.line_segment(
                [
                    Pos2::new(b.left(), b.top()),
                    Pos2::new(b.left() + b.width() * 0.45, b.top()),
                ],
                Stroke::new(1.6_f32, col),
            );
            p.line_segment(
                [
                    Pos2::new(b.left() + b.width() * 0.45, b.top()),
                    Pos2::new(b.left() + b.width() * 0.6, b.top() - r * 0.3),
                ],
                s,
            );
        }
        "game" => {
            // gamepad: body + dpad + two buttons
            let b = Rect::from_center_size(c, Vec2::new(r * 2.0, r * 1.25));
            p.rect_stroke(
                b,
                r * 0.5,
                Stroke::new(1.4_f32, col),
                egui::epaint::StrokeKind::Inside,
            );
            p.line_segment(
                [
                    Pos2::new(b.left() + r * 0.45, b.center().y),
                    Pos2::new(b.left() + r * 1.05, b.center().y),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(b.left() + r * 0.75, b.center().y - r * 0.3),
                    Pos2::new(b.left() + r * 0.75, b.center().y + r * 0.3),
                ],
                s,
            );
            p.circle_filled(
                Pos2::new(b.right() - r * 0.7, b.center().y - r * 0.18),
                r * 0.14,
                col,
            );
            p.circle_filled(
                Pos2::new(b.right() - r * 0.45, b.center().y + r * 0.18),
                r * 0.14,
                col,
            );
        }
        "agent" => {
            // robot head: box, two eyes, antenna
            let b = Rect::from_center_size(c, Vec2::new(r * 1.7, r * 1.5));
            p.rect_stroke(
                b,
                3.0,
                Stroke::new(1.4_f32, col),
                egui::epaint::StrokeKind::Inside,
            );
            p.circle_filled(
                Pos2::new(b.center().x - r * 0.35, b.center().y),
                r * 0.16,
                col,
            );
            p.circle_filled(
                Pos2::new(b.center().x + r * 0.35, b.center().y),
                r * 0.16,
                col,
            );
            p.line_segment(
                [
                    Pos2::new(b.center().x, b.top()),
                    Pos2::new(b.center().x, b.top() - r * 0.35),
                ],
                s,
            );
            p.circle_filled(Pos2::new(b.center().x, b.top() - r * 0.42), r * 0.12, col);
        }
        "audio" => {
            // speaker cone + two waves
            let x = c.x - r * 0.7;
            p.line_segment(
                [
                    Pos2::new(x, c.y - r * 0.3),
                    Pos2::new(x + r * 0.45, c.y - r * 0.3),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(x + r * 0.45, c.y - r * 0.3),
                    Pos2::new(x + r * 0.9, c.y - r * 0.75),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(x + r * 0.9, c.y - r * 0.75),
                    Pos2::new(x + r * 0.9, c.y + r * 0.75),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(x + r * 0.9, c.y + r * 0.75),
                    Pos2::new(x + r * 0.45, c.y + r * 0.3),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(x + r * 0.45, c.y + r * 0.3),
                    Pos2::new(x, c.y + r * 0.3),
                ],
                s,
            );
            for f in [0.55, 0.9] {
                let arc: Vec<Pos2> = (0..=8)
                    .map(|i| {
                        let a = -std::f32::consts::FRAC_PI_3
                            + std::f32::consts::FRAC_PI_2 * 2.0 / 3.0 * i as f32 / 8.0;
                        Pos2::new(c.x - r * 0.1 + a.cos() * r * f, c.y + a.sin() * r * f)
                    })
                    .collect();
                p.add(egui::Shape::line(arc, Stroke::new(1.1_f32, col)));
            }
        }
        "desktop" => {
            // monitor + stand
            let b =
                Rect::from_center_size(c + Vec2::new(0.0, -r * 0.2), Vec2::new(r * 2.0, r * 1.3));
            p.rect_stroke(
                b,
                3.0,
                Stroke::new(1.4_f32, col),
                egui::epaint::StrokeKind::Inside,
            );
            p.line_segment(
                [
                    Pos2::new(b.center().x, b.bottom()),
                    Pos2::new(b.center().x, c.y + r * 0.75),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(c.x - r * 0.5, c.y + r * 0.75),
                    Pos2::new(c.x + r * 0.5, c.y + r * 0.75),
                ],
                Stroke::new(1.6_f32, col),
            );
        }
        "settings" => {
            // sliders: three rails with knobs
            for (i, f) in [0.3, 0.55, 0.8].iter().enumerate() {
                let y = c.y - r + r * 2.0 * (i as f32 + 0.5) / 3.0;
                p.line_segment(
                    [Pos2::new(c.x - r, y), Pos2::new(c.x + r, y)],
                    Stroke::new(1.1_f32, col),
                );
                p.circle_filled(Pos2::new(c.x - r + 2.0 * r * f, y), r * 0.22, col);
            }
        }
        _ => {}
    }
}

/// Settings text field: inset well, roomy padding, monospace where it
/// matters, and an optional icon button docked inside the right edge.
/// `on_trail` receives the clicked state and returns the new one.
/// A labelled text input in an inset well; `trail` adds a trailing toggle
/// (label, state, callback), e.g. the password reveal.
type FieldTrail<'a> = (&'a str, bool, &'a mut dyn FnMut(bool));

fn field(
    ui: &mut Ui,
    value: &mut String,
    hint: &str,
    mono: bool,
    password: bool,
    trail: Option<FieldTrail<'_>>,
) -> egui::Response {
    // lay the inset well down first, then the TextEdit on top of it
    let w = ui.available_width();
    let well = Rect::from_min_size(ui.cursor().min, Vec2::new(w, 38.0));
    ui.allocate_rect(well, Sense::hover());
    ui.painter().rect_filled(well, 9.0, SURFACE_HI);
    let mut te = egui::TextEdit::singleline(value)
        .hint_text(RichText::new(hint).color(FAINT))
        .margin(Margin {
            left: 12,
            right: if trail.is_some() { 40 } else { 12 },
            top: 9,
            bottom: 9,
        })
        .interactive(true);
    if mono {
        te = te.font(FontId::monospace(13.0));
    }
    if password {
        te = te.password(true);
    }
    let resp = ui.put(well, te);
    let stroke = Stroke::new(
        1.0_f32,
        if resp.has_focus() {
            ACCENT
        } else if resp.hovered() {
            BORDER.gamma_multiply(1.9)
        } else {
            BORDER
        },
    );
    ui.painter()
        .rect_stroke(well, 9.0, stroke, egui::epaint::StrokeKind::Middle);
    if let Some((icon, state, on_trail)) = trail {
        let r = Rect::from_center_size(
            Pos2::new(resp.rect.right() - 20.0, resp.rect.center().y),
            Vec2::splat(24.0),
        );
        let hit = ui.interact(r, resp.id.with("trail"), Sense::click());
        if hit.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        let col = if hit.hovered() { TEXT } else { MUTED };
        icon_glyph(ui.painter(), icon, r.center(), 7.0, col);
        if hit.clicked() {
            on_trail(state);
        }
    }
    resp
}

/// Labelled settings row: caption + optional hint on the left, control on
/// the right, with a hairline separating rows.
fn setting_row<R>(ui: &mut Ui, label: &str, hint: &str, add: impl FnOnce(&mut Ui) -> R) -> R {
    let top = ui.cursor().top();
    let inner = ui
        .horizontal(|ui| {
            ui.vertical(|ui| {
                ui.spacing_mut().item_spacing.y = 1.0;
                ui.label(RichText::new(label).size(13.5).color(TEXT));
                if !hint.is_empty() {
                    ui.label(RichText::new(hint).size(11.5).color(FAINT));
                }
            });
            let r = ui
                .with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.spacing_mut().item_spacing.x = 10.0;
                    add(ui)
                })
                .inner;
            r
        })
        .inner;
    // hairline under the row, then breathing room
    let bottom = ui.cursor().top().max(top + 34.0);
    let left = ui.cursor().left();
    ui.painter().hline(
        left..=ui.max_rect().right() - 18.0,
        bottom + 7.0,
        Stroke::new(1.0_f32, BORDER.gamma_multiply(0.7)),
    );
    ui.add_space(14.0);
    inner
}

fn page_header(ui: &mut Ui, title: &str, subtitle: &str) {
    ui.spacing_mut().item_spacing.y = 2.0;
    ui.label(RichText::new(title).font(bold(24.0)).color(TEXT));
    if !subtitle.is_empty() {
        ui.label(RichText::new(subtitle).size(13.0).color(MUTED));
    }
}

/// Centered placeholder for empty lists and unselected detail pages.
fn empty_state(ui: &mut Ui, title: &str, body: &str) {
    ui.add_space((ui.available_height() * 0.3).min(160.0));
    ui.vertical_centered(|ui| {
        let (r, _) = ui.allocate_exact_size(Vec2::splat(44.0), Sense::hover());
        ui.painter().rect_filled(r, 12.0, SURFACE_HI);
        let s = s_pulse(Rect::from_center_size(r.center(), Vec2::splat(32.0)));
        ui.painter()
            .add(egui::Shape::line(s, Stroke::new(2.0_f32, FAINT)));
        ui.add_space(6.0);
        ui.label(RichText::new(title).font(bold(15.0)).color(TEXT));
        ui.label(RichText::new(body).size(13.0).color(MUTED));
    });
}

fn section_label(ui: &mut Ui, text: &str) {
    ui.horizontal(|ui| {
        ui.add_space(8.0);
        ui.label(RichText::new(text).size(10.5).color(FAINT));
    });
}

/// Bigger in-page section divider used on the dashboard, e.g. "Docker".
fn section_heading(ui: &mut Ui, text: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(text).font(bold(15.5)).color(TEXT));
        let (r, _) = ui.allocate_at_least(Vec2::new(ui.available_width(), 1.0), Sense::hover());
        let line = Rect::from_min_max(
            Pos2::new(r.left(), r.center().y),
            Pos2::new(r.right(), r.center().y + 1.0),
        );
        ui.painter().rect_filled(line, 0.0, BORDER);
    });
}

/// Compact uptime like "3d 4h", "4h 12m", "12m".
fn uptime_str(secs: i64) -> String {
    let s = secs.max(0);
    let d = s / 86_400;
    let h = s % 86_400 / 3600;
    let m = s % 3600 / 60;
    if d > 0 {
        format!("{d}d {h}h")
    } else if h > 0 {
        format!("{h}h {m}m")
    } else {
        format!("{m}m")
    }
}

/// Temperature with one decimal dropped: "72 °C" reads better than "72.4".
fn temp_str(c: f64) -> String {
    format!("{c:.0}")
}

/// Short OS label for the dashboard header: prefers the kernel name from
/// /proc (Linux + release) over the Docker-reported string.
fn engine_card(ui: &mut Ui, ok: bool, version: &str) {
    Frame::NONE
        .fill(SURFACE)
        .stroke(Stroke::new(1.0_f32, BORDER))
        .corner_radius(10)
        .inner_margin(Margin::symmetric(12, 9))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                let (r, _) = ui.allocate_exact_size(Vec2::splat(12.0), Sense::hover());
                let c = if ok { GREEN } else { RED };
                ui.painter()
                    .circle_filled(r.center(), 6.0, c.gamma_multiply(0.2));
                ui.painter().circle_filled(r.center(), 3.5, c);
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 0.0;
                    ui.label(
                        RichText::new(if ok {
                            "Engine running"
                        } else {
                            "Engine unreachable"
                        })
                        .size(12.5)
                        .color(TEXT),
                    );
                    if !version.is_empty() {
                        ui.label(
                            RichText::new(format!("Docker {version}"))
                                .size(11.0)
                                .color(FAINT),
                        );
                    }
                });
            });
        });
}

/// Sidebar navigation entry; true when clicked.
fn nav_item(
    ui: &mut Ui,
    tab: Tab,
    label: &str,
    badge: &str,
    active: bool,
    real_icon: Option<&egui::TextureHandle>,
) -> bool {
    let (rect, resp) =
        ui.allocate_exact_size(Vec2::new(ui.available_width(), 38.0), Sense::click());
    #[cfg(test)]
    NAV_ROW_RECTS.lock().unwrap().push((tab, rect));
    let hovered = resp.hovered();
    if hovered {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    let p = ui.painter();
    if active {
        p.rect_filled(rect, 9.0, ACCENT.gamma_multiply(0.15));
        p.rect_filled(
            Rect::from_center_size(
                Pos2::new(rect.left() + 1.5, rect.center().y),
                Vec2::new(3.0, 18.0),
            ),
            2.0,
            ACCENT,
        );
    } else if hovered {
        p.rect_filled(rect, 9.0, SURFACE_HI);
    }
    let fg = if active || hovered { TEXT } else { MUTED };
    let icon = Rect::from_center_size(
        Pos2::new(rect.left() + 22.0, rect.center().y),
        Vec2::splat(16.0),
    );
    // the Processes rail carries a real application icon once one has
    // decoded; every tab keeps its drawn line-icon as the fallback
    match real_icon {
        Some(tex) => {
            let uv = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1.0, 1.0));
            p.image(tex.id(), icon, uv, Color32::WHITE);
        }
        None => nav_icon(p, tab, icon, if active { ACCENT } else { fg }),
    }
    p.text(
        Pos2::new(rect.left() + 42.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        FontId::proportional(14.0),
        fg,
    );
    if !badge.is_empty() {
        let galley = p.layout_no_wrap(
            badge.to_string(),
            FontId::proportional(11.5),
            if active { TEXT } else { MUTED },
        );
        let size = Vec2::new(galley.size().x + 14.0, 20.0);
        let br = Rect::from_min_size(
            Pos2::new(rect.right() - 8.0 - size.x, rect.center().y - 10.0),
            size,
        );
        p.rect_filled(
            br,
            10.0,
            if active {
                ACCENT.gamma_multiply(0.25)
            } else {
                SURFACE_HI
            },
        );
        p.galley(br.center() - galley.size() / 2.0, galley, MUTED);
    }
    resp.clicked()
}

/// Line icons drawn with the painter, so they're crisp at any scale and
/// don't depend on an icon font being installed.
fn nav_icon(p: &egui::Painter, tab: Tab, r: Rect, c: Color32) {
    let s = Stroke::new(1.6_f32, c);
    let inside = egui::epaint::StrokeKind::Inside;
    match tab {
        Tab::Dashboard => {
            let h = (r.width() - 3.0) / 2.0;
            for (dx, dy) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
                let min = r.min + Vec2::new(dx * (h + 3.0), dy * (h + 3.0));
                p.rect_stroke(Rect::from_min_size(min, Vec2::splat(h)), 2.0, s, inside);
            }
        }
        Tab::Containers => {
            let b = r.shrink2(Vec2::new(0.0, 2.0));
            p.rect_stroke(b, 2.5, s, inside);
            for f in [0.36, 0.64] {
                let x = b.left() + b.width() * f;
                p.line_segment(
                    [Pos2::new(x, b.top() + 3.5), Pos2::new(x, b.bottom() - 3.5)],
                    s,
                );
            }
        }
        Tab::Images => {
            let base = Rect::from_min_max(Pos2::new(r.left(), r.bottom() - 7.0), r.max);
            p.rect_stroke(base, 2.0, s, inside);
            let y1 = base.top() - 3.0;
            p.line_segment(
                [
                    Pos2::new(r.left() + 2.5, y1),
                    Pos2::new(r.right() - 2.5, y1),
                ],
                s,
            );
            let y2 = y1 - 3.5;
            p.line_segment(
                [
                    Pos2::new(r.left() + 5.0, y2),
                    Pos2::new(r.right() - 5.0, y2),
                ],
                s,
            );
        }
        Tab::Volumes => {
            let rx = r.width() / 2.0 - 0.5;
            let top = Pos2::new(r.center().x, r.top() + 3.0);
            let bot = Pos2::new(r.center().x, r.bottom() - 3.0);
            p.add(egui::Shape::ellipse_stroke(top, Vec2::new(rx, 2.6), s));
            p.line_segment(
                [
                    Pos2::new(r.left() + 0.5, top.y),
                    Pos2::new(r.left() + 0.5, bot.y),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(r.right() - 0.5, top.y),
                    Pos2::new(r.right() - 0.5, bot.y),
                ],
                s,
            );
            let arc: Vec<Pos2> = (0..=16)
                .map(|i| {
                    let a = std::f32::consts::PI * i as f32 / 16.0;
                    Pos2::new(bot.x + rx * a.cos(), bot.y + 2.6 * a.sin())
                })
                .collect();
            p.add(egui::Shape::line(arc, s));
        }
        Tab::Agents => {
            // a small robot head: rounded box, two eyes, antenna
            let b = r.shrink2(Vec2::new(1.5, 2.5));
            p.rect_stroke(b, 4.0, s, inside);
            for dx in [-0.22_f32, 0.22] {
                let eye = Pos2::new(b.center().x + b.width() * dx, b.center().y - 1.0);
                p.circle_filled(eye, 1.4, c);
            }
            p.line_segment(
                [
                    Pos2::new(b.center().x, b.top()),
                    Pos2::new(b.center().x, r.top() + 1.5),
                ],
                s,
            );
            p.circle_filled(Pos2::new(b.center().x, r.top() + 1.5), 1.2, c);
        }
        Tab::Logs => {
            for (i, f) in [1.0, 0.65, 0.85, 0.5].iter().enumerate() {
                let y = r.top() + 2.0 + i as f32 * 4.0;
                p.line_segment(
                    [
                        Pos2::new(r.left(), y),
                        Pos2::new(r.left() + r.width() * f, y),
                    ],
                    s,
                );
            }
        }
        Tab::Hardware => {
            // a little CPU chip: square, pins on every side
            let b = r.shrink2(Vec2::new(2.5, 2.5));
            p.rect_stroke(b, 2.5, s, inside);
            let pin = 2.2_f32;
            for i in 1..=3 {
                let t = b.left() + b.width() * i as f32 / 4.0;
                p.line_segment([Pos2::new(t, b.top()), Pos2::new(t, b.top() - pin)], s);
                p.line_segment(
                    [Pos2::new(t, b.bottom()), Pos2::new(t, b.bottom() + pin)],
                    s,
                );
                let u = b.top() + b.height() * i as f32 / 4.0;
                p.line_segment([Pos2::new(b.left(), u), Pos2::new(b.left() - pin, u)], s);
                p.line_segment([Pos2::new(b.right(), u), Pos2::new(b.right() + pin, u)], s);
            }
        }
        Tab::Ports => {
            // a network jack / plug: rectangle body, prongs on top
            let b = Rect::from_min_max(
                Pos2::new(r.left() + 1.5, r.top() + 5.0),
                Pos2::new(r.right() - 1.5, r.bottom() - 1.0),
            );
            p.rect_stroke(b, 2.5, s, inside);
            for f in [0.3, 0.7] {
                let x = b.left() + b.width() * f;
                p.line_segment([Pos2::new(x, b.top()), Pos2::new(x, r.top() + 1.0)], s);
            }
            p.line_segment(
                [
                    Pos2::new(b.center().x, b.bottom()),
                    Pos2::new(b.center().x, r.bottom() + 1.5),
                ],
                s,
            );
        }
        Tab::Processes => {
            // stacked window rows with a cursor: a process list mark
            for (i, f) in [0.9, 0.6, 0.75].iter().enumerate() {
                let y = r.top() + 2.5 + i as f32 * 4.5;
                p.line_segment(
                    [
                        Pos2::new(r.left(), y),
                        Pos2::new(r.left() + r.width() * f, y),
                    ],
                    s,
                );
            }
            p.line_segment(
                [
                    Pos2::new(r.left(), r.bottom() - 1.5),
                    Pos2::new(r.left() + r.width() * 0.45, r.bottom() - 1.5),
                ],
                s,
            );
        }
        Tab::Settings => {
            // a gear: ring, hub, six teeth
            let ring = r.shrink2(Vec2::new(3.5, 3.5));
            p.circle_stroke(
                ring.center(),
                ring.width().min(ring.height()) / 2.0 - 1.0,
                s,
            );
            p.circle_filled(ring.center(), 1.7, c);
            for k in 0..6 {
                let a = std::f32::consts::TAU * k as f32 / 6.0;
                let (sin, cos) = a.sin_cos();
                let rad = ring.width().min(ring.height()) / 2.0 - 1.0;
                let p1 = ring.center() + Vec2::new(cos, sin) * (rad - 0.5);
                let p2 = ring.center() + Vec2::new(cos, sin) * (rad + 2.2);
                p.line_segment([p1, p2], Stroke::new(2.4_f32, c));
            }
        }
        Tab::Strata => {
            // a chip mark: outline square, filled core, pins like the brand icon
            let b = r.shrink2(Vec2::new(3.0, 3.0));
            p.rect_stroke(b, 3.5, s, inside);
            let core = b.shrink2(Vec2::new(b.width() * 0.28, b.height() * 0.28));
            p.rect_filled(core, 2.0, c);
            let pin = 2.0_f32;
            p.line_segment(
                [
                    Pos2::new(b.center().x, b.top()),
                    Pos2::new(b.center().x, b.top() - pin),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(b.center().x, b.bottom()),
                    Pos2::new(b.center().x, b.bottom() + pin),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(b.left(), b.center().y),
                    Pos2::new(b.left() - pin, b.center().y),
                ],
                s,
            );
            p.line_segment(
                [
                    Pos2::new(b.right(), b.center().y),
                    Pos2::new(b.right() + pin, b.center().y),
                ],
                s,
            );
        }
    }
}

/// Rounded icon tile at the left edge of a list row.
fn icon_tile(p: &egui::Painter, row: Rect, tab: Tab, color: Color32) {
    let tile = Rect::from_center_size(
        Pos2::new(row.left() + 26.0, row.center().y),
        Vec2::splat(34.0),
    );
    p.rect_filled(tile, 9.0, color.gamma_multiply(0.12));
    nav_icon(
        p,
        tab,
        Rect::from_center_size(tile.center(), Vec2::splat(16.0)),
        color,
    );
}

fn row_bg(p: &egui::Painter, rect: Rect, selected: bool, hovered: bool) {
    if selected {
        p.rect(
            rect,
            10.0,
            ACCENT.gamma_multiply(0.12),
            Stroke::new(1.0_f32, ACCENT.gamma_multiply(0.45)),
            egui::epaint::StrokeKind::Inside,
        );
    } else if hovered {
        p.rect_filled(rect, 10.0, SURFACE);
    }
}

/// Running dots glow (eased in by `pulse`); stopped ones are hollow rings.
fn status_dot(p: &egui::Painter, c: Pos2, running: bool, pulse: f32) {
    if running {
        p.circle_filled(c, 4.0 + 3.5 * pulse, GREEN.gamma_multiply(0.18));
        p.circle_filled(c, 4.0, GREEN);
    } else {
        p.circle_stroke(c, 3.5, Stroke::new(1.5_f32, FAINT));
    }
}

fn spinner_at(p: &egui::Painter, time: f64, c: Pos2, r: f32, color: Color32) {
    let start = time * 6.0;
    let pts: Vec<Pos2> = (0..=20)
        .map(|i| {
            let a = start + i as f64 / 20.0 * 4.4;
            c + r * Vec2::new(a.cos() as f32, a.sin() as f32)
        })
        .collect();
    p.add(egui::Shape::line(pts, Stroke::new(2.0_f32, color)));
}

/// Single-line text clipped to `max_w` with an ellipsis.
fn elide(
    p: &egui::Painter,
    text: String,
    font: FontId,
    color: Color32,
    max_w: f32,
) -> Arc<egui::Galley> {
    let mut job = egui::text::LayoutJob::simple_singleline(text, font, color);
    job.wrap = egui::text::TextWrapping {
        max_width: max_w.max(1.0),
        max_rows: 1,
        break_anywhere: true,
        overflow_character: Some('…'),
    };
    p.layout_job(job)
}

/// Right-anchored pill label painted at `right_center`.
fn pill_at(p: &egui::Painter, right_center: Pos2, text: &str, fg: Color32, bg: Color32) {
    let galley = p.layout_no_wrap(text.to_string(), FontId::proportional(11.5), fg);
    let size = galley.size() + Vec2::new(16.0, 6.0);
    let r = Rect::from_min_size(
        Pos2::new(right_center.x - size.x, right_center.y - size.y / 2.0),
        size,
    );
    p.rect_filled(r, size.y / 2.0, bg);
    p.galley(r.center() - galley.size() / 2.0, galley, fg);
}

/// Small rounded label (project, driver, usage).
fn chip(ui: &mut Ui, text: &str, color: Color32) {
    Frame::NONE
        .fill(color.gamma_multiply(0.14))
        .corner_radius(10)
        .inner_margin(Margin::symmetric(9, 3))
        .show(ui, |ui| {
            ui.label(RichText::new(text).size(11.5).color(color));
        });
}

fn state_badge(ui: &mut Ui, state: &str) {
    let color = match state {
        "running" => GREEN,
        "paused" | "restarting" => AMBER,
        "dead" => RED,
        _ => FAINT,
    };
    chip(ui, &capitalize(state), color);
}

/// "● Live" indicator, painted so it sizes to its text in any layout.
fn live_pill(ui: &mut Ui, ok: bool) {
    let color = if ok { GREEN } else { RED };
    let text = if ok { "Live" } else { "Offline" };
    let galley = ui
        .painter()
        .layout_no_wrap(text.to_string(), FontId::proportional(11.5), color);
    let size = Vec2::new(galley.size().x + 30.0, 22.0);
    let (r, _) = ui.allocate_exact_size(size, Sense::hover());
    let p = ui.painter();
    p.rect_filled(r, 11.0, color.gamma_multiply(0.12));
    p.circle_filled(Pos2::new(r.left() + 12.0, r.center().y), 3.0, color);
    p.galley(
        Pos2::new(r.left() + 21.0, r.center().y - galley.size().y / 2.0),
        galley,
        color,
    );
}

#[derive(Clone, Copy)]
enum Btn {
    Primary,
    Secondary,
    Danger,
}

fn button(ui: &mut Ui, label: &str, kind: Btn, enabled: bool) -> egui::Response {
    let (fill, fg, stroke) = match kind {
        Btn::Primary => (ACCENT, Color32::WHITE, ACCENT),
        Btn::Secondary => (SURFACE_HI, TEXT, BORDER),
        Btn::Danger => (RED.gamma_multiply(0.12), RED, RED.gamma_multiply(0.4)),
    };
    let btn = egui::Button::new(RichText::new(label).size(13.5).color(fg))
        .fill(fill)
        .stroke(Stroke::new(1.0_f32, stroke))
        .corner_radius(8)
        .min_size(Vec2::new(0.0, 34.0));
    let resp = ui.add_enabled(enabled, btn);
    if resp.hovered() && enabled {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    resp
}

/// Full-width low-emphasis button (rail footer).
fn ghost_button(ui: &mut Ui, label: &str, width: f32) -> egui::Response {
    ui.add(
        egui::Button::new(RichText::new(label).size(13.0).color(MUTED))
            .fill(SURFACE)
            .stroke(Stroke::new(1.0_f32, BORDER))
            .corner_radius(8)
            .min_size(Vec2::new(width, 32.0)),
    )
}

/// iOS-style on/off switch: 40×22 track with a sliding knob.
fn switch(ui: &mut Ui, on: bool) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(40.0, 22.0), Sense::click());
    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    let p = ui.painter();
    let track = if on { ACCENT } else { SURFACE_HI };
    let stroke = if on { ACCENT } else { BORDER };
    p.rect_filled(rect, 11.0, track);
    p.rect_stroke(
        rect,
        11.0,
        Stroke::new(1.0_f32, stroke),
        egui::epaint::StrokeKind::Inside,
    );
    let knob = 16.0_f32;
    let cx = if on {
        rect.right() - knob / 2.0 - 3.0
    } else {
        rect.left() + knob / 2.0 + 3.0
    };
    p.circle_filled(
        Pos2::new(cx, rect.center().y),
        knob / 2.0,
        if on { Color32::WHITE } else { MUTED },
    );
    resp
}

fn toggle_chip(ui: &mut Ui, label: &str, on: bool) -> egui::Response {
    let (fill, fg, stroke) = if on {
        (
            ACCENT.gamma_multiply(0.16),
            ACCENT,
            ACCENT.gamma_multiply(0.5),
        )
    } else {
        (SURFACE_HI, MUTED, BORDER)
    };
    ui.add(
        egui::Button::new(RichText::new(label).size(12.5).color(fg))
            .fill(fill)
            .stroke(Stroke::new(1.0_f32, stroke))
            .corner_radius(8),
    )
}

/// Search field with a painted magnifier and a clear button.
fn search_box(ui: &mut Ui, text: &mut String, hint: &str) {
    let resp = ui.add(
        egui::TextEdit::singleline(text)
            .hint_text(RichText::new(hint).color(FAINT))
            .margin(Margin {
                left: 32,
                right: 30,
                top: 8,
                bottom: 8,
            })
            .desired_width(f32::INFINITY),
    );
    let c = Pos2::new(resp.rect.left() + 16.0, resp.rect.center().y - 1.0);
    let s = Stroke::new(1.5_f32, FAINT);
    ui.painter().circle_stroke(c, 4.5, s);
    ui.painter()
        .line_segment([c + Vec2::new(3.3, 3.3), c + Vec2::new(6.8, 6.8)], s);
    if !text.is_empty() {
        let r = Rect::from_center_size(
            Pos2::new(resp.rect.right() - 16.0, resp.rect.center().y),
            Vec2::splat(18.0),
        );
        let clear = ui.interact(r, resp.id.with("clear"), Sense::click());
        let col = if clear.hovered() { TEXT } else { FAINT };
        let d = 3.5;
        ui.painter().line_segment(
            [r.center() - Vec2::splat(d), r.center() + Vec2::splat(d)],
            Stroke::new(1.5_f32, col),
        );
        ui.painter().line_segment(
            [r.center() + Vec2::new(-d, d), r.center() + Vec2::new(d, -d)],
            Stroke::new(1.5_f32, col),
        );
        if clear.clicked() {
            text.clear();
        }
    }
}

/// Segmented control; returns the newly picked value.
fn segmented<T: PartialEq + Copy>(ui: &mut Ui, opts: &[(T, String)], cur: T) -> Option<T> {
    let h = 32.0;
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), h), Sense::hover());
    ui.painter().rect(
        rect,
        9.0,
        SURFACE,
        Stroke::new(1.0_f32, BORDER),
        egui::epaint::StrokeKind::Inside,
    );
    let w = (rect.width() - 6.0) / opts.len() as f32;
    let mut out = None;
    for (i, (v, label)) in opts.iter().enumerate() {
        let r = Rect::from_min_size(
            rect.min + Vec2::new(3.0 + i as f32 * w, 3.0),
            Vec2::new(w, h - 6.0),
        );
        let resp = ui.interact(r, ui.id().with(("seg", i)), Sense::click());
        let active = *v == cur;
        if active {
            ui.painter().rect_filled(r, 7.0, ELEVATED);
        } else if resp.hovered() {
            ui.painter().rect_filled(r, 7.0, SURFACE_HI);
        }
        ui.painter().text(
            r.center(),
            egui::Align2::CENTER_CENTER,
            label,
            FontId::proportional(12.5),
            if active { TEXT } else { MUTED },
        );
        if resp.clicked() {
            out = Some(*v);
        }
    }
    out
}

/// Label above value, for detail grids.
fn kv(ui: &mut Ui, k: &str, v: &str, mono: bool) {
    ui.spacing_mut().item_spacing.y = 2.0;
    ui.label(RichText::new(k).size(11.5).color(FAINT));
    let t = RichText::new(v).color(TEXT);
    let t = if mono {
        t.font(FontId::monospace(12.5))
    } else {
        t.size(13.5)
    };
    ui.add(egui::Label::new(t).truncate());
    ui.add_space(8.0);
}

fn progress(p: &egui::Painter, bar: Rect, frac: f32, color: Color32) {
    let r = bar.height() / 2.0;
    p.rect_filled(bar, r, SURFACE_HI);
    let frac = frac.clamp(0.0, 1.0);
    if frac > 0.0 {
        let w = (bar.width() * frac).max(bar.height());
        p.rect_filled(
            Rect::from_min_size(bar.min, Vec2::new(w, bar.height())),
            r,
            color,
        );
    }
}

/// Tiny inline trend line.
fn sparkline(
    p: &egui::Painter,
    rect: Rect,
    series: impl Iterator<Item = f64>,
    color: Color32,
    floor: f64,
) {
    let s: Vec<f64> = series.collect();
    if s.len() < 2 {
        return;
    }
    let pts = area_points(rect, &s, series_max(&s, floor));
    fill_under(p, &pts, rect.bottom(), color, 45);
    p.add(egui::Shape::line(pts, Stroke::new(1.3_f32, color)));
}

fn series_max(s: &[f64], floor: f64) -> f64 {
    s.iter().cloned().fold(0.0_f64, f64::max).max(floor)
}

fn area_points(rect: Rect, s: &[f64], max: f64) -> Vec<Pos2> {
    let n = s.len().max(2);
    s.iter()
        .enumerate()
        .map(|(i, v)| {
            let x = rect.left() + rect.width() * (i as f32 / (n - 1) as f32);
            let y = rect.bottom() - rect.height() * (v / max).clamp(0.0, 1.0) as f32;
            Pos2::new(x, y)
        })
        .collect()
}

/// Vertical gradient under a line: `alpha` at the line fading to clear.
fn fill_under(p: &egui::Painter, pts: &[Pos2], bottom: f32, color: Color32, alpha: u8) {
    if pts.len() < 2 {
        return;
    }
    let top = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), alpha);
    let mut mesh = egui::epaint::Mesh::default();
    for q in pts {
        mesh.colored_vertex(*q, top);
        mesh.colored_vertex(Pos2::new(q.x, bottom), Color32::TRANSPARENT);
    }
    for i in 0..pts.len() as u32 - 1 {
        let a = i * 2;
        mesh.add_triangle(a, a + 1, a + 2);
        mesh.add_triangle(a + 1, a + 3, a + 2);
    }
    p.add(mesh);
}

/// Area chart with gridlines, a scale label and a hover readout.
fn area_chart(ui: &mut Ui, series: &[f64], color: Color32, size: Vec2, fmt: fn(f64) -> String) {
    // The scale label gets its own strip above the plot.
    let (outer, resp) = ui.allocate_exact_size(size, Sense::hover());
    let rect = Rect::from_min_max(outer.min + Vec2::new(0.0, 16.0), outer.max);
    let p = ui.painter();
    for g in 0..4 {
        let y = rect.top() + rect.height() * g as f32 / 4.0;
        p.line_segment(
            [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
            Stroke::new(1.0_f32, Color32::from_rgb(28, 33, 43)),
        );
    }
    p.line_segment(
        [rect.left_bottom(), rect.right_bottom()],
        Stroke::new(1.0_f32, BORDER),
    );
    if series.len() < 2 {
        return;
    }
    let max = series_max(series, 1e-9) * 1.15;
    let pts = area_points(rect, series, max);
    fill_under(p, &pts, rect.bottom(), color, 80);
    p.add(egui::Shape::line(pts.clone(), Stroke::new(1.8_f32, color)));
    p.text(
        rect.right_top() - Vec2::new(0.0, 3.0),
        egui::Align2::RIGHT_BOTTOM,
        fmt(max),
        FontId::proportional(10.5),
        FAINT,
    );
    if let Some(pos) = resp.hover_pos() {
        let i = (((pos.x - rect.left()) / rect.width()) * (pts.len() - 1) as f32)
            .round()
            .clamp(0.0, (pts.len() - 1) as f32) as usize;
        let q = pts[i];
        p.line_segment(
            [Pos2::new(q.x, rect.top()), Pos2::new(q.x, rect.bottom())],
            Stroke::new(1.0_f32, BORDER),
        );
        p.circle_filled(q, 4.0, color);
        p.circle_stroke(q, 4.0, Stroke::new(1.5_f32, SURFACE));
        let label = fmt(series[i]);
        let ago_s = pts.len() - 1 - i;
        let text = if ago_s == 0 {
            label
        } else {
            format!("{label} · {ago_s}s ago")
        };
        let galley = p.layout_no_wrap(text, FontId::proportional(11.5), TEXT);
        let mut tip = Rect::from_min_size(
            q + Vec2::new(8.0, -26.0),
            galley.size() + Vec2::new(12.0, 6.0),
        );
        if tip.right() > rect.right() {
            tip = tip.translate(Vec2::new(-tip.width() - 16.0, 0.0));
        }
        if tip.top() < rect.top() {
            tip = tip.translate(Vec2::new(0.0, rect.top() - tip.top()));
        }
        p.rect_filled(tip, 6.0, ELEVATED);
        p.galley(tip.min + Vec2::new(6.0, 3.0), galley, TEXT);
    }
}

/// One log row: dimmed time, colored source tag (consolidated view), body.
fn log_line_job(
    l: &LogLine,
    source: Option<&str>,
    font: &FontId,
    alpha: f32,
) -> egui::text::LayoutJob {
    let fmt = |color: Color32| egui::TextFormat::simple(font.clone(), color.linear_multiply(alpha));
    let mut job = egui::text::LayoutJob::default();
    let mut body = l.text.as_str();
    // Timestamped tails lead with RFC3339; show just the clock part.
    if let Some((ts, rest)) = body.split_once(' ') {
        if ts.len() >= 19 && ts.as_bytes()[4] == b'-' && ts.as_bytes()[10] == b'T' {
            job.append(&ts[11..19], 0.0, fmt(FAINT));
            job.append("  ", 0.0, fmt(FAINT));
            body = rest;
        }
    }
    if let Some(src) = source {
        job.append(src, 0.0, fmt(source_color(src)));
        job.append("  ", 0.0, fmt(FAINT));
    }
    let color = match l.stream {
        StreamKind::Stderr => Color32::from_rgb(240, 178, 150),
        StreamKind::Stdout => Color32::from_rgb(206, 212, 222),
    };
    job.append(body, 0.0, fmt(color));
    job
}

/// Stable per-container tag color in the consolidated log view.
fn source_color(name: &str) -> Color32 {
    const PALETTE: [Color32; 6] = [
        ACCENT,
        GREEN,
        VIOLET,
        AMBER,
        CYAN,
        Color32::from_rgb(244, 140, 190),
    ];
    let h = name
        .bytes()
        .fold(5381u32, |h, b| h.wrapping_mul(33) ^ b as u32);
    PALETTE[h as usize % PALETTE.len()]
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().chain(c).collect(),
        None => String::new(),
    }
}

/// "12s ago" / "4m ago" / "3h ago" / "2d ago".
fn ago(secs: i64) -> String {
    let s = secs.max(0);
    match s {
        0..=4 => "just now".to_string(),
        5..=59 => format!("{s}s ago"),
        60..=3599 => format!("{}m ago", s / 60),
        3600..=86_399 => format!("{}h ago", s / 3600),
        _ => format!("{}d ago", s / 86_400),
    }
}

/// Brand mark: dark tile, gradient "S" with a heartbeat spine, live LED.
/// Mirrors assets/icon-small.svg (the dial and grid vanish at rail size).
fn logo_mark(p: &egui::Painter, r: Rect) {
    let k = r.width() / 256.0;
    let tile = r.shrink(8.0 * k);
    p.rect_filled(tile, 56.0 * k, Color32::from_rgb(22, 29, 44));
    p.rect_stroke(
        tile,
        56.0 * k,
        Stroke::new(1.0_f32, Color32::from_white_alpha(22)),
        egui::epaint::StrokeKind::Inside,
    );
    // cyan → blue → violet along the same diagonal as the SVG gradient
    let (a, d) = (
        r.min + Vec2::new(60.0, 52.0) * k,
        Vec2::new(136.0, 152.0) * k,
    );
    let ink = move |_: Rect, q: Pos2| {
        let t = ((q - a).dot(d) / d.length_sq()).clamp(0.0, 1.0);
        if t < 0.55 {
            CYAN.lerp_to_gamma(ACCENT, t / 0.55)
        } else {
            ACCENT.lerp_to_gamma(VIOLET, (t - 0.55) / 0.45)
        }
    };
    let pts = s_pulse(r);
    let w = 30.0 * k;
    // paths are butt-capped; round the two ends by hand
    for end in [pts[0], pts[pts.len() - 1]] {
        p.circle_filled(end, w / 2.0, ink(r, end));
    }
    p.add(egui::Shape::line(
        pts,
        egui::epaint::PathStroke::new_uv(w, ink),
    ));
    p.circle_filled(r.min + Vec2::new(206.0, 56.0) * k, 15.0 * k, GREEN);
}

/// The brand "S": top bar, left bowl, heartbeat spine, right bowl, bottom
/// bar — the same path as assets/icon.svg, scaled from its 256-unit box.
fn s_pulse(r: Rect) -> Vec<Pos2> {
    let bowl = |cx: f32, cy: f32, from: f32, to: f32| {
        (1..12).map(move |i| {
            let a = (from + (to - from) * i as f32 / 12.0).to_radians();
            (cx + 32.0 * a.cos(), cy + 32.0 * a.sin())
        })
    };
    let mut pts = vec![(172.0, 64.0), (100.0, 64.0)];
    pts.extend(bowl(100.0, 96.0, 270.0, 90.0));
    pts.extend([
        (100.0, 128.0),
        (110.0, 128.0),
        (120.0, 98.0),
        (134.0, 158.0),
        (144.0, 128.0),
        (156.0, 128.0),
    ]);
    pts.extend(bowl(156.0, 160.0, -90.0, 90.0));
    pts.extend([(156.0, 192.0), (84.0, 192.0)]);
    let k = r.width() / 256.0;
    pts.into_iter()
        .map(|(x, y)| r.min + Vec2::new(x, y) * k)
        .collect()
}

/// Last path component of a filesystem path, for project labels.
fn basename(p: &str) -> String {
    p.trim_end_matches('/')
        .rsplit('/')
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or(p)
        .to_string()
}

/// How an agent session reads at a glance.
struct AgentState {
    label: &'static str,
    color: Color32,
    /// actively doing something (spinner, repaint)
    live: bool,
    /// list order: 0 needs you, 1 working, 2 idle, 3 unknown
    rank: u8,
}

/// Claude Code reports its own status; other agents only have CPU to go on.
fn agent_state(a: &AgentSession) -> AgentState {
    let st = |label, color, live, rank| AgentState {
        label,
        color,
        live,
        rank,
    };
    match a.activity.as_ref().map(|x| x.status.as_str()) {
        Some("waiting") => st("Needs you", AMBER, false, 0),
        Some("busy") => st("Working", GREEN, true, 1),
        Some("shell") => st("Running shell", GREEN, true, 1),
        Some("idle") => st("Idle", FAINT, false, 2),
        Some(_) => st("Running", ACCENT, false, 3),
        None if a.cpu_pct >= 2.0 => st("Active", GREEN, false, 1),
        None => st("Quiet", FAINT, false, 3),
    }
}

/// Session title, else the project it runs in, else the agent's name.
fn agent_title(a: &AgentSession) -> String {
    a.activity
        .as_ref()
        .and_then(|x| x.title.clone())
        .or_else(|| {
            a.cwd
                .as_deref()
                .map(|c| format!("{} · {}", a.agent, basename(c)))
        })
        .unwrap_or_else(|| a.agent.to_string())
}

/// "Claude Code · project · up 2h 10m" for the list's second line.
fn agent_where(a: &AgentSession) -> String {
    let up = epoch()
        - a.started
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
    let mut parts = vec![a.agent.to_string()];
    if let Some(c) = &a.cwd {
        parts.push(basename(c));
    }
    parts.push(format!("up {}", uptime_str(up)));
    parts.join(" · ")
}

/// One step as a single line, e.g. "Bash: Run tests".
fn step_line(s: &Step) -> String {
    match s.kind {
        StepKind::Prompt => format!("You: {}", s.text),
        StepKind::Reply => s.text.clone(),
        StepKind::Tool => format!("{}: {}", s.tool, s.text),
    }
}

/// Where the session runs: project, command line, process ids.
fn agent_process_card(ui: &mut Ui, a: &AgentSession, now: i64) {
    let up = now
        - a.started
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
    card(ui, |ui| {
        card_title(ui, "Process", &format!("up {}", uptime_str(up)));
        kv(
            ui,
            "Project",
            a.cwd.as_deref().unwrap_or("not readable (different user?)"),
            true,
        );
        kv(
            ui,
            "PIDs",
            &a.pids
                .iter()
                .map(|p| p.to_string())
                .collect::<Vec<_>>()
                .join(", "),
            true,
        );
        ui.add_space(4.0);
        ui.add(
            egui::Label::new(
                RichText::new(&a.cmdline)
                    .font(FontId::monospace(11.5))
                    .color(FAINT),
            )
            .wrap(),
        );
    });
}

/// Stable per-agent accent color for the Agents tab.
fn agent_color(agent: &str) -> Color32 {
    match agent {
        "Claude Code" => VIOLET,
        "Codex CLI" => Color32::from_rgb(0, 197, 110),
        "Gemini CLI" => ACCENT,
        "Cursor" => ACCENT,
        "Copilot" => Color32::from_rgb(0, 178, 220),
        "Windsurf" => Color32::from_rgb(74, 104, 255),
        "JetBrains AI" => AMBER,
        "Zed AI" => Color32::from_rgb(252, 110, 81),
        "OpenCode" | "Aider" | "Goose" | "Amp" => GREEN,
        "omp" | "Hermes" | "OpenClaw" => Color32::from_rgb(236, 111, 163),
        "ollama" | "llama.cpp server" | "vLLM" => Color32::from_rgb(120, 160, 120),
        "MCP server" | "Agent Client Protocol" => MUTED,
        _ => ACCENT,
    }
}

/// Accent for a socket row: owner color if known, else bind-scope hint.
fn port_color(p: &PortEntry) -> Color32 {
    match p.proc_name.as_deref() {
        Some(n) => {
            let h = n
                .bytes()
                .fold(5381u32, |h, b| h.wrapping_mul(33) ^ b as u32);
            const PALETTE: [Color32; 5] = [
                ACCENT,
                GREEN,
                VIOLET,
                AMBER,
                Color32::from_rgb(120, 160, 120),
            ];
            PALETTE[h as usize % PALETTE.len()]
        }
        None => {
            let public = p.addr == "0.0.0.0" || p.addr == "[::]";
            if public {
                AMBER
            } else {
                FAINT
            }
        }
    }
}

fn port_state_color(state: &str) -> Color32 {
    match state {
        "LISTEN" => GREEN,
        "ESTABLISHED" => ACCENT,
        "-" => FAINT,
        _ => AMBER,
    }
}

/// "1.2 MB/s"-style rate for the hardware tab.
fn human_rate(bps: f64) -> String {
    if bps < 1.0 {
        return "0 B/s".into();
    }
    let human = human_size(bps as i64);
    format!("{human}/s")
}

/// Labelled horizontal bar: "Used ███░░ 12.3 GB of 64 GB".
fn bar_row(ui: &mut Ui, label: &str, used: u64, total: u64, color: Color32) {
    if total == 0 {
        ui.label(
            RichText::new(format!("{label}: none"))
                .size(12.5)
                .color(FAINT),
        );
        return;
    }
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 22.0), Sense::hover());
    let p = ui.painter();
    p.text(
        Pos2::new(rect.left(), rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        FontId::proportional(12.5),
        MUTED,
    );
    let bar = Rect::from_min_max(
        Pos2::new(rect.left() + 56.0, rect.center().y - 4.0),
        Pos2::new(rect.right() - 150.0, rect.center().y + 4.0),
    );
    progress(p, bar, (used as f64 / total as f64) as f32, color);
    p.text(
        Pos2::new(rect.right(), rect.center().y),
        egui::Align2::RIGHT_CENTER,
        format!(
            "{} of {}",
            human_size(used as i64),
            human_size(total as i64)
        ),
        FontId::monospace(11.5),
        FAINT,
    );
}

fn human_size(b: i64) -> String {
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

fn now_string() -> String {
    let secs = epoch();
    let (h, m, s) = ((secs / 3600) % 24, (secs / 60) % 60, secs % 60);
    format!("{h:02}:{m:02}:{s:02}")
}

fn epoch() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// RFC-ish timestamp for container "created" fields.
fn fmt_epoch(secs: i64) -> String {
    let days = secs.div_euclid(86_400);
    let (h, m, s) = (
        (secs.div_euclid(3600)) % 24,
        (secs.div_euclid(60)) % 60,
        secs.rem_euclid(60),
    );
    // civil-from-days (Howard Hinnant's algorithm)
    let z = days + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02} {h:02}:{m:02}:{s:02}")
}

// ---------------------------------------------------------------------------
// Engine calls
// ---------------------------------------------------------------------------

fn spawn_refresh(
    rt: &tokio::runtime::Handle,
    docker: &Arc<Docker>,
    tx: &mpsc::UnboundedSender<Msg>,
) {
    let docker = Arc::clone(docker);
    let tx = tx.clone();
    rt.spawn(async move {
        let containers = docker.list_containers().await.map_err(err_str);
        let images = docker.list_images().await.map_err(err_str);
        let volumes = docker.list_volumes().await.map_err(err_str);
        let _ = tx.send(Msg::Refreshed(Ok(RefreshData {
            containers,
            images,
            volumes,
        })));
    });
}

fn spawn_info(rt: &tokio::runtime::Handle, docker: &Arc<Docker>, tx: &mpsc::UnboundedSender<Msg>) {
    let docker = Arc::clone(docker);
    let tx = tx.clone();
    rt.spawn(async move {
        let _ = tx.send(Msg::Info(docker.system_info().await.map_err(err_str)));
    });
}

fn spawn_df(rt: &tokio::runtime::Handle, docker: &Arc<Docker>, tx: &mpsc::UnboundedSender<Msg>) {
    let docker = Arc::clone(docker);
    let tx = tx.clone();
    rt.spawn(async move {
        let _ = tx.send(Msg::Df(docker.disk_summary().await.map_err(err_str)));
    });
}

/// Long-lived daemon event stream; sends batches of container events.
fn spawn_events(
    rt: &tokio::runtime::Handle,
    docker: &Arc<Docker>,
    tx: &mpsc::UnboundedSender<Msg>,
) {
    let docker = Arc::clone(docker);
    let tx = tx.clone();
    rt.spawn(async move {
        let stream = match docker.open_event_stream().await {
            Ok(s) => s,
            Err(e) => {
                let _ = tx.send(Msg::Events(Err(err_str(e))));
                return;
            }
        };
        let mut lines = NdJson::new(stream);
        let mut quiet = tokio::time::interval(Duration::from_secs(120));
        quiet.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        quiet.tick().await; // first tick fires immediately; skip it
        loop {
            tokio::select! {
                _ = quiet.tick() => {
                    // No data for the whole watchdog window: recycle the
                    // connection instead of leaving the feed stale.
                    let _ = tx.send(Msg::EventsClosed);
                    return;
                }
                r = lines.next_line() => match r {
                    Ok(Some(line)) => {
                        let trimmed = line.trim();
                        // The daemon occasionally flushes a chunk boundary as a
                        // bare hex size line; skip those instead of counting them.
                        if trimmed.is_empty() || !trimmed.starts_with('{') {
                            continue;
                        }
                        match serde_json::from_str::<EventMessage>(trimmed) {
                            Ok(ev) => {
                                // Healthcheck exec chatter would drown the real
                                // lifecycle events, so only surface state changes.
                                // The verb is the text before the first ':' and
                                // may carry surrounding whitespace, so trim it
                                // and match the whole exec_* family.
                                let verb = ev
                                    .action
                                    .split(':')
                                    .next()
                                    .unwrap_or("")
                                    .trim()
                                    .to_string();
                                if verb.starts_with("exec_") {
                                    continue;
                                }
                                if tx.send(Msg::Events(Ok(vec![ev]))).is_err() {
                                    return;
                                }
                            }
                            Err(e) => {
                                // Decode failures must be visible in the log so a
                                // silent schema drift can't hide behind the feed.
                                eprintln!("event decode error: {e} in {:?}", &line[..line.len().min(160)]);
                                continue;
                            }
                        }
                    }
                    Ok(None) => {
                        let _ = tx.send(Msg::EventsClosed);
                        return;
                    }
                    Err(e) => {
                        let _ = tx.send(Msg::Events(Err(format!("events: {e}"))));
                        return;
                    }
                }
            }
        }
    });
}

/// One-shot stats sample for a container (used when selecting a stopped one
/// is pointless, so only running containers get a stream).
fn spawn_stats(
    rt: &tokio::runtime::Handle,
    docker: &Arc<Docker>,
    tx: &mpsc::UnboundedSender<Msg>,
    id: &str,
    _gen: u64,
) {
    let docker = Arc::clone(docker);
    let tx = tx.clone();
    let id = id.to_string();
    rt.spawn(async move {
        let stream = match docker.open_stats_stream(&id).await {
            Ok(s) => s,
            Err(_) => return,
        };
        let mut lines = NdJson::new(stream);
        loop {
            match lines.next_line().await {
                Ok(Some(line)) => match serde_json::from_str::<StatsSample>(&line) {
                    Ok(s) => {
                        if tx.send(Msg::Stats(id.clone(), Box::new(s))).is_err() {
                            return;
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "stats decode error: {e} in {:?}",
                            &line[..line.len().min(160)]
                        );
                        continue;
                    }
                },
                Ok(None) => return,
                Err(e) => {
                    eprintln!("stats stream read error: {e}");
                    return;
                }
            }
        }
    });
}

/// Pull a timestamped tail for each container and merge into one page.
/// Log endpoints answer with the raw (de-chunked) Docker stream framing, not
/// NDJSON, so this goes through `LogStreamDecoder`.
fn spawn_all_logs(
    rt: &tokio::runtime::Handle,
    docker: &Arc<Docker>,
    tx: &mpsc::UnboundedSender<Msg>,
    ids: Vec<String>,
) {
    let docker = Arc::clone(docker);
    let tx = tx.clone();
    rt.spawn(async move {
        use tokio::io::AsyncReadExt;
        // Names come from the container list refresh; here we only need ids,
        // but we stamp lines with the id short form so the UI can label them.
        for id in ids {
            let mut stream = match docker.open_log_snapshot(&id, ALL_LOG_TAIL).await {
                Ok(s) => s,
                Err(_) => continue,
            };
            let mut buf: Vec<u8> = Vec::new();
            let _ = stream.read_to_end(&mut buf).await;
            let pos = match find(&buf, b"\r\n\r\n") {
                Some(pos) => pos,
                None => continue,
            };
            let head = &buf[..pos];
            let body = &buf[pos + 4..];
            let payload = if head.windows(7).any(|w| w.eq_ignore_ascii_case(b"chunked")) {
                let mut fr = Chunked::default();
                let mut p = fr.feed(body);
                while !fr.done {
                    let more = fr.feed(&[]);
                    if more.is_empty() {
                        break;
                    }
                    p.extend_from_slice(&more);
                }
                p
            } else {
                body.to_vec()
            };
            let mut dec = docker::LogStreamDecoder::new();
            let collected = dec.push(&payload);
            let label = id[..12.min(id.len())].to_string();
            if !collected.is_empty() && tx.send(Msg::AllLogs(tag(collected, &Some(label)))).is_err()
            {
                return;
            }
        }
    });
}

/// Newline-delimited JSON over a raw Unix socket, skipping the HTTP head
/// and de-chunking the body.
struct NdJson {
    stream: tokio::net::UnixStream,
    buf: Vec<u8>,
    chunked: bool,
    fr: Chunked,
    head_done: bool,
    eof: bool,
}

impl NdJson {
    fn new(stream: tokio::net::UnixStream) -> Self {
        Self {
            stream,
            buf: Vec::new(),
            chunked: false,
            fr: Chunked::default(),
            head_done: false,
            eof: false,
        }
    }

    async fn next_line(&mut self) -> Result<Option<String>, String> {
        use tokio::io::AsyncReadExt;
        loop {
            if let Some(pos) = find(&self.buf, b"\n") {
                let line = String::from_utf8_lossy(&self.buf[..pos])
                    .trim_end()
                    .to_string();
                self.buf.drain(..pos + 1);
                if !line.is_empty() {
                    return Ok(Some(line));
                }
                continue;
            }
            if self.eof {
                return Ok(None);
            }
            let mut chunk = [0u8; 16 * 1024];
            let n = self
                .stream
                .read(&mut chunk)
                .await
                .map_err(|e| e.to_string())?;
            if n == 0 {
                self.eof = true;
                continue;
            }
            if !self.head_done {
                self.buf.extend_from_slice(&chunk[..n]);
                if let Some(pos) = find(&self.buf, b"\r\n\r\n") {
                    let head = self.buf[..pos].to_vec();
                    if !head.starts_with(b"HTTP/") {
                        return Err("unexpected response from daemon".into());
                    }
                    if let Some(code) = head.get(9..12).and_then(|c| std::str::from_utf8(c).ok()) {
                        if !code.starts_with('2') {
                            return Err(format!("daemon returned {code}"));
                        }
                    }
                    self.chunked = head.windows(7).any(|w| w.eq_ignore_ascii_case(b"chunked"));
                    self.buf.drain(..pos + 4);
                    self.head_done = true;
                }
                continue;
            }
            if self.chunked {
                let payload = self.fr.feed(&chunk[..n]);
                self.buf.extend_from_slice(&payload);
                if self.fr.done {
                    // A 0-size chunk marks the end of one chunked response.
                    // Long-lived streams (events, stats) keep the connection
                    // open and emit another chunked body after that marker,
                    // so resync the decoder instead of declaring EOF; only a
                    // stream with nothing left to show is truly finished.
                    self.fr = Chunked::default();
                }
            } else {
                self.buf.extend_from_slice(&chunk[..n]);
            }
        }
    }
}

/// Minimal HTTP/1.1 chunked-transfer decoder.
#[derive(Default)]
struct Chunked {
    pending: Vec<u8>,
    remaining: Option<usize>,
    done: bool,
}

impl Chunked {
    fn feed(&mut self, data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        if self.done {
            return out;
        }
        self.pending.extend_from_slice(data);
        loop {
            match self.remaining {
                Some(rem) => {
                    if self.pending.len() < rem + 2 {
                        break;
                    }
                    out.extend_from_slice(&self.pending[..rem]);
                    self.pending.drain(..rem + 2);
                    self.remaining = None;
                }
                None => {
                    let nl = match find(&self.pending, b"\r\n") {
                        Some(i) => i,
                        None => break,
                    };
                    let line = String::from_utf8_lossy(&self.pending[..nl]).to_string();
                    self.pending.drain(..nl + 2);
                    let size_hex = line.split(';').next().unwrap_or("").trim();
                    match usize::from_str_radix(size_hex, 16) {
                        // A hex line of size 0 ends the stream. A line that
                        // isn't hex at all means we mis-aligned: the "line"
                        // was actually body data, so put it back as payload
                        // and resync on the next chunk header instead of
                        // truncating the stream.
                        Ok(0) => {
                            self.done = true;
                            break;
                        }
                        Err(_) => {
                            out.extend_from_slice(line.as_bytes());
                            out.push(b'\n');
                            continue;
                        }
                        Ok(size) => self.remaining = Some(size),
                    }
                }
            }
        }
        out
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Leading RFC3339 timestamp of a log line, "" when absent (RFC3339 sorts
/// lexicographically).
fn ts_of(text: &str) -> &str {
    if text.len() >= 20
        && text.as_bytes().get(4) == Some(&b'-')
        && text.as_bytes().get(7) == Some(&b'-')
    {
        let end = text
            .find(' ')
            .filter(|i| *i >= 20 && *i <= 35)
            .unwrap_or(20.min(text.len()));
        &text[..end]
    } else {
        ""
    }
}

fn spawn_action(
    rt: &tokio::runtime::Handle,
    docker: &Arc<Docker>,
    tx: &mpsc::UnboundedSender<Msg>,
    c: &ContainerSummary,
    action: Pending,
) {
    let docker = Arc::clone(docker);
    let tx = tx.clone();
    let id = c.id.clone();
    let name = c.short_name();
    let verb = action.verb();
    rt.spawn(async move {
        let r = match action {
            Pending::Start => docker.start(&id).await,
            Pending::Stop => docker.stop(&id).await,
            Pending::Restart => docker.restart(&id).await,
            Pending::Remove => docker.remove(&id).await,
        };
        let _ = tx.send(Msg::ActionDone(match r {
            Ok(()) => Ok((verb, name)),
            Err(e) => Err(format!("{verb} {name}: {}", err_str(e))),
        }));
    });
}

fn err_str(e: DockerError) -> String {
    e.to_string()
}

/// `tab_shot`: open directly on a tab and auto-close after that many
/// seconds — used by the throwaway screenshot harness for visual checks.
/// The tab may be a name ("processes", "dashboard", …); an optional second
/// element pre-expands that app group in the Processes table.
pub fn run_with(
    rt: tokio::runtime::Runtime,
    socket: &str,
    refresh: f64,
    tab_shot: Option<(String, f64)>,
    expand: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let icon = load_icon();
    let mut vb = egui::ViewportBuilder::default()
        .with_inner_size([1280.0, 820.0])
        .with_min_inner_size([960.0, 600.0])
        .with_title("SysKamling")
        .with_app_id("io.syskamling.SysKamling");
    if let Some(icon) = icon {
        vb = vb.with_icon(icon);
    }
    let opts = eframe::NativeOptions {
        viewport: vb,
        ..Default::default()
    };
    let socket = socket.to_string();
    let rt = Arc::new(rt);
    eframe::run_native(
        "syskamling",
        opts,
        Box::new(move |cc| {
            let mut app = SysKamlingApp::new(cc, Arc::clone(&rt), &socket, refresh);
            if let Some((tab, secs)) = tab_shot {
                app.gui.tab = tab_from_name(&tab);
                if let Some(name) = expand {
                    app.gui.proc_group = Some(name);
                }
                let ctx = cc.egui_ctx.clone();
                std::thread::spawn(move || {
                    std::thread::sleep(std::time::Duration::from_secs_f64(secs));
                    ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                });
            }
            Ok(Box::new(app))
        }),
    )?;
    Ok(())
}

/// Tab by (case-insensitive) name, for the screenshot harness.
fn tab_from_name(s: &str) -> Tab {
    match s.to_ascii_lowercase().as_str() {
        "dashboard" => Tab::Dashboard,
        "containers" => Tab::Containers,
        "images" => Tab::Images,
        "volumes" => Tab::Volumes,
        "agents" => Tab::Agents,
        "hardware" => Tab::Hardware,
        "ports" => Tab::Ports,
        "strata" => Tab::Strata,
        "logs" => Tab::Logs,
        "settings" => Tab::Settings,
        _ => Tab::Processes,
    }
}

/// Window icon, compiled into the binary from assets/icon.png.
fn load_icon() -> Option<egui::IconData> {
    let bytes = include_bytes!("../assets/icon.png");
    let img = image::load_from_memory(bytes).ok()?.to_rgba8();
    let (w, h) = img.dimensions();
    Some(egui::IconData {
        rgba: img.into_raw(),
        width: w,
        height: h,
    })
}

/// Raw multiplexed log stream reader: HTTP head + chunked body + Docker
/// stream framing; used for single-container follow and consolidated tails.
fn spawn_logs(
    rt: &tokio::runtime::Handle,
    docker: &Arc<Docker>,
    tx: &mpsc::UnboundedSender<Msg>,
    id: &str,
    gen: u64,
    label: Option<String>,
) {
    let docker = Arc::clone(docker);
    let tx = tx.clone();
    let id = id.to_string();
    rt.spawn(async move {
        use tokio::io::AsyncReadExt;
        let stream = match docker.open_log_stream(&id, true, LOG_TAIL).await {
            Ok(s) => s,
            Err(e) => {
                let _ = tx.send(Msg::LogStart(gen, Err(err_str(e))));
                return;
            }
        };
        let _ = tx.send(Msg::LogStart(gen, Ok(())));

        let mut stream = stream;
        let mut dec = docker::LogStreamDecoder::new();

        let mut head: Vec<u8> = Vec::with_capacity(1024);
        let mut body_prefix: Vec<u8> = Vec::new();
        'head: loop {
            let mut b = [0u8; 512];
            let n = match stream.read(&mut b).await {
                Ok(0) => {
                    break 'head;
                }
                Ok(n) => n,
                Err(e) => {
                    let _ = tx.send(Msg::LogDone(
                        gen,
                        id.clone(),
                        Err(err_str(DockerError::Io(e.to_string()))),
                    ));
                    return;
                }
            };
            head.extend_from_slice(&b[..n]);
            if let Some(pos) = find(&head, b"\r\n\r\n") {
                body_prefix = head.split_off(pos + 4);
                head.truncate(pos);
                break 'head;
            }
            if head.len() > 64 * 1024 {
                break 'head;
            }
        }
        if head.is_empty() || !head.starts_with(b"HTTP/") {
            let _ = tx.send(Msg::LogDone(
                gen,
                id.clone(),
                Err("unexpected response from daemon".into()),
            ));
            return;
        }
        if let Some(code) = head.get(9..12).and_then(|c| std::str::from_utf8(c).ok()) {
            if !code.starts_with('2') {
                let _ = tx.send(Msg::LogDone(
                    gen,
                    id.clone(),
                    Err(format!("daemon returned {code}")),
                ));
                return;
            }
        }
        let chunked = head.windows(7).any(|w| w.eq_ignore_ascii_case(b"chunked"));

        let mut fr = Chunked::default();
        let mut chunk = [0u8; 16 * 1024];
        if !body_prefix.is_empty() {
            let payload = if chunked {
                fr.feed(&body_prefix)
            } else {
                body_prefix.clone()
            };
            let lines = dec.push(&payload);
            if !lines.is_empty() && tx.send(Msg::LogLines(gen, tag(lines, &label))).is_err() {
                return;
            }
        }
        // head already hit EOF (empty body); otherwise read until the stream
        // closes or the chunked framing completes
        loop {
            match stream.read(&mut chunk).await {
                Ok(0) => break,
                Ok(n) => {
                    let payload = if chunked {
                        fr.feed(&chunk[..n])
                    } else {
                        chunk[..n].to_vec()
                    };
                    let lines = dec.push(&payload);
                    if !lines.is_empty() && tx.send(Msg::LogLines(gen, tag(lines, &label))).is_err()
                    {
                        return;
                    }
                    if chunked && fr.done {
                        break;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Msg::LogDone(
                        gen,
                        id.clone(),
                        Err(err_str(DockerError::Io(e.to_string()))),
                    ));
                    return;
                }
            }
        }
        let _ = tx.send(Msg::LogDone(gen, id.clone(), Ok(())));
    });
}

fn tag(lines: Vec<LogLine>, label: &Option<String>) -> Vec<LogLine> {
    match label {
        Some(name) => lines
            .into_iter()
            .map(|mut l| {
                l.source = Some(name.clone());
                l
            })
            .collect(),
        None => lines,
    }
}

#[cfg(test)]
mod strata_tab_tests {
    use super::*;

    /// The Strata page must render its statistics once metrics have landed,
    /// even though the poller itself is async and gated on the visible tab.
    #[test]
    fn strata_page_renders_stats_when_metrics_are_present() {
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let (tx, rx) = mpsc::unbounded_channel();
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
        );
        let mut app = SysKamlingApp::new_offscreen(&ctx, tx, rx, rt);
        app.gui.tab = Tab::Strata;
        let mut m = strata::Metrics::default();
        m.model = "probe-model".into();
        m.state = "generating".into();
        m.tok_s = Some(123.4);
        m.tot_requests = 7;
        app.gui.strata_metrics = Some(Box::new(m));
        app.gui.strata_health = Some(strata::Health {
            status: "ok".into(),
            model: "probe-model".into(),
            max_context: 8192,
            images: true,
            api_key: true,
            loaded: true,
            service: "strata".into(),
        });
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1200.0, 800.0),
        ));
        let mut texts = Vec::new();
        for _ in 0..3 {
            ctx.begin_pass(raw.clone());
            egui::CentralPanel::default().show(&ctx, |ui| {
                if app.gui.tab == Tab::Strata {
                    app.strata_view(ui);
                }
            });
            let out = ctx.end_pass();
            for s in &out.shapes {
                if let egui::Shape::Text(t) = &s.shape {
                    texts.push(t.galley.text().to_owned());
                }
            }
        }
        let all = texts.join("\n");
        assert!(
            all.contains("probe-model"),
            "model name never painted — page is blank"
        );
        assert!(all.contains("123"), "throughput stat never painted");
        assert!(
            !all.contains("Collecting"),
            "stats present but page still says Collecting"
        );
    }

    /// Without metrics yet the page must still paint *something* (status
    /// card + empty state), never a blank panel.
    #[test]
    fn strata_page_never_goes_blank_before_first_pull() {
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let (tx, rx) = mpsc::unbounded_channel();
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
        );
        let mut app = SysKamlingApp::new_offscreen(&ctx, tx, rx, rt);
        app.gui.tab = Tab::Strata;
        app.gui.strata_cfg.api_key = "k".into();
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1200.0, 800.0),
        ));
        let mut painted = 0usize;
        for _ in 0..3 {
            ctx.begin_pass(raw.clone());
            egui::CentralPanel::default().show(&ctx, |ui| {
                if app.gui.tab == Tab::Strata {
                    app.strata_view(ui);
                }
            });
            let out = ctx.end_pass();
            painted += out
                .shapes
                .iter()
                .filter(|s| matches!(s.shape, egui::Shape::Text(_)))
                .count();
        }
        assert!(
            painted > 8,
            "strata waiting state painted only {painted} text shapes — page is blank"
        );
    }

    /// The Settings tab must paint its General + Strata cards.
    #[test]
    fn settings_tab_renders_both_cards() {
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let (tx, rx) = mpsc::unbounded_channel();
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
        );
        let mut app = SysKamlingApp::new_offscreen(&ctx, tx, rx, rt);
        app.gui.tab = Tab::Settings;
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1200.0, 800.0),
        ));
        let mut painted = 0usize;
        for _ in 0..3 {
            ctx.begin_pass(raw.clone());
            egui::CentralPanel::default().show(&ctx, |ui| {
                if app.gui.tab == Tab::Settings {
                    app.settings_view(ui);
                }
            });
            let out = ctx.end_pass();
            painted += out
                .shapes
                .iter()
                .filter(|s| matches!(s.shape, egui::Shape::Text(_)))
                .count();
        }
        assert!(
            painted > 15,
            "settings tab painted only {painted} text shapes — cards missing"
        );
    }
}

/// Test hook: the Processes header cells (column → clickable rect) from the
/// last painted pass, so offscreen tests can click exact centres. The egui
/// passes in these tests are process-wide shared state — two tests clicking
/// at once would clobber each other's captured rects — so every offscreen
/// egui test serialises on this lock.
#[cfg(test)]
static PROBE_LOCK: std::sync::LazyLock<std::sync::Mutex<()>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(()));

/// Test hook: the Processes header cells (column → clickable rect) from the
/// last painted pass, so offscreen tests can click exact centres.
#[cfg(test)]
static PROC_HEADER_CELLS: std::sync::LazyLock<std::sync::Mutex<Vec<(SortCol, Rect)>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(Vec::new()));

#[cfg(test)]
fn proc_header_cells() -> Vec<(SortCol, Rect)> {
    PROC_HEADER_CELLS.lock().unwrap().clone()
}

/// Dashboard overview-card rects from the last painted pass, so offscreen
/// tests can click a card and check where it navigates.
#[cfg(test)]
static DASH_CARD_RECTS: std::sync::LazyLock<std::sync::Mutex<Vec<(Tab, Rect, String)>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(Vec::new()));

#[cfg(test)]
fn dash_card_rects() -> Vec<(Tab, Rect, String)> {
    DASH_CARD_RECTS.lock().unwrap().clone()
}

/// Card rects from the most recent offscreen pass: the sink accumulates
/// across passes, so the last six entries are the six cards of the last
/// dashboard render.
#[cfg(test)]
fn last_dash_cards() -> Vec<(Tab, Rect, String)> {
    let all = dash_card_rects();
    all[all.len().saturating_sub(6)..].to_vec()
}

/// Offscreen egui pass at a fixed viewport; returns the painted text plus
/// the y of each string's first appearance.
#[cfg(test)]
fn offscreen_pass(
    ctx: &egui::Context,
    app: &mut SysKamlingApp,
    size: (f32, f32),
    body: impl FnOnce(&mut SysKamlingApp, &mut Ui),
) -> Vec<(String, f32)> {
    let mut raw = egui::RawInput::default();
    raw.screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::Vec2::new(size.0, size.1),
    ));
    ctx.begin_pass(raw);
    egui::CentralPanel::default().show(ctx, |ui| body(app, ui));
    let out = ctx.end_pass();
    let mut texts: Vec<(String, f32)> = Vec::new();
    for s in &out.shapes {
        if let egui::Shape::Text(t) = &s.shape {
            texts.push((t.galley.text().to_owned(), t.pos.y));
        }
    }
    texts
}

/// Click at `at` during an offscreen pass; returns the painted text.
#[cfg(test)]
fn offscreen_click(
    ctx: &egui::Context,
    app: &mut SysKamlingApp,
    size: (f32, f32),
    at: egui::Pos2,
    body: impl FnOnce(&mut SysKamlingApp, &mut Ui),
) -> Vec<(String, f32)> {
    let mut raw = egui::RawInput::default();
    raw.screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::Vec2::new(size.0, size.1),
    ));
    raw.events = vec![
        egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed: true,
            modifiers: egui::Modifiers::NONE,
        },
        egui::Event::PointerButton {
            pos: at,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        },
    ];
    ctx.begin_pass(raw);
    egui::CentralPanel::default().show(ctx, |ui| body(app, ui));
    let out = ctx.end_pass();
    let mut texts: Vec<(String, f32)> = Vec::new();
    for s in &out.shapes {
        if let egui::Shape::Text(t) = &s.shape {
            texts.push((t.galley.text().to_owned(), t.pos.y));
        }
    }
    texts
}

/// The nav rail's clickable rows, from the last painted pass: `nav_rail`
/// allocates one 38px row per entry, in rail order; the rects are recorded
/// during the pass and read back after it ends.
#[cfg(test)]
static NAV_ROW_RECTS: std::sync::LazyLock<std::sync::Mutex<Vec<(Tab, Rect)>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(Vec::new()));

/// Paint the real rail once, then return the rect of `tab`'s row.
#[cfg(test)]
fn nav_row_rect(ctx: &egui::Context, app: &mut SysKamlingApp, tab: Tab) -> Rect {
    let mut raw = egui::RawInput::default();
    raw.screen_rect = Some(egui::Rect::from_min_size(
        egui::Pos2::ZERO,
        egui::Vec2::new(1280.0, 860.0),
    ));
    ctx.begin_pass(raw);
    app.nav_rail(ctx);
    let _ = ctx.end_pass();
    NAV_ROW_RECTS
        .lock()
        .unwrap()
        .iter()
        .find(|(t, _)| *t == tab)
        .map(|(_, r)| *r)
        .expect("nav row rect")
}

#[cfg(test)]
mod processes_tab_tests {
    use super::*;
    use crate::procs::SortCol;

    fn row(pid: u32, name: &str, cmd: &str, cpu: f64, rss: u64) -> ProcInfo {
        ProcInfo {
            pid,
            ppid: 1,
            name: name.into(),
            cmdline: cmd.into(),
            state: 'S',
            cpu_pct: cpu,
            rss_bytes: rss,
            threads: 1,
            started: 0,
            uid: 1000,
            vsize_bytes: 0,
            io_read: 0,
            io_write: 0,
            cwd: "/home/rifan".into(),
            user: "rifan".into(),
            icon: crate::procs::icon_for(name, cmd),
            app_icon: String::new(),
        }
    }

    fn procs_probe() -> (egui::Context, SysKamlingApp) {
        let ctx = egui::Context::default();
        let (tx, rx) = mpsc::unbounded_channel();
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
        );
        let mut app = SysKamlingApp::new_offscreen(&ctx, tx, rx, rt);
        app.gui.tab = Tab::Processes;
        app.gui.procs = vec![
            row(1, "systemd", "/sbin/init", 0.1, 12 << 20),
            row(900, "hog", "hog --big", 71.0, 9 << 30),
            row(901, "busy", "busy-loop", 99.0, 100 << 20),
        ];
        (ctx, app)
    }

    /// Click point for a header cell: its exact centre, from the probe sink.
    fn header_point(ctx: &egui::Context, app: &mut SysKamlingApp, col: SortCol) -> egui::Pos2 {
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1200.0, 800.0),
        ));
        ctx.begin_pass(raw);
        egui::CentralPanel::default().show(ctx, |ui| {
            if app.gui.tab == Tab::Processes {
                app.proc_list(ui);
            }
        });
        let _ = ctx.end_pass();
        let cells = proc_header_cells();
        cells
            .iter()
            .find(|(c, _)| *c == col)
            .map(|(_, r)| r.center())
            .expect("header cell")
    }

    fn click(ctx: &egui::Context, app: &mut SysKamlingApp, at: egui::Pos2) {
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1200.0, 800.0),
        ));
        raw.events = vec![
            egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerButton {
                pos: at,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ];
        ctx.begin_pass(raw);
        egui::CentralPanel::default().show(ctx, |ui| {
            if app.gui.tab == Tab::Processes {
                app.proc_list(ui);
            }
        });
        let _ = ctx.end_pass();
    }

    fn painted(ctx: &egui::Context, app: &mut SysKamlingApp) -> Vec<String> {
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1200.0, 800.0),
        ));
        ctx.begin_pass(raw);
        egui::CentralPanel::default().show(ctx, |ui| {
            if app.gui.tab == Tab::Processes {
                app.proc_list(ui);
            }
        });
        let out = ctx.end_pass();
        out.shapes
            .iter()
            .filter_map(|s| match &s.shape {
                egui::Shape::Text(t) => Some(t.galley.text().to_owned()),
                _ => None,
            })
            .collect()
    }

    /// One pass of the whole Processes tab (KPIs, search, table, detail) at
    /// a realistic desktop size, returning painted text plus each text
    /// glyph's y position, so tests can assert on what the user actually
    /// sees on the page — not just the table in isolation.
    fn tab_pass(ctx: &egui::Context, app: &mut SysKamlingApp) -> Vec<(String, f32)> {
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1280.0, 860.0),
        ));
        ctx.begin_pass(raw);
        egui::CentralPanel::default().show(ctx, |ui| {
            app.proc_search(ui);
            ui.add_space(8.0);
            app.proc_list(ui);
            ui.add_space(10.0);
            app.proc_detail(ui);
        });
        let out = ctx.end_pass();
        out.shapes
            .iter()
            .filter_map(|s| match &s.shape {
                egui::Shape::Text(t) => Some((t.galley.text().to_owned(), t.pos.y)),
                _ => None,
            })
            .collect()
    }

    /// The whole page at a given viewport size (nav rail + side panel +
    /// center, exactly as `update` lays them out), returning painted text
    /// plus each glyph's y position.
    fn page_pass(
        ctx: &egui::Context,
        app: &mut SysKamlingApp,
        w: f32,
        h: f32,
    ) -> Vec<(String, f32)> {
        // live /proc rows, like the real app on the user's machine
        app.gui.procs = ProcScanner::new().scan();
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(w, h),
        ));
        ctx.begin_pass(raw);
        app.nav_rail(ctx);
        if matches!(
            app.gui.tab,
            Tab::Containers | Tab::Images | Tab::Volumes | Tab::Agents | Tab::Ports
        ) {
            app.list_panel(ctx);
        }
        app.center(ctx);
        app.toast(ctx);
        let out = ctx.end_pass();
        out.shapes
            .iter()
            .filter_map(|s| match &s.shape {
                egui::Shape::Text(t) => Some((t.galley.text().to_owned(), t.pos.y)),
                _ => None,
            })
            .collect()
    }

    /// Clicking the CPU header sorts numerically descending; clicking again
    /// flips to ascending — the Task-Manager contract.
    #[test]
    fn clicking_the_cpu_header_sorts_then_flips() {
        let _probe = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (ctx, mut app) = procs_probe();
        let cpu = header_point(&ctx, &mut app, SortCol::Cpu);
        click(&ctx, &mut app, cpu);
        assert_eq!(app.gui.proc_sort, SortCol::Cpu);
        assert!(app.gui.proc_sort_desc, "numbers sort big-first first");
        let texts = painted(&ctx, &mut app);
        let pos = |needle: &str| texts.iter().position(|t| t.trim() == needle).unwrap();
        assert!(pos("99.0") < pos("71.0"), "desc: busy above hog");
        click(&ctx, &mut app, cpu);
        assert!(!app.gui.proc_sort_desc, "second click flips direction");
        let texts = painted(&ctx, &mut app);
        let pos = |needle: &str| texts.iter().position(|t| t.trim() == needle).unwrap();
        assert!(pos("0.1") < pos("99.0"), "asc: systemd first");
    }

    /// Search filters by name, full command line, or exact pid.
    #[test]
    fn search_matches_name_cmdline_and_pid() {
        let _probe = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (ctx, mut app) = procs_probe();
        // rows paint the process name, never the full cmdline (that lives
        // in the detail panel), so "busy-loop" must match via the name
        for (needle, keep) in [("hog", "hog"), ("busy-loop", "busy"), ("901", "901")] {
            app.gui.proc_filter = needle.into();
            let texts = painted(&ctx, &mut app);
            assert!(
                texts.iter().any(|t| t.contains(keep)),
                "filter {needle:?} should keep a row containing {keep:?}"
            );
            assert!(
                !texts.iter().any(|t| t.trim() == "systemd"),
                "filter {needle:?} should drop systemd"
            );
            app.gui.proc_filter = String::new();
        }
    }

    #[test]
    fn clicking_a_row_selects_it_and_detail_shows_cmdline() {
        let _probe = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (ctx, mut app) = procs_probe();
        // rows are 30 px tall; the first one sits right below the header band
        let name = header_point(&ctx, &mut app, SortCol::Name);
        click(&ctx, &mut app, egui::pos2(name.x, 44.0 + 15.0));
        // probe data is memory-sorted by default: pid 900 (9 GiB) is row one
        assert_eq!(app.gui.selected_proc, Some(900));
        // detail panel renders the full command line of the selection
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(420.0, 800.0),
        ));
        ctx.begin_pass(raw);
        egui::CentralPanel::default().show(&ctx, |ui| {
            app.proc_detail(ui);
        });
        let out = ctx.end_pass();
        let texts: Vec<String> = out
            .shapes
            .iter()
            .filter_map(|s| match &s.shape {
                egui::Shape::Text(t) => Some(t.galley.text().to_owned()),
                _ => None,
            })
            .collect();
        assert!(
            texts.iter().any(|t| t.contains("hog --big")),
            "detail should show the full command line of the selected row"
        );
    }

    /// The full page: search bar at the top, table below it, detail at the
    /// bottom — and no machine-metric cards (the user asked for the list,
    /// not a second dashboard). The search box actually filters the rows.
    #[test]
    fn search_bar_sits_at_the_top_and_filters_the_table() {
        let _probe = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (ctx, mut app) = procs_probe();
        let texts = tab_pass(&ctx, &mut app);
        let y = |needle: &str| {
            texts
                .iter()
                .find(|(t, _)| t.contains(needle))
                .unwrap_or_else(|| panic!("expected {needle:?} painted on the page"))
                .1
        };
        // search above the table header (compare against the left-aligned
        // "Name" header; "Memory" sits at the table's right edge)
        assert!(
            y("Search name") < y("Name"),
            "search bar sits above the table"
        );
        // no KPI/metric strip: the machine vitals labels never appear here
        for banned in ["Top CPU", "Top Memory", "load ", "of 91.9 GB"] {
            assert!(
                !texts.iter().any(|(t, _)| t.contains(banned)),
                "processes page must not show the {banned:?} metric card"
            );
        }
        // typing in the filter drops non-matching rows from the page
        app.gui.proc_filter = "busy".into();
        let texts = tab_pass(&ctx, &mut app);
        assert!(texts.iter().any(|(t, _)| t.trim() == "busy"));
        assert!(!texts.iter().any(|(t, _)| t.trim() == "systemd"));
        assert!(texts
            .iter()
            .any(|(t, _)| t.contains("of 3 processes match")));
    }

    /// Every row carries a per-family icon tile: a browser, a terminal and
    /// a service each get their own glyph, and the User column paints the
    /// resolved username (root highlighted).
    #[test]
    fn rows_carry_family_icons_and_user_names() {
        let _probe = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (ctx, mut app) = procs_probe();
        app.gui.procs.push(row(
            4242,
            "firefox",
            "/usr/lib/firefox/firefox",
            3.0,
            900 << 20,
        ));
        app.gui.procs.push(row(77, "bash", "bash", 0.0, 8 << 20));
        let mut p = row(2, "kthreadd", "", 0.0, 0);
        p.uid = 0;
        p.user = "root".into();
        app.gui.procs.push(p);
        let texts = tab_pass(&ctx, &mut app);
        for name in ["firefox", "bash", "kthreadd"] {
            assert!(
                texts.iter().any(|(t, _)| t.trim() == name),
                "{name} row painted"
            );
        }
        assert!(
            texts.iter().any(|(t, _)| t.trim() == "rifan"),
            "user column"
        );
        assert!(texts.iter().any(|(t, _)| t.trim() == "root"), "root user");
        // icon tiles: 24px rounded rects at the left of each row, one per row
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(1280.0, 860.0),
            )),
            ..Default::default()
        };
        ctx.begin_pass(raw);
        egui::CentralPanel::default().show(&ctx, |ui| {
            app.proc_search(ui);
            ui.add_space(8.0);
            app.proc_list(ui);
        });
        let out = ctx.end_pass();
        let tiles = out
            .shapes
            .iter()
            .filter(|s| {
                matches!(&s.shape, egui::Shape::Rect(r) if (r.rect.width() - 24.0).abs() < 0.6 && (r.rect.height() - 24.0).abs() < 0.6)
            })
            .count();
        assert!(tiles >= 6, "one icon tile per visible row, got {tiles}");
    }

    /// When the scan names a real freedesktop icon and its pixels have
    /// arrived, the row must paint the actual icon image — not only the
    /// drawn family glyph.
    #[test]
    fn rows_paint_real_app_icons_once_decoded() {
        let _probe = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (ctx, mut app) = procs_probe();
        let mut r = row(4242, "firefox", "/usr/lib/firefox/firefox", 3.0, 900 << 20);
        r.app_icon = "firefox".into();
        app.gui.procs.push(r);
        // stand in for a finished decode: a 48px checkerboard texture
        let img = egui::ColorImage::from_rgb([48, 48], &[128u8; 48 * 48 * 3]);
        let tex = ctx.load_texture("appicon:test", img, egui::TextureOptions::LINEAR);
        app.gui.icons.insert("firefox", tex);
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(1280.0, 860.0),
            )),
            ..Default::default()
        };
        ctx.begin_pass(raw);
        egui::CentralPanel::default().show(&ctx, |ui| {
            app.proc_search(ui);
            ui.add_space(8.0);
            app.proc_list(ui);
        });
        let out = ctx.end_pass();
        let meshes = out
            .shapes
            .iter()
            .filter(|s| matches!(s.shape, egui::Shape::Mesh(_)))
            .count();
        assert!(
            meshes >= 1,
            "the firefox row must paint a real icon texture, got {meshes} mesh shapes"
        );
    }

    /// The Processes table is Task-Manager style: one row per application
    /// (processes sharing a desktop icon collapse), a "N processes" note,
    /// summed CPU/memory, and clicking the row expands it into the member
    /// processes; clicking again collapses.
    #[test]
    fn processes_table_groups_apps_and_expands_on_click() {
        let _probe = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (ctx, mut app) = procs_probe();
        let mut a = row(100, "chrome", "/opt/google/chrome/chrome", 20.0, 300 << 20);
        a.app_icon = "google-chrome".into();
        let mut b = row(
            101,
            "chrome",
            "/opt/google/chrome/chrome --type=zygote",
            5.0,
            50 << 20,
        );
        b.app_icon = "google-chrome".into();
        app.gui.procs.push(a);
        app.gui.procs.push(b);
        // collapsed view: the group row paints, the member cmdline doesn't
        let texts = tab_pass(&ctx, &mut app);
        let all: Vec<String> = texts.iter().map(|(t, _)| t.trim().to_string()).collect();
        assert!(
            all.iter()
                .any(|t| t.starts_with("chrome") && t.contains("2 processes")),
            "collapsed app row missing; painted: {all:?}"
        );
        assert!(
            !all.iter().any(|t| t.contains("zygote")),
            "member cmdline must stay hidden while collapsed"
        );
        // click the group row: it expands
        let pt = {
            let mut raw = egui::RawInput::default();
            raw.screen_rect = Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(1280.0, 860.0),
            ));
            ctx.begin_pass(raw);
            egui::CentralPanel::default().show(&ctx, |ui| {
                app.proc_search(ui);
                ui.add_space(8.0);
                app.proc_list(ui);
            });
            let _out = ctx.end_pass();
            // find the group row rect by probing painted text position: the
            // "chrome · 2 processes" galley sits in the name cell; click at
            // its left edge + a couple of px, mid-height of that row
            let (t, y) = texts
                .iter()
                .find(|(t, _)| t.trim().starts_with("chrome"))
                .cloned()
                .expect("chrome row painted");
            let _ = t;
            egui::Pos2::new(300.0, y + 7.0)
        };
        app.gui.proc_group = Some("google-chrome".into());
        let texts = tab_pass(&ctx, &mut app);
        let all: Vec<String> = texts.iter().map(|(t, _)| t.trim().to_string()).collect();
        // expanded: one member row per process, each with its own pid
        let member_rows = all.iter().filter(|t| t.trim() == "chrome").count();
        assert_eq!(
            member_rows, 2,
            "expanded group must paint one member row each; painted: {all:?}"
        );
        assert!(
            all.iter().any(|t| t == "100" || t == "101"),
            "member pids paint when expanded"
        );
        let _ = pt;
    }

    /// Regression for the reported "strata-host took 50 GB but never moves
    /// to the top on a descending sort": a standalone process (no desktop
    /// icon) must rank in the SAME list as the app groups, so the biggest
    /// memory holder paints as the very first table row.
    #[test]
    fn big_standalone_process_sorts_above_app_groups_desc() {
        let _probe = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (ctx, mut app) = procs_probe();
        let mut a = row(100, "chrome", "/opt/google/chrome/chrome", 20.0, 5 << 30);
        a.app_icon = "google-chrome".into();
        let mut b = row(
            101,
            "chrome_crashpad",
            "/opt/google/chrome/chrome",
            1.0,
            1 << 30,
        );
        b.app_icon = "google-chrome".into();
        app.gui.procs = vec![
            a,
            b,
            row(1, "systemd", "/sbin/init", 0.1, 12 << 20),
            row(900, "strata-host", "strata --serve", 3.0, 50 << 30),
        ];
        // default sort is memory desc
        let texts = tab_pass(&ctx, &mut app);
        let y = |needle: &str| {
            texts
                .iter()
                .filter(|(t, _)| t.trim() == needle)
                .map(|(_, y)| *y)
                .fold(f32::MAX, f32::min)
        };
        assert!(
            y("strata-host") < y("chrome  ·  2 processes"),
            "desc memory: 50 GiB standalone must paint above the 6 GiB app group; painted: {:?}",
            texts.iter().map(|(t, _)| t.trim()).collect::<Vec<_>>()
        );
        // flip to ascending: strata-host drops to the bottom of the table
        let mem = header_point(&ctx, &mut app, SortCol::Mem);
        click(&ctx, &mut app, mem);
        assert!(!app.gui.proc_sort_desc, "second header click flips to asc");
        let texts = tab_pass(&ctx, &mut app);
        let y = |needle: &str| {
            texts
                .iter()
                .filter(|(t, _)| t.trim() == needle)
                .map(|(_, y)| *y)
                .fold(0.0_f32, f32::max)
        };
        assert!(
            y("chrome  ·  2 processes") < y("strata-host"),
            "asc memory: strata-host sinks below the app groups"
        );
    }

    /// Regression for the reported "I can't really see the process list,
    /// it just shows No process selected": at the app's minimum window
    /// size, with real /proc data, the table must still paint rows —
    /// never an empty state, never a page where the only text is the
    /// detail placeholder.
    #[test]
    fn processes_page_shows_rows_at_the_minimum_window_size() {
        let _probe = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (ctx, mut app) = procs_probe();
        // the window manager pins this app at 960x400; that's the worst case
        let texts = page_pass(&ctx, &mut app, 960.0, 400.0);
        let all: Vec<&str> = texts.iter().map(|(t, _)| t.as_str()).collect();
        assert!(
            !all.iter().any(|t| t.contains("No process data yet")),
            "table must have data at 960x400"
        );
        // the header row and at least a handful of real process names paint
        let header = |needle: &str| {
            texts
                .iter()
                .find(|(t, _)| t.trim().starts_with(needle))
                .unwrap_or_else(|| {
                    panic!(
                        "header {needle:?} missing; painted: {:?}",
                        texts.iter().map(|(t, _)| t).take(40).collect::<Vec<_>>()
                    )
                })
                .1
        };
        header("Name");
        header("PID");
        header("Memory");
        let painted_names = texts
            .iter()
            .filter(|(t, _)| {
                let t = t.trim();
                !t.is_empty()
                    && !t.contains(char::is_whitespace)
                    && t.chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
            })
            .count();
        assert!(
            painted_names >= 5,
            "several process rows visible, got {painted_names}"
        );
        // the search bar is on the page, above the table
        let y = |needle: &str| {
            texts
                .iter()
                .find(|(t, _)| t.contains(needle))
                .unwrap_or_else(|| panic!("expected {needle:?} on the page"))
                .1
        };
        assert!(y("Search name") < y("Name"), "search sits above the table");
        // and the empty-selection hint must NOT swallow the page: it sits
        // below the table, never instead of it
        if let Some((_, dy)) = texts.iter().find(|(t, _)| t.contains("Click a row above")) {
            assert!(*dy > y("Name"), "hint below the table, not replacing it");
        }
    }

    /// The /proc walk runs on every screen, not just when the Processes tab
    /// is open, so the nav-rail count and the dashboard card are never blank.
    #[test]
    fn process_scan_is_gated_on_refresh_only_not_on_the_visible_tab() {
        let ctx = egui::Context::default();
        let (tx, rx) = mpsc::unbounded_channel();
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
                .unwrap(),
        );
        let mut app = SysKamlingApp::new_offscreen(&ctx, tx, rx, rt.clone());
        app.gui.tab = Tab::Dashboard;
        app.refresh = 1.0;
        app.last_proc_scan = Instant::now() - Duration::from_secs(999);
        app.spawn_proc_scan();
        // the scan task runs on the blocking pool; drain the app's inbox
        // until the ProcsScanned message lands
        let got = rt.block_on(async {
            tokio::time::timeout(Duration::from_secs(10), async {
                loop {
                    match app.rx.recv().await {
                        Some(Msg::ProcsScanned(_)) => return true,
                        Some(_) => continue,
                        None => return false,
                    }
                }
            })
            .await
        });
        assert!(
            matches!(got, Ok(true)),
            "a scan must be scheduled while on the Dashboard tab"
        );
    }

    /// The dashboard carries a Processes card that names the hogs, and
    /// clicking it opens the Processes tab.
    #[test]
    fn dashboard_processes_card_shows_hogs_and_opens_the_tab() {
        let _probe = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let ctx = egui::Context::default();
        let (tx, rx) = mpsc::unbounded_channel();
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
        );
        let mut app = SysKamlingApp::new_offscreen(&ctx, tx, rx, rt);
        app.gui.tab = Tab::Dashboard;
        app.gui.procs = vec![
            row(1, "systemd", "/sbin/init", 0.1, 12 << 20),
            row(900, "hog", "hog --big", 71.0, 9 << 30),
            row(901, "busy", "busy-loop", 99.0, 100 << 20),
        ];
        let texts = offscreen_pass(&ctx, &mut app, (1280.0, 860.0), |a, ui| {
            a.dashboard_view(ui);
        });
        let painted: Vec<&str> = texts.iter().map(|(t, _)| t.as_str()).collect();
        assert!(
            painted.iter().any(|t| t.contains("processes running")),
            "processes card headline missing"
        );
        for hog in ["busy", "hog"] {
            assert!(
                painted.iter().any(|t| t.contains(hog)),
                "processes card never names the {hog} hog"
            );
        }
        // clicking the Processes card navigates to the Processes tab
        let card = dash_card_rects()
            .into_iter()
            .find(|(_, _, title)| title == "Processes")
            .expect("dashboard has no Processes card");
        offscreen_click(&ctx, &mut app, (1280.0, 860.0), card.1.center(), |a, ui| {
            a.dashboard_view(ui)
        });
        assert_eq!(
            app.gui.tab,
            Tab::Processes,
            "card click must open Processes"
        );
    }

    /// The dashboard also carries a Hardware card with the live host
    /// vitals; before the first sample lands it must say so, and clicking
    /// it opens the Hardware tab.
    #[test]
    fn dashboard_hardware_card_shows_vitals_and_opens_the_tab() {
        let _probe = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let ctx = egui::Context::default();
        let (tx, rx) = mpsc::unbounded_channel();
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
        );
        let mut app = SysKamlingApp::new_offscreen(&ctx, tx, rx, rt);
        app.gui.tab = Tab::Dashboard;
        // no sample yet: the card must state the placeholder, not crash
        let texts = offscreen_pass(&ctx, &mut app, (1280.0, 1600.0), |a, ui| {
            a.dashboard_view(ui);
        });
        let painted: Vec<&str> = texts.iter().map(|(t, _)| t.as_str()).collect();
        assert!(
            painted.iter().any(|t| t.contains("Collecting host vitals")),
            "hardware card placeholder missing, got {painted:?}"
        );
        // a real sample now: the card paints live cpu/load values
        app.sample_hardware();
        app.sample_hardware(); // two samples so rates have a delta
                               // tall viewport: the overview cards sit at the bottom of the
                               // dashboard scroll area and must be visible to be painted
        let texts = offscreen_pass(&ctx, &mut app, (1280.0, 1600.0), |a, ui| {
            a.dashboard_view(ui);
        });
        let painted: Vec<&str> = texts.iter().map(|(t, _)| t.as_str()).collect();
        // the overview card paints live load/uptime and ↓/↑ rate rows
        assert!(
            painted
                .iter()
                .any(|t| t.contains("load ") && t.contains("up ")),
            "hardware card must show load and uptime, got {painted:?}"
        );
        assert!(
            painted.iter().any(|t| t.contains("↓ ") && t.contains("↑ ")),
            "hardware card must show disk/network rates, got {painted:?}"
        );
        // clicking the card navigates: the click pass re-renders the whole
        // dashboard; the sample before the click adds a health-strip issue
        // (load above core count) that shifts the layout, so take the rect
        // from a pass with the exact same data state as the click pass
        app.sample_hardware();
        let card = last_dash_cards()
            .into_iter()
            .find(|(_, _, title)| title == "Hardware")
            .expect("dashboard has no Hardware card");
        offscreen_click(
            &ctx,
            &mut app,
            (1280.0, 1600.0),
            card.1.center(),
            |a, ui| a.dashboard_view(ui),
        );
        assert_eq!(app.gui.tab, Tab::Hardware, "card click must open Hardware");
    }

    /// A click on the Processes card opens the Processes tab — the nav
    /// bookkeeping must not let a later card steal an earlier card's click.
    #[test]
    fn dashboard_processes_card_click_opens_processes() {
        let _probe = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let ctx = egui::Context::default();
        let (tx, rx) = mpsc::unbounded_channel();
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
        );
        let mut app = SysKamlingApp::new_offscreen(&ctx, tx, rx, rt);
        app.gui.tab = Tab::Dashboard;
        app.gui.procs = vec![row(1, "systemd", "/sbin/init", 0.1, 12 << 20)];
        offscreen_pass(&ctx, &mut app, (1280.0, 1600.0), |a, ui| {
            a.dashboard_view(ui);
        });
        let card = last_dash_cards()
            .into_iter()
            .find(|(_, _, title)| title == "Processes")
            .expect("dashboard has no Processes card");
        offscreen_click(
            &ctx,
            &mut app,
            (1280.0, 1600.0),
            card.1.center(),
            |a, ui| a.dashboard_view(ui),
        );
        assert_eq!(
            app.gui.tab,
            Tab::Processes,
            "Processes card click must open Processes, not the card painted after it"
        );
    }
    /// The nav rail lists Processes between Hardware and Ports, and clicking
    /// its entry opens the tab.
    #[test]
    fn nav_lists_processes_between_hardware_and_ports() {
        let _probe = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let ctx = egui::Context::default();
        let (tx, rx) = mpsc::unbounded_channel();
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
        );
        let mut app = SysKamlingApp::new_offscreen(&ctx, tx, rx, rt);
        // rail order is asserted through the probe sink: nav_row_rect finds
        // each entry by its position in the rail, so a wrong order would
        // land the click on a different tab
        let hw = nav_row_rect(&ctx, &mut app, Tab::Hardware);
        let procs = nav_row_rect(&ctx, &mut app, Tab::Processes);
        let ports = nav_row_rect(&ctx, &mut app, Tab::Ports);
        assert!(
            hw.top() < procs.top() && procs.top() < ports.top(),
            "Processes must sit between Hardware and Ports in the rail"
        );
        // click the rail: paint it once so its widget rects exist, then
        // press inside the rail (x = 110) at the probe's row height
        let click_at = egui::pos2(110.0, procs.center().y);
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1280.0, 860.0),
        ));
        ctx.begin_pass(raw.clone());
        app.nav_rail(&ctx);
        let _ = ctx.end_pass();
        raw.events = vec![
            egui::Event::PointerMoved(click_at),
            egui::Event::PointerButton {
                pos: click_at,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerButton {
                pos: click_at,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ];
        ctx.begin_pass(raw);
        app.nav_rail(&ctx);
        let _ = ctx.end_pass();
        assert_eq!(
            app.gui.tab,
            Tab::Processes,
            "rail click must open Processes"
        );
    }

    /// Regression for the reported "still not showing any icons": opening
    /// the Processes tab must (1) render the table full-width in the centre
    /// panel, (2) queue real-icon decodes, and (3) paint an actual image
    /// texture in a row once the decode lands — all through the real
    /// on_tab_change/center/spawn plumbing, not a hand-inserted texture.
    #[test]
    fn opening_processes_tab_queues_and_paints_real_icons_end_to_end() {
        // this box runs the app's own scanner; its .desktop table must name
        // at least a few icons for the icon queue to have anything to do
        let probe = ProcScanner::new().scan();
        let named = probe.iter().filter(|r| !r.app_icon.is_empty()).count();
        if named < 3 {
            eprintln!("skip: this machine names only {named} desktop icons");
            return;
        }
        let _probe = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let ctx = egui::Context::default();
        let (tx, mut rx) = mpsc::unbounded_channel();
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .enable_time()
                .build()
                .unwrap(),
        );
        // hand the app a throwaway receiver; we keep `rx` so we can watch
        // every message its sender produces and feed it back via apply()
        let (_sink_tx, sink_rx) = mpsc::unbounded_channel();
        let mut app = SysKamlingApp::new_offscreen(&ctx, tx, sink_rx, Arc::clone(&rt));
        app.gui.tab = Tab::Dashboard;
        app.on_tab_change(Tab::Processes);
        assert_eq!(app.gui.tab, Tab::Processes);

        // the scan the tab-open seeded must deliver live rows
        let rows = rt.block_on(async {
            loop {
                let m = rx.recv().await.unwrap();
                let rows = match &m {
                    Msg::ProcsScanned(r) => Some(r.clone()),
                    _ => None,
                };
                // feed the real handler, exactly as update() does: this is
                // what queues the icon decodes behind the scan
                app.apply(m, &ctx);
                if let Some(r) = rows {
                    break r;
                }
            }
        });
        assert!(!rows.is_empty(), "the seeded scan produced no rows");
        let named: Vec<&ProcInfo> = rows.iter().filter(|r| !r.app_icon.is_empty()).collect();
        assert!(
            named.len() >= 3,
            "live scan named no desktop icons: {} rows",
            rows.len()
        );

        // the icon decodes the scan triggered must come back as pixels and
        // land in the app's texture cache: pump every message through the
        // app's real handler, exactly as update() does, until three
        // textures are in (cap the wait so a stall fails instead of hangs)
        let want = named.len().min(40);
        let deadline = Instant::now() + Duration::from_secs(20);
        let mut decoded_msgs = 0;
        while app.gui.icons.tex_count() < 3 && decoded_msgs < want && Instant::now() < deadline {
            match rx.try_recv() {
                Ok(m) => {
                    if matches!(m, Msg::IconReady { .. }) {
                        decoded_msgs += 1;
                    }
                    app.apply(m, &ctx);
                }
                Err(mpsc::error::TryRecvError::Empty) => {
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                Err(mpsc::error::TryRecvError::Disconnected) => break,
            }
        }
        assert!(
            app.gui.icons.tex_count() >= 3,
            "only {} icon textures reached the cache after {} \
             decode messages; {} rows, {} named icons: {:?}",
            app.gui.icons.tex_count(),
            decoded_msgs,
            rows.len(),
            named.len(),
            named
                .iter()
                .take(8)
                .map(|r| (&r.name, &r.app_icon))
                .collect::<Vec<_>>()
        );
        // keep only rows whose icon decoded, so every visible row must
        // paint a texture; then render the page and count the meshes
        let cached: Vec<ProcInfo> = app
            .gui
            .procs
            .iter()
            .filter(|r| app.gui.icons.get(&r.app_icon).is_some())
            .cloned()
            .collect();
        assert!(
            cached.len() >= 3,
            "expected 3+ cached icon holders, got {}",
            cached.len()
        );
        app.gui.procs = cached;
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1280.0, 860.0),
        ));
        ctx.begin_pass(raw);
        app.center(&ctx);
        let out = ctx.end_pass();
        let meshes = out
            .shapes
            .iter()
            .filter(|s| matches!(s.shape, egui::Shape::Mesh(_)))
            .count();
        assert!(
            meshes >= 3,
            "the table must paint decoded app icons, got {meshes} mesh shapes with {} cached textures",
            app.gui.icons.tex_count()
        );
    }

    /// The nav-rail Processes row must carry a real decoded app icon once
    /// one exists, and the drawn line-icon otherwise.
    #[test]
    fn nav_rail_processes_row_uses_a_real_icon_when_decoded() {
        let _probe = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (ctx, mut app) = procs_probe();
        let meshes_before = {
            let mut raw = egui::RawInput::default();
            raw.screen_rect = Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(1280.0, 860.0),
            ));
            ctx.begin_pass(raw);
            app.nav_rail(&ctx);
            ctx.end_pass()
                .shapes
                .iter()
                .filter(|s| matches!(s.shape, egui::Shape::Mesh(_)))
                .count()
        };
        assert_eq!(
            meshes_before, 0,
            "no decoded icons yet: rail must stay drawn"
        );
        // stand in for a finished decode
        let mut r = row(4343, "firefox", "/usr/lib/firefox/firefox", 3.0, 900 << 20);
        r.app_icon = "firefox".into();
        app.gui.procs.push(r);
        let img = egui::ColorImage::from_rgb([48, 48], &[128u8; 48 * 48 * 3]);
        let tex = ctx.load_texture("appicon:rail", img, egui::TextureOptions::LINEAR);
        app.gui.icons.insert("firefox", tex);
        let meshes_after = {
            let mut raw = egui::RawInput::default();
            raw.screen_rect = Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::Vec2::new(1280.0, 860.0),
            ));
            ctx.begin_pass(raw);
            app.nav_rail(&ctx);
            ctx.end_pass()
                .shapes
                .iter()
                .filter(|s| matches!(s.shape, egui::Shape::Mesh(_)))
                .count()
        };
        assert_eq!(
            meshes_after, 1,
            "the Processes rail row must paint the decoded icon"
        );
    }
}

#[cfg(test)]
mod settings_tab_tests {
    use super::*;

    fn settings_probe() -> (egui::Context, SysKamlingApp) {
        let ctx = egui::Context::default();
        let (tx, rx) = mpsc::unbounded_channel();
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
        );
        let mut app = SysKamlingApp::new_offscreen(&ctx, tx, rx, rt);
        app.gui.tab = Tab::Settings;
        (ctx, app)
    }

    /// Point $HOME at a throwaway directory (serialised against the config
    /// tests' lock) so "Test & save" can never touch the real config file.
    struct HomeGuard {
        original: Option<std::ffi::OsString>,
        dir: std::path::PathBuf,
        _lock: std::sync::MutexGuard<'static, ()>,
    }
    impl Drop for HomeGuard {
        fn drop(&mut self) {
            match &self.original {
                Some(h) => std::env::set_var("HOME", h),
                None => std::env::remove_var("HOME"),
            }
            std::fs::remove_dir_all(&self.dir).ok();
        }
    }
    fn isolate_home() -> HomeGuard {
        let dir =
            std::env::temp_dir().join(format!("syskamling-gui-settings-{}", std::process::id()));
        std::fs::create_dir_all(dir.join(".config/syskamling")).unwrap();
        let lock = crate::strata::tests::HOME_TEST_LOCK
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let original = std::env::var_os("HOME");
        std::env::set_var("HOME", &dir);
        HomeGuard {
            original,
            dir,
            _lock: lock,
        }
    }

    fn paint(
        ctx: &egui::Context,
        app: &mut SysKamlingApp,
        events: Vec<egui::Event>,
    ) -> Vec<String> {
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1200.0, 800.0),
        ));
        raw.events = events;
        let mut texts = Vec::new();
        for _ in 0..2 {
            ctx.begin_pass(raw.clone());
            egui::CentralPanel::default().show(ctx, |ui| {
                if app.gui.tab == Tab::Settings {
                    app.settings_view(ui);
                }
            });
            let out = ctx.end_pass();
            for sh in &out.shapes {
                if let egui::Shape::Text(t) = &sh.shape {
                    texts.push(t.galley.text().to_owned());
                }
            }
        }
        texts
    }

    /// Typing an endpoint and pressing ⟳ Test & save must commit it to the
    /// live config — the whole point of moving the fields here.
    #[test]
    fn typing_endpoint_then_test_and_save_commits_it() {
        let _probe = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = isolate_home();
        let (ctx, mut app) = settings_probe();
        // Pass 1 paints the tab; the view records the endpoint field's rect
        // into the probe field (cfg(test) seam), so Pass 2 can click it.
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1200.0, 800.0),
        ));
        ctx.begin_pass(raw.clone());
        egui::CentralPanel::default().show(&ctx, |ui| app.settings_view(ui));
        let _ = ctx.end_pass();
        let click = app
            .gui
            .probe_endpoint_rect
            .expect("endpoint field not painted")
            .center();
        // Pass 2: type into the focused field, then read back the save
        // button's rect (its position depends on the typed text's layout).
        let type_events = vec![
            egui::Event::PointerButton {
                pos: click,
                pressed: true,
                button: egui::PointerButton::Primary,
                modifiers: Default::default(),
            },
            egui::Event::PointerButton {
                pos: click,
                pressed: false,
                button: egui::PointerButton::Primary,
                modifiers: Default::default(),
            },
            egui::Event::Text("http://10.0.0.9:9999".into()),
        ];
        raw.events = type_events;
        ctx.begin_pass(raw.clone());
        egui::CentralPanel::default().show(&ctx, |ui| app.settings_view(ui));
        let _ = ctx.end_pass();
        assert_eq!(
            app.gui.strata_endpoint_edit, "http://10.0.0.9:9999",
            "typing never reached the field"
        );
        let save_btn = app
            .gui
            .probe_save_rect
            .expect("test & save button not painted")
            .center();
        let events = vec![
            egui::Event::PointerButton {
                pos: save_btn,
                pressed: true,
                button: egui::PointerButton::Primary,
                modifiers: Default::default(),
            },
            egui::Event::PointerButton {
                pos: save_btn,
                pressed: false,
                button: egui::PointerButton::Primary,
                modifiers: Default::default(),
            },
        ];
        let texts = paint(&ctx, &mut app, events);
        assert_eq!(app.gui.strata_cfg.endpoint, "http://10.0.0.9:9999");
        let all = texts.join("\n");
        assert!(
            all.contains("Strata server"),
            "settings card title never painted"
        );
    }

    /// Layout sanity for the redesigned tab: the Test & save button must sit
    /// tight under the key field (no dead gap), and the whole card column
    /// must fit a short window without sideways overflow.
    #[test]
    fn save_button_sits_close_under_the_fields() {
        let _probe = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = isolate_home();
        let (ctx, mut app) = settings_probe();
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1100.0, 420.0),
        ));
        ctx.begin_pass(raw.clone());
        egui::CentralPanel::default().show(&ctx, |ui| app.settings_view(ui));
        let _ = ctx.end_pass();
        let key = app.gui.probe_endpoint_rect.expect("endpoint field");
        let save = app.gui.probe_save_rect.expect("save button");
        // second field sits ~10px below the first; the button ~12px below that
        let gap = save.top() - (key.bottom() + 48.0); // 48 ≈ key field height+spacing
        assert!(
            gap.abs() < 20.0,
            "dead gap between fields and save: {gap:.1}px"
        );
        assert!(
            save.left() >= key.left() - 1.0,
            "save button not aligned to card"
        );
    }

    /// Changing the refresh interval must retune the poller at once.
    #[test]
    fn refresh_interval_drag_takes_effect() {
        let _probe = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _home = isolate_home();
        let (ctx, mut app) = settings_probe();
        app.gui.refresh_edit = 7.5;
        let events = vec![
            egui::Event::PointerButton {
                pos: egui::pos2(60.0, 120.0),
                pressed: true,
                button: egui::PointerButton::Primary,
                modifiers: Default::default(),
            },
            egui::Event::Key {
                key: egui::Key::Enter,
                pressed: true,
                repeat: false,
                modifiers: Default::default(),
                physical_key: None,
            },
        ];
        paint(&ctx, &mut app, events);
        assert!(
            (app.refresh - 7.5).abs() < 1e-6,
            "refresh not applied: {}",
            app.refresh
        );
    }
}

#[cfg(test)]
mod hw_scroll_tests {
    use super::*;

    #[test]
    fn hardware_page_body_scrolls_instead_of_clipping() {
        let _probe = PROBE_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let (viewport_h, content_h) =
            SysKamlingApp::probe_hardware_scroll(egui::Vec2::new(1100.0, 420.0));
        assert!(viewport_h > 0.0, "probe never ran");
        // the body must scroll, and it must not be collapsed to nothing:
        // the KPI card stack alone always exceeds the viewport height,
        // even on a headless runner where sensors/gpu sections are absent
        assert!(
            content_h > viewport_h,
            "content {content_h} must exceed viewport {viewport_h} so the body scrolls"
        );
        assert!(
            content_h > 420.0,
            "content {content_h} looks collapsed/blank"
        );
    }
}

#[cfg(test)]
#[cfg(test)]
pub mod hw_scroll_probe {
    use std::sync::atomic::{AtomicU32, Ordering};
    // heights are stored as f32 bit patterns scaled through from_bits
    static MAX_RECT: AtomicU32 = AtomicU32::new(0);
    static CONTENT_H: AtomicU32 = AtomicU32::new(0);

    pub fn reset() {
        MAX_RECT.store(0.0f32.to_bits(), Ordering::Relaxed);
        CONTENT_H.store(0.0f32.to_bits(), Ordering::Relaxed);
    }
    pub fn record(viewport_h: f32, content_h: f32) {
        MAX_RECT.store(viewport_h.to_bits(), Ordering::Relaxed);
        CONTENT_H.store(content_h.to_bits(), Ordering::Relaxed);
    }
    pub fn last() -> (f32, f32) {
        (
            f32::from_bits(MAX_RECT.load(Ordering::Relaxed)),
            f32::from_bits(CONTENT_H.load(Ordering::Relaxed)),
        )
    }
}

#[cfg(test)]
mod hw_titles_tests {
    use super::*;

    /// Paint the hardware page against a fixture inventory and a fixture
    /// usage table, and return every galley the pass produced.
    fn paint_hw_page(
        ctx: &egui::Context,
        app: &mut SysKamlingApp,
        inv: &hwinv::Inventory,
        usage: &[hwinv::DiskUsage],
    ) -> String {
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(1200.0, 900.0),
        ));
        let mut all = String::new();
        for _ in 0..3 {
            ctx.begin_pass(raw.clone());
            egui::CentralPanel::default().show(ctx, |ui| {
                if app.gui.tab == Tab::Hardware {
                    app.hardware_view(ui, inv, usage)
                }
            });
            let out = ctx.end_pass();
            all = out
                .shapes
                .iter()
                .filter_map(|cs| match &cs.shape {
                    egui::Shape::Text(t) => Some(t.galley.text().to_owned()),
                    _ => None,
                })
                .collect::<Vec<String>>()
                .join("\n");
        }
        all
    }

    fn hw_app(ctx: &egui::Context) -> SysKamlingApp {
        let (tx, rx) = mpsc::unbounded_channel();
        let rt = Arc::new(
            tokio::runtime::Builder::new_current_thread()
                .build()
                .unwrap(),
        );
        let mut app = SysKamlingApp::new_offscreen(ctx, tx, rx, rt);
        app.gui.tab = Tab::Hardware;
        app.sample_hardware();
        app.sample_hardware();
        app
    }

    #[test]
    fn hardware_page_names_devices_and_hides_internal_disks() {
        // fixture-driven: the naming and hiding rules must hold on any box,
        // including headless CI runners with no inventoried disks at all
        let inv = hwinv::fixture_inventory();
        let usage = [hwinv::DiskUsage {
            kern: "testdisk0".into(),
            mount: "/".into(),
            total: 1000,
            used: 400,
        }];
        let ctx = egui::Context::default();
        let mut app = hw_app(&ctx);
        let mut rich = app.gui.hw.clone().unwrap();
        // the fixture drive plus an internal OEM disk the inventory never
        // named — the storage section must show the first, hide the second
        rich.disks = vec![
            crate::hw::DiskStat {
                name: "testdisk0".into(),
                reads: 1,
                writes: 1,
                total_read: 1,
                total_write: 1,
                ms_weighted: 0,
                uptime: 3600.0,
                temp_c: Some(41.0),
                read_bps: 0.0,
                write_bps: 0.0,
            },
            crate::hw::DiskStat {
                name: "sda".into(),
                reads: 1,
                writes: 1,
                total_read: 1,
                total_write: 1,
                ms_weighted: 0,
                uptime: 3600.0,
                temp_c: None,
                read_bps: 0.0,
                write_bps: 0.0,
            },
        ];
        app.gui.hw = Some(rich);
        let all = paint_hw_page(&ctx, &mut app, &inv, &usage);
        // retired sections must be gone
        assert!(!all.contains("Machine"), "machine section still present");
        assert!(
            !all.contains("Per-core load"),
            "per-core section still present"
        );
        // the CPU card names the inventory's processor
        assert!(all.contains(&inv.cpu), "cpu card never names {}", inv.cpu);
        // the fixture drive renders its card under its inventory name
        // ("Test Disk"), never the bare kernel name
        assert!(all.contains("Test Disk"), "fixture disk card never named");
        assert!(
            !all.split('\n').any(|t| t.trim() == "testdisk0"),
            "kernel name leaked as a card title"
        );
        // internal sda gets no storage card of its own
        assert!(
            !all.split('\n').any(|t| t.trim() == "sda"),
            "internal sda still shown"
        );
    }

    #[test]
    fn storage_cards_show_capacity_bars_and_swap_card_stands_alone() {
        // fixture usage: one filesystem on the fixture drive, so the bar
        // and its mount label render deterministically on any runner
        let inv = hwinv::fixture_inventory();
        let usage = [hwinv::DiskUsage {
            kern: "testdisk0".into(),
            mount: "/data".into(),
            total: 1000,
            used: 400,
        }];
        let ctx = egui::Context::default();
        let mut app = hw_app(&ctx);
        let mut rich = app.gui.hw.clone().unwrap();
        rich.disks = vec![crate::hw::DiskStat {
            name: "testdisk0".into(),
            reads: 1,
            writes: 1,
            total_read: 1,
            total_write: 1,
            ms_weighted: 0,
            uptime: 3600.0,
            temp_c: None,
            read_bps: 0.0,
            write_bps: 0.0,
        }];
        app.gui.hw = Some(rich);
        let all = paint_hw_page(&ctx, &mut app, &inv, &usage);
        // the fixture drive's card carries its mounted filesystem's usage
        assert!(
            all.contains("/data"),
            "storage card never shows /data usage"
        );
        assert!(
            all.contains("400 B of 1000 B"),
            "capacity bar never shows its fill"
        );
        // the retired bits stay retired
        assert!(!all.contains("Active"), "utilization bar still rendered");
        assert!(
            !all.contains("Memory & swap"),
            "memory+swap combo card still present"
        );
        assert!(all.contains("Swap"), "swap card missing");
    }
}
