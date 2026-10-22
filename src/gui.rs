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

use crate::agents::{AgentScanner, AgentSession};
use crate::docker::{
    self, ContainerSummary, DiskSummary, Docker, DockerError, EventMessage, ImageSummary,
    LogLine, StatsSample, StreamKind, SystemInfo,
};
use crate::hw::{HwSample, Sampler};
use crate::hw_inventory as hwinv;
use crate::hw_inventory::{gpu_stats, gpu_stats_hwmon_only, GpuStat};
use crate::ports::{self as portsmod, PortEntry};
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
    selected_agent: Option<usize>,
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
                    || a.cwd.as_deref().map(|c| c.to_lowercase()).unwrap_or_default().contains(&f)
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
                f.is_empty()
                    || p.proto.contains(&f)
                    || p.addr.contains(&f)
                    || p.port.to_string().contains(&f)
                    || p.proc_name.as_deref().map(|n| n.to_lowercase()).unwrap_or_default().contains(&f)
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
        push(&mut self.hw_swap_hist, s.swap_total.saturating_sub(s.swap_free) as f64);
        push(&mut self.hw_disk_r_hist, s.disk_read_bps);
        push(&mut self.hw_disk_w_hist, s.disk_write_bps);
        push(&mut self.hw_net_rx_hist, s.net_rx_bps);
        push(&mut self.hw_net_tx_hist, s.net_tx_bps);
        push_opt(&mut self.hw_cpu_temp_hist, s.cpu_temp_c);
        // GPU aggregates: busiest GPU busy%, most-used VRAM across GPUs,
        // hottest GPU/SSD sensor — one row each regardless of GPU count.
        let busy = gpus.iter().filter_map(|g| g.busy_pct).fold(None::<f64>, |a, b| Some(a.map_or(b, |a| a.max(b))));
        let vram = gpus.iter().map(|g| g.vram_used as f64).fold(0.0_f64, f64::max);
        let gpu_temp = gpus.iter().filter_map(|g| g.temp_c).fold(None::<f64>, |a, b| Some(a.map_or(b, |a| a.max(b))));
        let disk_temp = s.disks.iter().filter_map(|d| d.temp_c).fold(None::<f64>, |a, b| Some(a.map_or(b, |a| a.max(b))));
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
        self.logs.sort_by(|a, b| ts_of(&a.text).cmp(&ts_of(&b.text)));
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
    // --- hardware tab ---
    sampler: Sampler,
    last_hw_sample: Instant,
    // --- ports tab ---
    last_port_scan: Instant,
    port_scan_busy: Arc<Mutex<bool>>,
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
            sampler: Sampler::default(),
            last_hw_sample: Instant::now() - Duration::from_secs(999),
            last_port_scan: Instant::now() - Duration::from_secs(999),
            port_scan_busy: Arc::new(Mutex::new(false)),
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
            sampler: Sampler::default(),
            last_hw_sample: Instant::now() - Duration::from_secs(999),
            last_port_scan: Instant::now() - Duration::from_secs(999),
            port_scan_busy: Arc::new(Mutex::new(false)),
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
        let rt = Arc::new(tokio::runtime::Builder::new_current_thread().build().unwrap());
        let mut app = Self::new_offscreen(&ctx, tx, rx, rt);
        app.gui.tab = Tab::Hardware;
        app.sample_hardware();
        app.sample_hardware(); // second sample so rates/sparklines populate
        let mut rich = app.gui.hw.clone().unwrap();
        rich.disks = (0..8)
            .map(|i| crate::hw::DiskStat {
                name: format!("nvme{}n1", i),
                reads: 10,
                writes: 20,
                total_read: 1000,
                total_write: 2000,
                ms_busy: 50,
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
        for _ in 0..3 {
            ctx.begin_pass(raw.clone());
            egui::CentralPanel::default().show(&ctx, |ui| match app.gui.tab {
                Tab::Hardware => app.hardware_view(ui),
                _ => {}
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
    w.open = w.active.clone();
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
    style.text_styles.insert(TextStyle::Body, FontId::proportional(14.0));
    style.text_styles.insert(TextStyle::Button, FontId::proportional(13.5));
    style.text_styles.insert(TextStyle::Monospace, FontId::monospace(12.5));
    style.text_styles.insert(TextStyle::Small, FontId::proportional(12.0));

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
            .chain(files.iter().map(|f| format!("/usr/share/fonts/truetype/hack/{f}")))
            .find_map(|p| std::fs::read(p).ok())?;
        fonts
            .font_data
            .insert(key.to_string(), FontData::from_owned(bytes).into());
        Some(key.to_string())
    };
    let ui = load("syskamling-ui", &["NotoSans-Regular.ttf", "DejaVuSans.ttf"]);
    let ui_bold = load("syskamling-bold", &["NotoSans-Bold.ttf", "DejaVuSans-Bold.ttf"]);
    let mono = load(
        "syskamling-mono",
        &["Hack-Regular.ttf", "DejaVuSansMono.ttf", "NotoSansMono-Regular.ttf"],
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
    fonts.families.insert(FontFamily::Name("bold".into()), bold_fam);
    fonts.families.insert(FontFamily::Monospace, mono_fam);
    ctx.set_fonts(fonts);
}

impl eframe::App for SysKamlingApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain all pending engine messages.
        while let Ok(m) = self.rx.try_recv() {
            self.apply(m, ctx);
        }

        // Periodic refresh + continuous repaint at ~4 fps when idle so the
        // list stays live without burning CPU.
        if self.tick.elapsed() >= Duration::from_millis(if self.refresh > 0.0 { 250 } else { 100 }) {
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
            self.spawn_strata_poll();
            // nvidia-smi (~50 ms) runs off the UI thread only when the
            // Hardware tab is visible; the result lands via the channel.
            if self.gui.tab == Tab::Hardware && self.last_gpu_stats.elapsed() >= Duration::from_secs_f64(self.refresh.max(1.0)) {
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
        let tx = self.tx.clone();
        self.rt.spawn_blocking(move || {
            let mut scanner = AgentScanner::default();
            let sessions = scanner.scan();
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
        let nvidia = self.gui.gpus.iter().filter(|g| g.driver == "nvidia").cloned().collect::<Vec<_>>();
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

    /// Strata monitor poll: one /health + /metrics round-trip per refresh,
    /// only while the Strata tab or the dashboard is open, on the blocking pool so a slow or
    /// unreachable server never touches the UI thread.
    fn spawn_strata_poll(&mut self) {
        self.spawn_strata_poll_inner(false)
    }

    /// `force` bypasses the visible-tab gate (used for the first pull when
    /// the tab opens and right after settings change).
    fn spawn_strata_poll_inner(&mut self, force: bool) {
        if (!force && !matches!(self.gui.tab, Tab::Strata | Tab::Dashboard)) || self.refresh <= 0.0 {
            return;
        }
        if !force && self.last_strata_poll.elapsed() < Duration::from_secs_f64(self.refresh.max(1.0)) {
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
                Ok((200, body)) => serde_json::from_slice(&body).map_err(|e| format!("/health: bad JSON: {e}")),
                Ok((s, _)) => Err(format!("/health: HTTP {s}")),
                Err(e) => Err(e),
            };
            let _ = tx.send(Msg::StrataHealth(h));
            let m = match strata::get_blocking(&cfg, "/metrics") {
                Ok((200, body)) => strata::parse_metrics(&body),
                Ok((401, _)) => Err("the server needs an API key (401) — set it in the fields above".into()),
                Ok((s, body)) => {
                    let text = String::from_utf8_lossy(&body).chars().take(160).collect::<String>();
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
            if self.gui.pending_action.as_ref().is_some_and(|(i, _)| i == id) {
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
                        self.all_logs_follow.retain(|id| {
                            want.iter().any(|(wid, _)| wid == id)
                        });
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
                // keep the selection pointing at the same session
                if let Some(key) = g
                    .selected_agent
                    .and_then(|i| g.agents.get(i))
                    .map(|a| (a.agent, a.lead_pid))
                {
                    g.selected_agent =
                        g.agents.iter().position(|a| a.agent == key.0 && a.lead_pid == key.1);
                }
            }
            Msg::PortsScanned(entries) => {
                g.ports = entries;
                // keep the selection pointing at the same socket
                if let Some(key) = g.selected_port.clone() {
                    g.selected_port = g
                        .ports
                        .iter()
                        .find(|p| (p.proto, p.addr.as_str(), p.port) == (key.0.as_str(), key.1.as_str(), key.2))
                        .map(|p| (p.proto.to_string(), p.addr.clone(), p.port));
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
        let mut new_tab: Option<Tab> = None;
        let mut refresh_now = false;
        let g = &self.gui;
        let n_run = g.containers.iter().filter(|c| c.state == "running").count();
        let items = [
            (Tab::Dashboard, "Dashboard", String::new()),
            (Tab::Containers, "Containers", format!("{n_run}/{}", g.containers.len())),
            (Tab::Images, "Images", g.images.len().to_string()),
            (Tab::Volumes, "Volumes", g.volumes.len().to_string()),
            (Tab::Agents, "Agents", g.agents.len().to_string()),
            (Tab::Hardware, "Hardware", String::new()),
            (Tab::Ports, "Ports", g.ports.len().to_string()),
            (Tab::Strata, "Strata", String::new()),
            (Tab::Logs, "Logs", String::new()),
        ];
        let sys_items = [(Tab::Settings, "Settings", String::new())];
        let version = g.info.as_ref().map(|i| i.server_version.clone()).unwrap_or_default();
        let synced = self.last_sync.map(|t| t.elapsed().as_secs());
        egui::SidePanel::left("nav")
            .exact_width(216.0)
            .resizable(false)
            .frame(Frame::NONE.fill(RAIL).inner_margin(Margin::symmetric(12, 18)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.add_space(4.0);
                    let (r, _) = ui.allocate_exact_size(Vec2::splat(32.0), Sense::hover());
                    logo_mark(ui.painter(), r);
                    ui.vertical(|ui| {
                        ui.spacing_mut().item_spacing.y = 0.0;
                        ui.label(RichText::new("SysKamling").font(bold(15.0)).color(TEXT));
                        ui.label(RichText::new("Developer system monitor").size(11.5).color(FAINT));
                    });
                });
                ui.add_space(16.0);
                engine_card(ui, g.daemon_ok, &version);
                ui.add_space(20.0);
                section_label(ui, "WORKSPACE");
                ui.add_space(2.0);
                ui.spacing_mut().item_spacing.y = 2.0;
                for (tab, label, badge) in &items {
                    if nav_item(ui, *tab, label, badge, g.tab == *tab) {
                        new_tab = Some(*tab);
                    }
                }
                ui.add_space(14.0);
                section_label(ui, "SYSTEM");
                ui.add_space(2.0);
                for (tab, label, badge) in &sys_items {
                    if nav_item(ui, *tab, label, badge, g.tab == *tab) {
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
                    Tab::Ports => ("Ports", self.gui.ports.len(), "Search port, process or address"),
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
                    let mut owned = self.gui.ports_owned_only;
                    if ui
                        .checkbox(&mut owned, RichText::new("Owned by a process only").size(12.5).color(MUTED))
                        .changed()
                    {
                        self.gui.ports_owned_only = owned;
                    }
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
                    let (rect, resp) = ui.allocate_exact_size(
                        Vec2::new(ui.available_width(), 58.0),
                        Sense::click(),
                    );
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
                        pill_at(p, Pos2::new(rect.right() - 12.0, rect.center().y), &label, FAINT, SURFACE_HI);
                    }
                    let ports = c.port_summary();
                    resp.on_hover_text(format!(
                        "{}\n{}{}",
                        c.image,
                        c.status,
                        if ports.is_empty() { String::new() } else { format!("\n{ports}") }
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
                    let (rect, resp) = ui.allocate_exact_size(
                        Vec2::new(ui.available_width(), 58.0),
                        Sense::click(),
                    );
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
                    let name = elide(p, repo.to_string(), FontId::proportional(14.0), TEXT, text_w);
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
        let rows = self.gui.filtered_agents();
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
        let now = epoch();
        let mut new_sel: Option<Option<usize>> = None;
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = 4.0;
                for (i, a) in rows {
                    let (rect, resp) = ui.allocate_exact_size(
                        Vec2::new(ui.available_width(), 58.0),
                        Sense::click(),
                    );
                    let is_sel = sel == Some(i);
                    if resp.clicked() {
                        new_sel = Some(if is_sel { None } else { Some(i) });
                    }
                    if !ui.is_rect_visible(rect) {
                        continue;
                    }
                    let p = ui.painter();
                    row_bg(p, rect, is_sel, resp.hovered());
                    let color = agent_color(a.agent);
                    icon_tile(p, rect, Tab::Agents, color);
                    let left = rect.left() + 54.0;
                    let text_w = (rect.width() - 54.0 - 86.0).max(40.0);
                    let name = elide(
                        p,
                        a.agent.to_string(),
                        FontId::proportional(14.0),
                        TEXT,
                        text_w,
                    );
                    p.galley(Pos2::new(left, rect.top() + 10.0), name, TEXT);
                    let proj = a
                        .cwd
                        .as_deref()
                        .map(basename)
                        .unwrap_or_else(|| a.vendor.to_string());
                    let sub = elide(
                        p,
                        format!("{} · pid {} · {}", proj, a.lead_pid, ago(now - a.started.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0))),
                        FontId::proportional(12.0),
                        MUTED,
                        text_w,
                    );
                    p.galley(Pos2::new(left, rect.top() + 32.0), sub, MUTED);
                    let busy = a.cpu_pct > 1.0;
                    p.text(
                        Pos2::new(rect.right() - 12.0, rect.center().y),
                        egui::Align2::RIGHT_CENTER,
                        format!("{:.0}%", a.cpu_pct),
                        FontId::monospace(12.0),
                        if busy { GREEN } else { FAINT },
                    );
                }
            });
        if let Some(s) = new_sel {
            self.gui.selected_agent = s;
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
                empty_state(ui, "No sockets match", "Try another search.");
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
                    let (rect, resp) =
                        ui.allocate_exact_size(Vec2::new(ui.available_width(), rect_h), Sense::click());
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
                    let sub = elide(
                        paint,
                        sub,
                        FontId::proportional(12.5),
                        MUTED,
                        text_w,
                    );
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
                    let (rect, resp) = ui.allocate_exact_size(
                        Vec2::new(ui.available_width(), 58.0),
                        Sense::click(),
                    );
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
                    icon_tile(p, rect, Tab::Volumes, if refs != Some(0) { VIOLET } else { FAINT });
                    let left = rect.left() + 54.0;
                    let text_w = (rect.width() - 54.0 - 86.0).max(40.0);
                    let name = elide(p, v.name.clone(), FontId::proportional(14.0), TEXT, text_w);
                    p.galley(Pos2::new(left, rect.top() + 10.0), name, TEXT);
                    let sub = elide(
                        p,
                        match refs {
                            Some(1) => format!("{} · 1 ref", v.driver),
                            Some(n) => format!("{} · {n} refs", v.driver),
                            None => format!("{} · created {}", v.driver, v.created.get(..10).unwrap_or("?")),
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
                    Tab::Hardware => self.hardware_view(ui),
                    Tab::Ports => self.port_detail(ui),
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
                            .add(egui::Button::new(RichText::new("Dismiss").size(12.5).color(MUTED)).frame(false))
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
        let hottest = |it: &mut dyn Iterator<Item = f64>| it.fold(None::<f64>, |a, b| Some(a.map_or(b, |a| a.max(b))));
        let mut issues: Vec<(Color32, String, Tab)> = Vec::new();
        if let Some(h) = &g.hw {
            let mem = (h.mem_total - h.mem_avail) as f64 / h.mem_total.max(1) as f64 * 100.0;
            if let Some(c) = level(mem, 80.0, 92.0) {
                issues.push((c, format!("Memory {mem:.0}% used · {} free", human_size(h.mem_avail as i64)), Tab::Hardware));
            }
            if h.swap_total > 0 {
                let swap = h.swap_total.saturating_sub(h.swap_free) as f64 / h.swap_total as f64 * 100.0;
                if let Some(c) = level(swap, 50.0, 85.0) {
                    issues.push((c, format!("Swap {swap:.0}% used"), Tab::Hardware));
                }
            }
            if h.load5 > h.ncpu as f64 {
                issues.push((AMBER, format!("Load {:.1} is above {} CPUs", h.load5, h.ncpu), Tab::Hardware));
            }
            let gpu_t = h.gpu_temp_c.or_else(|| hottest(&mut g.gpus.iter().filter_map(|x| x.temp_c)));
            let ssd_t = hottest(&mut h.disks.iter().filter_map(|d| d.temp_c));
            for (what, t, warn, bad) in [
                ("CPU", h.cpu_temp_c.or(h.temp_c), 80.0, 90.0),
                ("GPU", gpu_t, 80.0, 90.0),
                ("SSD", ssd_t, 60.0, 70.0),
            ] {
                if let Some((c, t)) = t.and_then(|t| level(t, warn, bad).map(|c| (c, t))) {
                    issues.push((c, format!("{what} running hot at {} °C", temp_str(t)), Tab::Hardware));
                }
            }
        }
        let restarting = g.containers.iter().filter(|c| c.state == "restarting").count();
        if restarting > 0 {
            issues.push((RED, format!("{restarting} container(s) stuck restarting"), Tab::Containers));
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
            issues.push((AMBER, format!("{crashed} container(s) exited with an error in the last hour"), Tab::Containers));
        }
        if !g.strata_cfg.api_key.trim().is_empty() && g.strata_error.is_some() {
            issues.push((AMBER, "Strata server is not responding".into(), Tab::Strata));
        }
        if let Some(df) = &g.df {
            let reclaim: i64 = [&df.image_usage, &df.container_usage, &df.volume_usage, &df.build_cache_usage]
                .into_iter()
                .flatten()
                .map(|u| u.reclaimable)
                .sum();
            if reclaim >= 10 << 30 {
                issues.push((ACCENT, format!("{} of Docker data can be reclaimed", human_size(reclaim)), Tab::Images));
            }
        }
        issues.sort_by_key(|(c, _, _)| [RED, AMBER, ACCENT].iter().position(|x| x == c));

        let mut nav = None;
        card(ui, |ui| {
            if issues.is_empty() {
                ui.horizontal(|ui| {
                    let (r, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
                    ui.painter().circle_filled(r.center(), 4.0, GREEN);
                    ui.label(RichText::new("All systems normal").font(bold(13.5)).color(TEXT));
                    ui.label(RichText::new("memory, temperatures, containers and Strata look healthy").size(12.5).color(FAINT));
                });
                return;
            }
            card_title(
                ui,
                "Needs attention",
                &format!("{} issue{}", issues.len(), if issues.len() == 1 { "" } else { "s" }),
            );
            ui.spacing_mut().item_spacing.y = 0.0;
            for (color, text, tab) in &issues {
                let (rect, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 30.0), Sense::click());
                let p = ui.painter();
                if resp.hovered() {
                    p.rect_filled(rect, 8.0, SURFACE_HI);
                    ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
                }
                p.circle_filled(Pos2::new(rect.left() + 10.0, rect.center().y), 4.0, *color);
                p.text(Pos2::new(rect.left() + 24.0, rect.center().y), egui::Align2::LEFT_CENTER, text, FontId::proportional(13.0), TEXT);
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
        let sum = |a: &VecDeque<f64>, b: &VecDeque<f64>| a.iter().zip(b.iter()).map(|(x, y)| x + y).collect::<Vec<f64>>();
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
                value: format!("{:.0}%", h.cpu_pct),
                sub: cpu_sub.join(" · "),
                color: ACCENT,
                viz: Viz::Spark(hist(&g.hw_cpu_hist)),
                glyph: Some("cpu"),
            },
            Kpi {
                label: "Memory",
                value: format!("{:.0}%", mem_used as f64 / h.mem_total.max(1) as f64 * 100.0),
                sub: format!(
                    "{} of {}{}",
                    human_size(mem_used as i64),
                    human_size(h.mem_total as i64),
                    if swap_used > 0 { format!(" · swap {}", human_size(swap_used as i64)) } else { String::new() }
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
            .max_by(|a, b| a.busy_pct.partial_cmp(&b.busy_pct).unwrap_or(std::cmp::Ordering::Equal));
        tiles.push(match gpu {
            Some(gp) => {
                let mut sub = Vec::new();
                if let Some(t) = gp.temp_c {
                    sub.push(format!("{} °C", temp_str(t)));
                }
                if gp.vram_total > 0 {
                    sub.push(format!("VRAM {} of {}", human_size(gp.vram_used as i64), human_size(gp.vram_total as i64)));
                }
                if sub.is_empty() {
                    sub.push(gp.name.clone());
                }
                Kpi {
                    label: "GPU",
                    value: format!("{:.0}%", gp.busy_pct.unwrap_or(0.0)),
                    sub: sub.join(" · "),
                    color: CYAN,
                    viz: Viz::Spark(hist(&g.hw_gpu_busy_hist)),
                    glyph: Some("gpu"),
                }
            }
            None => Kpi {
                label: "Disk I/O",
                value: human_rate(h.disk_read_bps + h.disk_write_bps),
                sub: format!("read {} · write {}", human_rate(h.disk_read_bps), human_rate(h.disk_write_bps)),
                color: AMBER,
                viz: Viz::Spark(sum(&g.hw_disk_r_hist, &g.hw_disk_w_hist)),
                glyph: Some("disk"),
            },
        });
        tiles.push(Kpi {
            label: "Network",
            value: human_rate(h.net_rx_bps + h.net_tx_bps),
            sub: format!("↓ {} · ↑ {}", human_rate(h.net_rx_bps), human_rate(h.net_tx_bps)),
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
        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.x = 14.0;
            ui.columns(2, |cols| {
                if overview_card(&mut cols[0], Tab::Containers, "Containers", |ui| self.containers_overview(ui)) {
                    nav = Some(Tab::Containers);
                }
                if overview_card(&mut cols[1], Tab::Strata, "Strata", |ui| self.strata_overview(ui)) {
                    // nothing to look at until it's configured
                    nav = Some(if self.gui.strata_cfg.api_key.trim().is_empty() { Tab::Settings } else { Tab::Strata });
                }
            });
            ui.add_space(6.0);
            ui.columns(2, |cols| {
                if overview_card(&mut cols[0], Tab::Agents, "Agents", |ui| self.agents_overview(ui)) {
                    nav = Some(Tab::Agents);
                }
                if overview_card(&mut cols[1], Tab::Ports, "Ports", |ui| self.ports_overview(ui)) {
                    nav = Some(Tab::Ports);
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
        headline(ui, &n_run.to_string(), &format!("running of {}", g.containers.len()), if n_run > 0 { GREEN } else { TEXT });
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
            [&d.image_usage, &d.container_usage, &d.volume_usage, &d.build_cache_usage]
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
            mini_row(ui, GREEN, name, &format!("{cpu:.1}% · {}", human_size(*mem)));
        }
        if rows.is_empty() {
            meta_line(ui, if n_run > 0 { "Collecting container stats…" } else { "No running containers." });
        }
        if let Some(e) = g.events.last() {
            let action = e.action.split(':').next().unwrap_or(&e.action);
            let color = match action {
                "start" | "create" | "unpause" => GREEN,
                "die" | "kill" | "stop" | "oom" => RED,
                _ => FAINT,
            };
            mini_row(ui, color, &format!("{} {action}", e.container_name()), &ago(epoch() - e.time));
        }
    }

    fn agents_overview(&self, ui: &mut Ui) {
        let g = &self.gui;
        let n = g.agents.len();
        headline(ui, &n.to_string(), "AI coding sessions", if n > 0 { GREEN } else { TEXT });
        if n == 0 {
            meta_line(ui, "No agents running.");
            return;
        }
        let cpu: f64 = g.agents.iter().map(|a| a.cpu_pct).sum();
        let mem: u64 = g.agents.iter().map(|a| a.mem_bytes).sum();
        meta_line(ui, &format!("{cpu:.0}% CPU · {} RSS", human_size(mem as i64)));
        let mut top: Vec<&AgentSession> = g.agents.iter().collect();
        top.sort_by(|a, b| b.cpu_pct.partial_cmp(&a.cpu_pct).unwrap_or(std::cmp::Ordering::Equal));
        for a in top.iter().take(3) {
            let name = match &a.cwd {
                Some(d) => format!("{} · {}", a.agent, basename(d)),
                None => a.agent.to_string(),
            };
            mini_row(ui, agent_color(a.agent), &name, &format!("{:.0}% · {}", a.cpu_pct, human_size(a.mem_bytes as i64)));
        }
    }

    fn ports_overview(&self, ui: &mut Ui) {
        let g = &self.gui;
        let listen: Vec<&PortEntry> = g.ports.iter().filter(|p| p.state == "LISTEN").collect();
        let exposed = listen.iter().filter(|p| p.addr == "0.0.0.0" || p.addr == "[::]").count();
        headline(ui, &listen.len().to_string(), "listening ports", CYAN);
        meta_line(ui, &format!("{exposed} open on all interfaces"));
        // dev servers first: sockets whose owner runs in a project directory
        let mut picks: Vec<&PortEntry> = listen.iter().copied().filter(|p| p.cwd.is_some()).collect();
        if picks.is_empty() {
            picks = listen.iter().copied().filter(|p| p.proc_name.is_some()).collect();
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
        let queued = if m.queued > 0 { format!(" · {} queued", m.queued) } else { String::new() };
        headline(ui, state, &format!("{}{queued}", if m.model.is_empty() { "strata" } else { &m.model }), color);
        let mut meta = vec![format!("{} requests served", fmt_tokens(m.tot_requests))];
        if let (Some(u), Some(t)) = (m.gpu_mem_used, m.gpu_mem_total) {
            meta.push(format!("VRAM {:.1} of {:.0} GB", u / 1e9, t / 1e9));
        }
        meta_line(ui, &meta.join(" · "));
        let last = m.requests.first();
        let dash = || "–".to_string();
        mini_row(ui, ACCENT, "Decode speed", &strata::decode_now(m, last).map(|v| format!("{v:.1} tok/s")).unwrap_or_else(dash));
        mini_row(ui, VIOLET, "First token (p50)", &m.ttft.p50.map(|v| format!("{v:.2} s")).unwrap_or_else(dash));
        let used = strata::context_used(m, last);
        mini_row(
            ui,
            CYAN,
            "Context",
            &if m.max_context > 0 {
                format!("{}% of {}", (used * 100) / m.max_context, fmt_tokens(m.max_context))
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
        let busy = self.gui.pending_action.as_ref().is_some_and(|(i, _)| *i == c.id);

        let mut action = None;
        let mut remove = false;
        ui.horizontal(|ui| {
            let title_w = (ui.available_width() - 330.0).max(160.0);
            ui.vertical(|ui| {
                ui.set_max_width(title_w);
                ui.spacing_mut().item_spacing.y = 6.0;
                ui.add(egui::Label::new(RichText::new(c.short_name()).font(bold(22.0)).color(TEXT)).truncate());
                ui.horizontal(|ui| {
                    state_badge(ui, &c.state);
                    if let Some(p) = c.project() {
                        chip(ui, p, MUTED);
                    }
                    ui.add(
                        egui::Label::new(RichText::new(&c.image).font(FontId::monospace(12.5)).color(MUTED))
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
                    action = Some(if running { Pending::Stop } else { Pending::Start });
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
                    if c.created > 0 { fmt_epoch(c.created) } else { "—".into() },
                    false,
                ),
                ("Image", c.image.clone(), true),
                ("Ports", if ports.is_empty() { "—".into() } else { ports }, true),
                ("Writable layer", c.size_rw.map(human_size).unwrap_or_else(|| "—".into()), false),
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
        let cpu = StatsSample::cpu_percent(g.prev_stats.get(&c.id), s).unwrap_or(0.0).max(0.0);
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
                value: format!("{cpu:.1}%"),
                sub: "of one core = 100%".to_string(),
                color: ACCENT,
                viz: Viz::Spark(series(&|h| h.cpu.iter().copied().collect())),
                glyph: Some("cpu"),
            },
            Kpi {
                label: "Memory",
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
                value: format!("↓ {}/s", human_size(rx)),
                sub: format!("↑ {}/s · {} pids", human_size(txb), s.pids_stats.current),
                color: CYAN,
                viz: Viz::Spark(series(&|h| h.rx.iter().zip(&h.tx).map(|(a, b)| (a + b) as f64).collect())),
                glyph: Some("net"),
            },
            Kpi {
                label: "Disk I/O",
                value: format!("{}/s", human_size(br + bw)),
                sub: format!("read {} · write {}", human_size(s.blk_read()), human_size(s.blk_write())),
                color: AMBER,
                viz: Viz::Spark(series(&|h| h.blk_r.iter().zip(&h.blk_w).map(|(a, b)| (a + b) as f64).collect())),
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
            empty_state(ui, "No image selected", "Pick an image on the left to see its size, tags and users.");
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
            chip(ui, if im.containers > 0 { "in use" } else { "unused" }, if im.containers > 0 { GREEN } else { FAINT });
        });
        ui.label(RichText::new(format!("sha256:{}", im.short_id())).font(FontId::monospace(12.5)).color(MUTED));
        ui.add_space(16.0);
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            let kpis = [
                Kpi {
                    label: "Size",
                    value: human_size(im.size),
                    sub: format!("{:.1}% of all layers", im.size as f64 / total.max(1) as f64 * 100.0),
                    color: ACCENT,
                    viz: Viz::Ratio(im.size as f32 / total.max(1) as f32),
                    glyph: None,
                },
                Kpi {
                    label: "Unique",
                    value: human_size((im.size - im.shared_size.max(0)).max(0)),
                    sub: format!("{} shared with other images", human_size(im.shared_size.max(0))),
                    color: VIOLET,
                    viz: Viz::Ratio((im.size - im.shared_size.max(0)).max(0) as f32 / im.size.max(1) as f32),
                    glyph: None,
                },
                Kpi {
                    label: "Containers",
                    value: im.containers.max(0).to_string(),
                    sub: if im.created > 0 { format!("created {}", fmt_epoch(im.created)) } else { String::new() },
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
                            ui.label(RichText::new("<none>").font(FontId::monospace(12.5)).color(FAINT));
                        }
                        for t in &im.repo_tags {
                            ui.add(egui::Label::new(RichText::new(t).font(FontId::monospace(12.5)).color(TEXT)).truncate());
                        }
                    });
                    card(&mut cols[1], |ui| {
                        card_title(ui, "Used by", &users.len().to_string());
                        if users.is_empty() {
                            ui.label(RichText::new("No containers use this image.").color(FAINT));
                        }
                        for u in users.iter().take(20) {
                            ui.horizontal(|ui| {
                                let (r, _) = ui.allocate_exact_size(Vec2::splat(8.0), Sense::hover());
                                ui.painter().circle_filled(r.center(), 3.0, GREEN);
                                ui.add(egui::Label::new(RichText::new(u).size(13.0).color(TEXT)).truncate());
                            });
                        }
                    });
                });
            });
        });
    }

    fn agent_detail(&mut self, ui: &mut Ui) {
        let Some(i) = self.gui.selected_agent else {
            empty_state(
                ui,
                "No agent selected",
                "Pick an agent session on the left to see its details.",
            );
            return;
        };
        // The selection indexes the unfiltered list; the filtered view may
        // hide it, in which case show the "ended" state rather than stale data.
        let Some(a) = self
            .gui
            .filtered_agents()
            .into_iter()
            .find(|(idx, _)| *idx == i)
            .map(|(_, a)| a)
            .cloned()
        else {
            empty_state(ui, "Session ended", "That agent session is no longer running.");
            return;
        };
        let color = agent_color(a.agent);
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new(a.agent).font(bold(22.0)).color(TEXT));
                ui.label(RichText::new(a.vendor).size(12.5).color(MUTED));
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                chip(ui, &format!("{} procs", a.pids.len()), color);
            });
        });
        ui.add_space(16.0);
        let uptime = epoch()
            - a.started
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs() as i64)
                .unwrap_or(0);
        let kpis = [
            Kpi {
                label: "CPU",
                value: format!("{:.1}%", a.cpu_pct),
                sub: "across the session tree".into(),
                color: GREEN,
                viz: Viz::None,
                glyph: Some("cpu"),
            },
            Kpi {
                label: "Memory",
                value: human_size(a.mem_bytes as i64),
                sub: "resident, summed".into(),
                color: VIOLET,
                viz: Viz::None,
                glyph: Some("ram"),
            },
            Kpi {
                label: "Uptime",
                value: ago(uptime),
                sub: format!("lead pid {}", a.lead_pid),
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

        card(ui, |ui| {
            card_title(ui, "Project", "");
            match &a.cwd {
                Some(c) => {
                    ui.label(RichText::new(c).font(FontId::monospace(12.5)).color(TEXT));
                }
                None => {
                    ui.label(RichText::new("not readable (different user?)").color(FAINT));
                }
            }
        });
        ui.add_space(10.0);
        card(ui, |ui| {
            card_title(ui, "Command line", "");
            ui.add(
                egui::Label::new(
                    RichText::new(&a.cmdline).font(FontId::monospace(12.0)).color(MUTED),
                )
                .wrap(),
            );
        });
        ui.add_space(10.0);
        card(ui, |ui| {
            card_title(ui, &format!("Processes ({})", a.pids.len()), "");
            for pid in &a.pids {
                ui.label(RichText::new(format!("pid {pid}")).font(FontId::monospace(12.0)).color(TEXT));
            }
        });
    }

    // ------------------------------------------------------------------
    // Hardware tab: host vitals, Task-Manager style
    // ------------------------------------------------------------------

    fn hardware_view(&mut self, ui: &mut Ui) {
        let Some(s) = self.gui.hw.clone() else {
            empty_state(ui, "Collecting…", "The first hardware sample lands in a moment.");
            return;
        };
        page_header(
            ui,
            "Hardware",
            &format!(
                "{} · {} cores · {} RAM · up {}",
                hwinv::INVENTORY.cpu,
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

        let mut kpis = vec![
            Kpi {
                label: "CPU",
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
                value: human_size(mem_used as i64),
                sub: format!("of {} ({:.0}%)", human_size(s.mem_total as i64), mem_used as f64 / s.mem_total.max(1) as f64 * 100.0),
                color: VIOLET,
                viz: Viz::Ratio(mem_used as f32 / s.mem_total.max(1) as f32),
                glyph: Some("ram"),
            },
            Kpi {
                label: "Disk I/O",
                value: format!("{} · {}", human_rate(s.disk_read_bps), human_rate(s.disk_write_bps)),
                sub: "read · write per second".into(),
                color: GREEN,
                viz: Viz::Spark(dr_hist.into_iter().zip(dw_hist).map(|(r, w)| r + w).collect()),
                glyph: Some("disk"),
            },
            Kpi {
                label: "Network",
                value: format!("{} · {}", human_rate(s.net_rx_bps), human_rate(s.net_tx_bps)),
                sub: "down · up per second".into(),
                color: AMBER,
                viz: Viz::Spark(rx_hist.into_iter().zip(tx_hist).map(|(r, t)| r + t).collect()),
                glyph: Some("net"),
            },
        ];
        if let Some(g) = gpus.iter().max_by(|a, b| a.busy_pct.unwrap_or(0.0).partial_cmp(&b.busy_pct.unwrap_or(0.0)).unwrap_or(std::cmp::Ordering::Equal)) {
            kpis.push(Kpi {
                label: "GPU",
                value: g.busy_pct.map(|b| format!("{b:.0}%")).unwrap_or_else(|| "—".into()),
                sub: match (g.temp_c, g.power_mw) {
                    (Some(t), Some(p)) => format!("{} · {} °C · {:.0} W", g.name, temp_str(t), p as f64 / 1000.0),
                    (Some(t), None) => format!("{} · {} °C", g.name, temp_str(t)),
                    _ => g.name.clone(),
                },
                color: CYAN,
                viz: Viz::Spark(gpu_busy_hist.clone()),
                glyph: Some("gpu"),
            });
        }
        if let Some(t) = s.disks.iter().filter_map(|d| d.temp_c).max_by(|a, b| a.partial_cmp(b).unwrap()) {
            kpis.push(Kpi {
                label: "SSD",
                value: format!("{} °C", temp_str(t)),
                sub: "drive temperature".into(),
                color: if t > 70.0 { RED } else if t > 60.0 { AMBER } else { GREEN },
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

        // ---- hardware inventory: every named device on the box ----
        card(ui, |ui| {
            card_title(ui, "Machine", &hwinv::INVENTORY.board);
            let icons: &[(&str, &str)] = &[
                ("cpu", "⬢"),
                ("board", "▦"),
                ("gpu", "◈"),
                ("memory", "▤"),
                ("disk", "⛁"),
                ("nic", "⇄"),
            ];
            for d in &hwinv::INVENTORY.devices {
                let icon = icons.iter().find(|(k, _)| *k == d.kind).map(|(_, i)| *i).unwrap_or("•");
                ui.horizontal(|ui| {
                    ui.label(RichText::new(icon).size(13.0).color(ACCENT));
                    ui.label(RichText::new(&d.name).font(bold(13.0)).color(TEXT));
                    if !d.detail.is_empty() {
                        ui.label(RichText::new(&d.detail).size(12.0).color(FAINT));
                    }
                });
            }
        });
        ui.add_space(10.0);

        card(ui, |ui| {
            card_title(
                ui,
                &format!("Per-core load ({})", s.per_core.len()),
                &s.cpu_temp_c.map(|t| format!("{} °C", temp_str(t))).unwrap_or_default(),
            );
            if s.per_core.is_empty() {
                ui.label(RichText::new("per-core stats unavailable").color(FAINT));
            } else {
                // grid of bars: one per core, labelled with its index
                let per_row = s.per_core.len().isqrt().max(4).min(8);
                for (row_i, row) in s.per_core.chunks(per_row).enumerate() {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 6.0;
                        for (j, v) in row.iter().enumerate() {
                            let idx = row_i * per_row + j;
                            let bar_w = 34.0_f32;
                            let (r, _) = ui.allocate_exact_size(Vec2::new(bar_w, 34.0), Sense::hover());
                            let p = ui.painter();
                            p.rect_filled(r, 4.0, WELL);
                            let frac = (v / 100.0).clamp(0.0, 1.0) as f32;
                            let fill = Rect::from_min_max(
                                Pos2::new(r.left() + 1.0, r.bottom() - 1.0 - (r.height() - 2.0) * frac),
                                Pos2::new(r.right() - 1.0, r.bottom() - 1.0),
                            );
                            let c = if *v > 85.0 {
                                RED
                            } else if *v > 60.0 {
                                AMBER
                            } else {
                                ACCENT
                            };
                            p.rect_filled(fill.shrink(1.0), 3.0, c.gamma_multiply(0.85));
                            p.text(
                                Pos2::new(r.center().x, r.center().y),
                                egui::Align2::CENTER_CENTER,
                                idx.to_string(),
                                FontId::monospace(9.5),
                                if *v > 40.0 { TEXT } else { FAINT },
                            );
                        }
                    });
                    ui.add_space(6.0);
                }
            }
            if let Some(note) = swap_note(&s) {
                ui.label(RichText::new(note).size(12.0).color(MUTED));
            }
        });
        ui.add_space(10.0);

        let mem_hist: Vec<f64> = self.gui.hw_mem_hist.iter().copied().collect();
        let swap_hist: Vec<f64> = self.gui.hw_swap_hist.iter().copied().collect();
        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.x = 14.0;
            ui.columns(2, |cols| {
                card(&mut cols[0], |ui| {
                    card_title(ui, "Memory & swap", "");
                    bar_row(ui, "Used", mem_used, s.mem_total, VIOLET);
                    bar_row(ui, "Swap", swap_used, s.swap_total, AMBER);
                    let w = ui.available_width();
                    area_chart(ui, &mem_hist, VIOLET, Vec2::new(w, 54.0), |v| human_size(v as i64));
                    let w = ui.available_width();
                    area_chart(ui, &swap_hist, AMBER, Vec2::new(w, 54.0), |v| human_size(v as i64));
                });
                card(&mut cols[1], |ui| {
                    card_title(ui, "Sensors", "");
                    if s.sensors.is_empty() {
                        ui.label(RichText::new("no hwmon temperature sensors found").color(FAINT));
                    } else {
                        for x in &s.sensors {
                            let c = if x.c > 85.0 { RED } else if x.c > 70.0 { AMBER } else { GREEN };
                            ui.horizontal(|ui| {
                                ui.label(RichText::new(x.group).size(11.5).color(FAINT));
                                ui.label(RichText::new(&x.label).size(12.5).color(MUTED));
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    ui.label(RichText::new(format!("{} °C", temp_str(x.c))).font(FontId::monospace(12.5)).color(c));
                                });
                            });
                        }
                    }
                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Disk").size(12.5).color(MUTED).weak());
                        ui.label(RichText::new(format!("↓ {}", human_rate(s.disk_read_bps))).font(FontId::monospace(12.5)).color(TEXT));
                        ui.label(RichText::new(format!("↑ {}", human_rate(s.disk_write_bps))).font(FontId::monospace(12.5)).color(TEXT));
                    });
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Net  ").size(12.5).color(MUTED).weak());
                        ui.label(RichText::new(format!("↓ {}", human_rate(s.net_rx_bps))).font(FontId::monospace(12.5)).color(TEXT));
                        ui.label(RichText::new(format!("↑ {}", human_rate(s.net_tx_bps))).font(FontId::monospace(12.5)).color(TEXT));
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
                let color = if busy > 85.0 { RED } else if busy > 60.0 { AMBER } else { CYAN };
                card(ui, |ui| {
                    card_title(
                        ui,
                        &g.name,
                        &match (g.temp_c, g.power_mw) {
                            (Some(t), Some(p)) => format!("{} °C · {:.0} W", temp_str(t), p as f64 / 1000.0),
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
                    area_chart(ui, &gpu_busy_hist, color, Vec2::new(w, 60.0), |v| format!("{v:.0}%"));
                });
                ui.add_space(8.0);
            }
        }

        // ---- Disks: per-device stats, Task-Manager style ----
        if !s.disks.is_empty() {
            ui.add_space(6.0);
            section_heading(ui, "Storage");
            ui.add_space(6.0);
            let models: std::collections::HashMap<String, hwinv::HwDevice> = hwinv::INVENTORY
                .devices
                .iter()
                .filter(|d| d.kind == "disk")
                .map(|d| (d.name.clone(), d.clone()))
                .collect();
            // match by kernel name order; the inventory keeps the same sort
            let inv_disks: Vec<&hwinv::HwDevice> = hwinv::INVENTORY.devices.iter().filter(|d| d.kind == "disk").collect();
            for (i, d) in s.disks.iter().enumerate() {
                let inv = models.get(&d.name).or_else(|| {
                    // diskstats names (nvme0n1, sda) vs inventory names: fall
                    // back to positional matching, both lists are sorted
                    inv_disks.get(i).map(|v| &**v)
                });
                let title = inv.map(|x| x.name.as_str()).unwrap_or(&d.name);
                let temp = d.temp_c.map(|t| format!("{} °C", temp_str(t))).unwrap_or_default();
                card(ui, |ui| {
                    card_title(ui, title, &temp);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Rate").size(12.5).color(MUTED).weak());
                        ui.label(RichText::new(format!("↓ {}", human_rate(d.read_bps))).font(FontId::monospace(12.5)).color(TEXT));
                        ui.label(RichText::new(format!("↑ {}", human_rate(d.write_bps))).font(FontId::monospace(12.5)).color(TEXT));
                        if let Some(inv) = inv {
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(RichText::new(&inv.detail).size(11.5).color(FAINT));
                            });
                        }
                    });
                    // active time since boot = utilization, averaged over uptime
                    let util = if d.uptime > 0.0 { (d.ms_busy as f64 / (d.uptime * 1000.0) * 100.0).min(100.0) } else { 0.0 };
                    bar_row(ui, "Active", (util * 10.0) as u64, 1000, GREEN);
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("Lifetime").size(12.5).color(MUTED).weak());
                        ui.label(RichText::new(format!("{} read", human_size(d.total_read as i64))).font(FontId::monospace(12.0)).color(MUTED));
                        ui.label(RichText::new(format!("{} written", human_size(d.total_write as i64))).font(FontId::monospace(12.0)).color(MUTED));
                        ui.label(RichText::new(format!("{} reads · {} writes", d.reads, d.writes)).size(11.5).color(FAINT));
                        // average queue depth = weighted busy ms / elapsed ms
                        let queue = if d.uptime > 0.0 { d.ms_weighted as f64 / (d.uptime * 1000.0) } else { 0.0 };
                        ui.label(RichText::new(format!("avg queue {queue:.2}")).size(11.5).color(FAINT));
                    });
                    if let Some(t) = d.temp_c {
                        let c = if t > 70.0 { RED } else if t > 60.0 { AMBER } else { GREEN };
                        ui.label(RichText::new(format!("drive temperature {} °C", temp_str(t))).size(11.5).color(c));
                    }
                });
                ui.add_space(8.0);
            }
        }
            });
        // regression: the body must actually scroll, not clip
        #[cfg(test)]
                #[cfg(test)]
        hw_scroll_probe::record(hw_body.inner_rect.height(), hw_body.content_size.y);
        #[cfg(not(test))]
        drop(hw_body);
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
            chip(ui, &format!("{}{}", p.proto, if p.family == 6 { "6" } else { "" }), color);
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
                value: p.proc_name.clone().unwrap_or_else(|| "—".into()),
                sub: p.pid.map(|x| format!("pid {x}")).unwrap_or_else(|| "owner not visible".into()),
                color: if p.pid.is_some() { GREEN } else { FAINT },
                viz: Viz::None,
                glyph: None,
            },
            Kpi {
                label: "Protocol",
                value: p.proto.to_uppercase(),
                sub: format!("IPv{}", p.family),
                color: ACCENT,
                viz: Viz::None,
                glyph: None,
            },
            Kpi {
                label: "Bind",
                value: if p.addr == "0.0.0.0" || p.addr == "[::]" { "public" } else if p.addr.starts_with("127.") || p.addr == "[::1]" { "localhost" } else { "interface" }.into(),
                sub: p.addr.clone(),
                color: if p.addr == "0.0.0.0" || p.addr == "[::]" { AMBER } else { GREEN },
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
                None => ui.label(RichText::new(if p.pid.is_some() { "not readable (different user?)" } else { "no owning process found" }).color(FAINT)),
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
                        RichText::new("changes apply the moment you touch them · saved automatically")
                            .size(12.0)
                            .color(FAINT),
                    );
                    ui.add_space(4.0);

                    // refresh interval: stepper + quick presets
                    let presets = [1.0_f64, 2.0, 5.0, 10.0];
                    let preset_hit = setting_row(ui, "Refresh interval", "seconds between live pulls · 0 disables polling", |ui| {
                        let mut hit = None;
                        for p in presets {
                            let on = (self.gui.refresh_edit - p).abs() < 0.01;
                            let b = ui.add_sized(
                                [40.0, 26.0],
                                egui::Button::new(RichText::new(format!("{p:.0}s")).size(12.0).color(if on { ACCENT } else { MUTED }))
                                    .fill(if on { ACCENT.gamma_multiply(0.18) } else { SURFACE_HI })
                                    .stroke(Stroke::new(1.0_f32, if on { ACCENT.gamma_multiply(0.6) } else { BORDER })),
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
                    });
                    let _ = preset_hit;
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
                            let vs = if v.fract() == 0.0 { format!("{}", v as i64) } else { format!("{v}") };
                            self.gui.status = format!("refresh every {vs} s");
                        }
                        general_dirty = false;
                    }

                    let autoscroll = setting_row(ui, "Follow log output", "stick to the bottom as new lines land", |ui| {
                        let on = self.gui.autoscroll_logs;
                        if switch(ui, on).clicked() {
                            self.gui.autoscroll_logs = !on;
                            self.gui.logs_follow = !on;
                            general_dirty = true;
                        }
                    });
                    let _ = autoscroll;
                    if general_dirty {
                        if let Err(e) = strata::patch(|c| {
                            c.general = strata::GeneralConfig {
                                refresh_secs: self.refresh,
                                autoscroll_logs: self.gui.autoscroll_logs,
                            };
                        }) {
                            self.gui.status = format!("could not save settings: {e}");
                        } else {
                            self.gui.status = if self.gui.autoscroll_logs { "log follow on".into() } else { "log follow off".into() };
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
                    let dot = if connected { GREEN } else if self.gui.strata_error.is_some() { RED } else { FAINT };
                    card_title(ui, "Strata server", "");
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("●").size(13.0).color(dot));
                        let sub = if let Some(h) = &self.gui.strata_health {
                            let svc = if h.service.is_empty() { "strata" } else { h.service.as_str() };
                            let model = if h.model.is_empty() { "no model" } else { &h.model };
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
                        ui.label(RichText::new(sub).size(12.0).color(if connected { GREEN } else { MUTED }));
                    });
                    ui.add_space(6.0);

                    let ep_resp = field(ui, &mut self.gui.strata_endpoint_edit, "http://127.0.0.1:9191", true, false, None);
                    #[cfg(test)]
                    {
                        self.gui.probe_endpoint_rect = Some(ep_resp.rect);
                    }
                    #[cfg(not(test))]
                    let _ = ep_resp;
                    ui.add_space(8.0);

                    let masked = self.gui.strata_key_masked;
                    let key_resp = field(ui, &mut self.gui.strata_key_edit, "sk-…", true, masked, Some((
                        if masked { "eye" } else { "eye_off" },
                        masked,
                        &mut |on: bool| {
                            self.gui.strata_key_masked = !on;
                        },
                    )));
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
                                        RichText::new("⟳  Test & save").font(bold(12.5)).color(ACCENT),
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
                                    let svc = if h.service.is_empty() { "strata" } else { h.service.as_str() };
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
                            self.strata_client = Arc::new(strata::HttpClient::new(&self.gui.strata_cfg));
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
                            ui.label(RichText::new(path).font(FontId::monospace(12.0)).color(MUTED));
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
                    let svc = if h.service.is_empty() { "strata" } else { h.service.as_str() };
                    let model = if h.model.is_empty() { "no model" } else { &h.model };
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
                empty_state(ui, "Collecting…", "The first /metrics pull lands in a moment.");
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
                        + &m.prompt_total.map(|t| format!(" / {} tokens read", fmt_tokens(t))).unwrap_or_default(),
                    m.prompt_total.filter(|t| *t > 0).and_then(|t| m.prompt_read.map(|r| r as f32 / t as f32)).unwrap_or(0.0),
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
                        m.tok_s.map(|v| format!("{v:.1}")).unwrap_or_else(|| "–".into())
                    ),
                    m.max_tokens.filter(|t| *t > 0).and_then(|t| m.generated.map(|g| (g as f32 / t as f32).min(1.0))).unwrap_or(0.0),
                )
            }
            _ => (
                "Idle".into(),
                FAINT,
                last.map(|r| {
                    format!(
                        "last: {} tokens{}",
                        fmt_tokens(r.output_tokens.unwrap_or(0)),
                        r.decode_tok_s.map(|d| format!(" at {d:.1} tok/s")).unwrap_or_default()
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
                        if m.model.is_empty() { "strata" } else { &m.model },
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
        let multi_gpu = m.gpu_name.as_deref().map(|n| n.contains(" + ")).unwrap_or(false);
        let gpu_sub = if multi_gpu {
            m.gpu_name.clone().unwrap_or_default()
        } else {
            m.gpu_name.clone().or_else(|| m.gpu_note.clone()).unwrap_or_default()
        };
        let gen = m.gpu_pcie_gen_max.or(m.gpu_pcie_gen);
        let ttft = &m.ttft;
        let kpis = vec![
            Kpi {
                label: "GPU SPEED",
                value: speed.map(|v| format!("{v:.1}")).unwrap_or_else(|| "–".into()),
                sub: if m.state == "generating" { "tok/s · decode now".into() } else { "tok/s · decode last request".into() },
                color: ACCENT,
                viz: Viz::Spark(hist("tok_s")),
                glyph: Some("gauge"),
            },
            Kpi {
                label: "GPU PREFILL",
                value: prefill.map(fmt_tokens_f).unwrap_or_else(|| "–".into()),
                sub: if m.state == "idle" { "t/s · prefill last request".into() } else { "t/s · prefill this request".into() },
                color: CYAN,
                viz: Viz::Spark(hist("prefill_tok_s_mean")),
                glyph: Some("gauge"),
            },
            Kpi {
                label: "GPU LOAD",
                value: m.gpu_util.map(|v| format!("{v:.0}")).unwrap_or_else(|| "–".into()),
                sub: format!("% · {}", gpu_sub),
                color: GREEN,
                viz: Viz::Spark(hist("gpu_util")),
                glyph: Some("gauge"),
            },
            Kpi {
                label: "GPU VRAM",
                value: m.gpu_mem_used.map(|v| format!("{:.1}", v / 1e9)).unwrap_or_else(|| "–".into()),
                sub: m
                    .gpu_mem_total
                    .map(|t| format!("of {:.0} GB · {}", t / 1e9, fmt_tokens(m.expert_slots.unwrap_or(0))))
                    .unwrap_or_else(|| "GB".into()),
                color: VIOLET,
                viz: Viz::Spark(hist("gpu_mem_used")),
                glyph: Some("gpu"),
            },
            Kpi {
                label: "GPU TEMP",
                value: m.gpu_temp.map(|v| format!("{v:.0}")).unwrap_or_else(|| "–".into()),
                sub: "°C".into(),
                color: temp_color(m.gpu_temp.unwrap_or(0.0), 85.0),
                viz: Viz::Spark(hist("gpu_temp")),
                glyph: Some("thermo"),
            },
            Kpi {
                label: "GPU POWER",
                value: m.gpu_power.map(|v| format!("{v:.0}")).unwrap_or_else(|| "–".into()),
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
                value: gen.map(|g| format!("Gen{g}")).unwrap_or_else(|| "–".into()),
                sub: format!(
                    "{} · to GPU {}",
                    m.gpu_pcie_width.map(|w| format!("x{w}")).unwrap_or_else(|| "x?".into()),
                    m.gpu_pcie_rx_mb.map(|v| format!("{v:.1} MB/s")).unwrap_or_else(|| "–".into())
                ),
                color: ORANGE,
                viz: Viz::Spark(hist("gpu_pcie_rx_mb")),
                glyph: Some("net"),
            },
            Kpi {
                label: "CPU",
                value: m.cpu.map(|v| format!("{v:.0}")).unwrap_or_else(|| "–".into()),
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
                value: m.disk_read_mb.map(|v| if v >= 1000.0 { format!("{:.2}", v / 1024.0) } else { format!("{v:.1}") }).unwrap_or_else(|| "–".into()),
                sub: m
                    .disk_write_mb
                    .map(|w| format!("{} · write {:.1} MB/s", if m.disk_read_mb.unwrap_or(0.0) >= 1000.0 { "GB/s" } else { "MB/s" }, w))
                    .unwrap_or_else(|| "MB/s".into()),
                color: GREEN,
                viz: Viz::Spark(hist("disk_read_mb")),
                glyph: Some("disk"),
            },
            Kpi {
                label: "FIRST TOKEN",
                value: ttft.p50.map(|v| format!("{v:.2}")).unwrap_or_else(|| "–".into()),
                sub: if ttft.count > 0 {
                    format!(
                        "s · p90 {:.2} s · {} requests{}",
                        ttft.p90.unwrap_or(0.0),
                        ttft.count,
                        m.itl.p50.map(|v| format!(" · {:.0} ms between tokens", v * 1000.0)).unwrap_or_default()
                    )
                } else {
                    "s · waiting for a request".into()
                },
                color: VIOLET,
                viz: Viz::Spark(m.requests.iter().rev().filter_map(|r| r.hit_rate.map(|h| h * 100.0)).collect()),
                glyph: Some("gauge"),
            },
            Kpi {
                label: "BATCH SLOTS",
                value: if m.parallel > 0 {
                    let busy = if m.running > 0 { m.running } else if m.state != "idle" { 1 } else { 0 };
                    format!("{busy}")
                } else {
                    (if m.queued > 0 { m.queued } else if m.state != "idle" { 1 } else { 0 }).to_string()
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
                    ui.label(RichText::new(format!("{}%", (frac_ctx * 100.0).round())).font(bold(20.0)).color(TEXT));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            RichText::new(format!("{} / {}", fmt_tokens(used), fmt_tokens(m.max_context)))
                                .size(12.5)
                                .color(MUTED),
                        );
                    });
                });
                ui.add_space(4.0);
                let cache_bytes = m.expert_cache_mib.unwrap_or(0) * 1_048_576;
                fact_row(ui, "Experts in VRAM", &match (m.expert_slots, m.gpu_mem_total) {
                    (Some(s), Some(_)) => format!(
                        "{} · {}",
                        fmt_tokens(s),
                        human_size(cache_bytes)
                    ),
                    (Some(s), None) => fmt_tokens(s),
                    _ => "–".into(),
                }, cache_bytes as f64 / m.gpu_mem_total.unwrap_or(0.0).max(1.0));
                fact_row(ui, "System RAM", &match (m.ram_used, m.ram_total) {
                    (Some(u), Some(t)) => format!("{} / {:.0} GB", format!("{:.1}", u / 1e9), t / 1e9),
                    _ => "–".into(),
                }, m.ram_used.unwrap_or(0.0) / m.ram_total.unwrap_or(1.0).max(1.0));
                fact_row(ui, "GPU temperature", &m.gpu_temp.map(|t| format!("{t:.0} °C")).unwrap_or_else(|| "–".into()), m.gpu_temp.unwrap_or(0.0) / 100.0);
            });
            card(&mut cols[1], |ui| {
                card_title(ui, "Engine", "");
                let kv = [
                    ("Model", m.model.clone()),
                    ("Context", fmt_tokens(m.max_context)),
                    ("Images", if m.images { "on" } else { "off" }.into()),
                    (
                        "CPU",
                        m.cpu_name.clone().unwrap_or_else(|| match (m.cores, m.threads) {
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
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                ui.label(RichText::new(v).size(12.5).color(TEXT));
                            });
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
                ui.label(RichText::new("Recent requests").font(bold(14.5)).color(TEXT));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if m.requests_kept > 12 {
                        if ui
                            .button(RichText::new(if show_all { "show fewer".to_string() } else { format!("show all ({})", m.requests_kept) }).size(11.5).color(ACCENT))
                            .clicked()
                        {
                            self.gui.strata_show_all = !show_all;
                        }
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
                            let align = if *right { egui::Align2::RIGHT_CENTER } else { egui::Align2::LEFT_CENTER };
                            let font = if mono { FontId::monospace(11.5) } else { FontId::proportional(11.5) };
                            ui.painter().text(
                                Pos2::new(if *right { cell.right() - 6.0 } else { cell.left() }, cell.center().y),
                                align,
                                text,
                                font,
                                color,
                            );
                        }
                    });
                };
                line(ui, cols.iter().map(|(h, _, _)| (h.to_string(), FAINT, false)).collect());
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
                        .map(|h| format!("{:.1}%", h * 100.0) + &r.pcie_share.map(|p| format!(" +{:.1}% PCIe", p * 100.0)).unwrap_or_default())
                        .unwrap_or_else(|| "–".into());
                    line(
                        ui,
                        vec![
                            (format!("{h:02}:{mnt:02}:{s:02}"), MUTED, true),
                            (finish_label(&r.finish), finish_color(&r.finish), false),
                            (r.prompt_tokens.map(fmt_tokens).unwrap_or_else(|| "–".into()), TEXT, true),
                            (r.reused.map(fmt_tokens).unwrap_or_else(|| "0".into()), MUTED, true),
                            (r.output_tokens.map(fmt_tokens).unwrap_or_else(|| "–".into()), TEXT, true),
                            (pre.map(|v| fmt_tokens_f(v)).unwrap_or_else(|| "–".into()), MUTED, true),
                            (r.decode_tok_s.map(|v| format!("{v:.1}")).unwrap_or_else(|| "–".into()), MUTED, true),
                            (hit, MUTED, true),
                            (r.duration_s.map(|v| format!("{v:.1} s")).unwrap_or_else(|| "–".into()), MUTED, true),
                        ],
                    );
                }
            }
            if m.tot_requests > 0 {
                let read = m.tot_prompt_tokens - m.tot_reused;
                let p_speed = if m.tot_prompt_ms > 0.0 && read > 0 {
                    format!(" at {} tok/s", fmt_tokens_f(read as f64 / (m.tot_prompt_ms / 1000.0)))
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
                    if let (Some(b), Some(bud)) = (c.bytes, c.budget_mib.map(|mib| mib * 1_048_576)) {
                        bar_row(ui, "Memory", b as u64, bud as u64, CYAN);
                    }
                    let switches = format!(
                        "{} parked · {} restored{}",
                        fmt_tokens(c.parks.unwrap_or(0)),
                        fmt_tokens(c.restores.unwrap_or(0)),
                        c.evictions.filter(|e| *e > 0).map(|e| format!(" · {} evicted", fmt_tokens(e))).unwrap_or_default()
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
                    format!(" ({:.1}% of all prompt tokens)", 100.0 * c.reused_tokens as f64 / c.prompt_tokens as f64)
                } else {
                    String::new()
                };
                ui.label(
                    RichText::new(format!("Reused since start: {} tokens{share}", fmt_tokens(c.reused_tokens)))
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
            empty_state(ui, "No volume selected", "Pick a volume on the left to see its details.");
            return;
        };
        let Some(v) = self.gui.volumes.iter().find(|v| v.name == name).cloned() else {
            empty_state(ui, "Volume not found", "That volume no longer exists.");
            return;
        };
        let refs = v.usage_data.as_ref().map(|u| u.ref_count);
        ui.horizontal(|ui| {
            ui.add(egui::Label::new(RichText::new(&v.name).font(bold(22.0)).color(TEXT)).truncate());
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
        let size = v.usage_data.as_ref().and_then(|u| u.size).filter(|s| *s >= 0);
        let kpis = [
            Kpi {
                label: "Size",
                value: size.map(human_size).unwrap_or_else(|| "—".into()),
                sub: if size.is_some() { "on disk".into() } else { "not reported by driver".into() },
                color: VIOLET,
                viz: Viz::None,
                glyph: None,
            },
            Kpi {
                label: "References",
                value: refs.map(|n| n.to_string()).unwrap_or_else(|| "—".into()),
                sub: if refs.is_some() { "containers mounting it".into() } else { "not reported by the daemon".into() },
                color: GREEN,
                viz: Viz::None,
                glyph: None,
            },
            Kpi {
                label: "Created",
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
            empty_state(ui, "No logs yet", "Pick a container, or reopen this tab to stream all containers.");
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
                        RichText::new(if g.logs.is_empty() { "Waiting for output…" } else { "No lines match the filter." })
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
                            let job = log_line_job(l, if all { l.source.as_deref() } else { None }, &font, a);
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
                ui.label(RichText::new("Remove container?").font(bold(17.0)).color(TEXT));
                ui.add_space(4.0);
                ui.label(
                    RichText::new(format!("{name} will be deleted. Its writable layer is lost; volumes are kept."))
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
    p.circle(r.center(), r.width() / 2.0 - 3.0, Color32::TRANSPARENT, stroke);
    if frac > 0.001 {
        let frac = frac.clamp(0.0, 1.0);
        let mut pts = Vec::new();
        let steps = (64.0 * frac).max(2.0) as usize;
        let rad = r.width() / 2.0 - 3.0;
        for i in 0..=steps {
            let a = std::f32::consts::TAU * (i as f32 / steps as f32) - std::f32::consts::FRAC_PI_2;
            pts.push(Pos2::new(r.center().x + rad * a.cos(), r.center().y + rad * a.sin()));
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
        if i > 0 && (digits.len() - i) % 3 == 0 {
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
            ui.horizontal(|ui| {
                let (r, _) = ui.allocate_exact_size(Vec2::splat(8.0), Sense::hover());
                ui.painter().circle_filled(r.center(), 3.5, k.color);
                ui.label(RichText::new(k.label).size(12.5).color(MUTED));
                if let Some(g) = k.glyph {
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let (gr, _) = ui.allocate_exact_size(Vec2::splat(16.0), Sense::hover());
                        icon_glyph(ui.painter(), g, gr.center(), 7.0, k.color.gamma_multiply(0.75));
                    });
                }
            });
            ui.add(egui::Label::new(RichText::new(&k.value).font(bold(24.0)).color(TEXT)).truncate());
            ui.add(egui::Label::new(RichText::new(&k.sub).size(12.0).color(FAINT)).truncate());
            ui.add_space(6.0);
            let (r, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 28.0), Sense::hover());
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
                    p.text(r.left_center(), egui::Align2::LEFT_CENTER, "collecting…", FontId::proportional(11.5), FAINT);
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
    p.galley(Pos2::new(rect.right() - vw, rect.center().y - v.size().y / 2.0), v, MUTED);
    let n = elide(p, name.to_string(), FontId::proportional(13.0), TEXT, rect.width() - vw - 28.0);
    p.galley(Pos2::new(rect.left() + 14.0, rect.center().y - n.size().y / 2.0), n, TEXT);
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
        Tab::Strata => "Strata",
        Tab::Logs => "Logs",
        Tab::Settings => "Settings",
    }
}

/// Surface card with standard padding; fills the available width.
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
                p.line_segment([Pos2::new(t, b.bottom()), Pos2::new(t, b.bottom() + r * 0.3)], s);
                let u = b.top() + b.height() * (i as f32 + 0.5) / 4.0;
                p.line_segment([Pos2::new(b.left(), u), Pos2::new(b.left() - r * 0.3, u)], s);
                p.line_segment([Pos2::new(b.right(), u), Pos2::new(b.right() + r * 0.3, u)], s);
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
                p.line_segment([Pos2::new(x, b.bottom()), Pos2::new(x, b.bottom() + r * 0.3)], s);
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
            let b = Rect::from_center_size(c + Vec2::new(r * 0.25, 0.0), Vec2::new(r * 2.1, r * 1.35));
            p.rect_stroke(
                b,
                2.0,
                Stroke::new(1.4_f32, col),
                egui::epaint::StrokeKind::Inside,
            );
            p.line_segment(
                [Pos2::new(b.left() - r * 0.45, b.top() - r * 0.2), Pos2::new(b.left() - r * 0.45, b.bottom() + r * 0.2)],
                Stroke::new(1.8_f32, col),
            );
            p.circle_stroke(Pos2::new(b.left() + b.width() * 0.33, b.center().y), r * 0.42, Stroke::new(1.1_f32, col));
            p.circle_stroke(Pos2::new(b.left() + b.width() * 0.72, b.center().y), r * 0.42, Stroke::new(1.1_f32, col));
        }
        "thermo" => {
            // thermometer: stem + bulb
            p.circle_stroke(Pos2::new(c.x, c.y + r * 0.62), r * 0.5, s);
            p.circle_filled(Pos2::new(c.x, c.y + r * 0.62), r * 0.24, col);
            p.line_segment([Pos2::new(c.x - r * 0.28, c.y - r * 0.9), Pos2::new(c.x - r * 0.28, c.y + r * 0.35)], s);
            p.line_segment([Pos2::new(c.x + r * 0.28, c.y - r * 0.9), Pos2::new(c.x + r * 0.28, c.y + r * 0.35)], s);
            p.line_segment([Pos2::new(c.x - r * 0.28, c.y - r * 0.9), Pos2::new(c.x + r * 0.28, c.y - r * 0.9)], s);
        }
        "net" => {
            // up/down arrows side by side
            let x1 = c.x - r * 0.5;
            let x2 = c.x + r * 0.5;
            p.line_segment([Pos2::new(x1, c.y + r * 0.8), Pos2::new(x1, c.y - r * 0.8)], s);
            p.line_segment([Pos2::new(x1 - r * 0.35, c.y - r * 0.4), Pos2::new(x1, c.y - r * 0.85)], s);
            p.line_segment([Pos2::new(x1 + r * 0.35, c.y - r * 0.4), Pos2::new(x1, c.y - r * 0.85)], s);
            p.line_segment([Pos2::new(x2, c.y - r * 0.8), Pos2::new(x2, c.y + r * 0.8)], s);
            p.line_segment([Pos2::new(x2 - r * 0.35, c.y + r * 0.4), Pos2::new(x2, c.y + r * 0.85)], s);
            p.line_segment([Pos2::new(x2 + r * 0.35, c.y + r * 0.4), Pos2::new(x2, c.y + r * 0.85)], s);
        }
        "eye" => {
            // almond outline + pupil
            let w = r * 1.35;
            let h = r * 0.78;
            p.add(
                egui::epaint::CubicBezierShape::from_points_stroke(
                    [
                        c + Vec2::new(-w, 0.0),
                        c + Vec2::new(-w * 0.42, -h),
                        c + Vec2::new(w * 0.42, -h),
                        c + Vec2::new(w, 0.0),
                    ],
                    false,
                    Color32::TRANSPARENT,
                    s,
                ),
            );
            p.add(
                egui::epaint::CubicBezierShape::from_points_stroke(
                    [
                        c + Vec2::new(w, 0.0),
                        c + Vec2::new(w * 0.42, h),
                        c + Vec2::new(-w * 0.42, h),
                        c + Vec2::new(-w, 0.0),
                    ],
                    false,
                    Color32::TRANSPARENT,
                    s,
                ),
            );
            p.circle_filled(c, r * 0.34, col);
        }
        "eye_off" => {
            // almond with the pupil shrunk to a ring, plus the slash
            let w = r * 1.35;
            let h = r * 0.78;
            p.add(
                egui::epaint::CubicBezierShape::from_points_stroke(
                    [
                        c + Vec2::new(-w, 0.0),
                        c + Vec2::new(-w * 0.42, -h),
                        c + Vec2::new(w * 0.42, -h),
                        c + Vec2::new(w, 0.0),
                    ],
                    false,
                    Color32::TRANSPARENT,
                    s,
                ),
            );
            p.add(
                egui::epaint::CubicBezierShape::from_points_stroke(
                    [
                        c + Vec2::new(w, 0.0),
                        c + Vec2::new(w * 0.42, h),
                        c + Vec2::new(-w * 0.42, h),
                        c + Vec2::new(-w, 0.0),
                    ],
                    false,
                    Color32::TRANSPARENT,
                    s,
                ),
            );
            p.circle_stroke(c, r * 0.34, s);
            p.line_segment(
                [c + Vec2::new(-r * 1.15, r * 1.05), c + Vec2::new(r * 1.15, -r * 1.05)],
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
            let tang = Vec2::new((a1 + std::f32::consts::FRAC_2_PI).cos(), (a1 + std::f32::consts::FRAC_2_PI).sin());
            let nrm = Vec2::new(-tang.y, tang.x);
            p.line_segment([tip, tip - tang * r * 0.55 + nrm * r * 0.38], s);
            p.line_segment([tip, tip - tang * r * 0.55 - nrm * r * 0.38], s);
        }
        _ => {}
    }
}

/// Settings text field: inset well, roomy padding, monospace where it
/// matters, and an optional icon button docked inside the right edge.
/// `on_trail` receives the clicked state and returns the new one.
fn field(
    ui: &mut Ui,
    value: &mut String,
    hint: &str,
    mono: bool,
    password: bool,
    trail: Option<(&str, bool, &mut dyn FnMut(bool))>,
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
    ui.painter().rect_stroke(well, 9.0, stroke, egui::epaint::StrokeKind::Middle);
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
fn setting_row<R>(
    ui: &mut Ui,
    label: &str,
    hint: &str,
    add: impl FnOnce(&mut Ui) -> R,
) -> R {
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
        ui.painter().add(egui::Shape::line(s, Stroke::new(2.0_f32, FAINT)));
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
        let (r, _) = ui.allocate_at_least(
            Vec2::new(ui.available_width(), 1.0),
            Sense::hover(),
        );
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
                ui.painter().circle_filled(r.center(), 6.0, c.gamma_multiply(0.2));
                ui.painter().circle_filled(r.center(), 3.5, c);
                ui.vertical(|ui| {
                    ui.spacing_mut().item_spacing.y = 0.0;
                    ui.label(
                        RichText::new(if ok { "Engine running" } else { "Engine unreachable" })
                            .size(12.5)
                            .color(TEXT),
                    );
                    if !version.is_empty() {
                        ui.label(RichText::new(format!("Docker {version}")).size(11.0).color(FAINT));
                    }
                });
            });
        });
}

/// Sidebar navigation entry; true when clicked.
fn nav_item(ui: &mut Ui, tab: Tab, label: &str, badge: &str, active: bool) -> bool {
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 38.0), Sense::click());
    let hovered = resp.hovered();
    if hovered {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    let p = ui.painter();
    if active {
        p.rect_filled(rect, 9.0, ACCENT.gamma_multiply(0.15));
        p.rect_filled(
            Rect::from_center_size(Pos2::new(rect.left() + 1.5, rect.center().y), Vec2::new(3.0, 18.0)),
            2.0,
            ACCENT,
        );
    } else if hovered {
        p.rect_filled(rect, 9.0, SURFACE_HI);
    }
    let fg = if active || hovered { TEXT } else { MUTED };
    let icon = Rect::from_center_size(Pos2::new(rect.left() + 22.0, rect.center().y), Vec2::splat(16.0));
    nav_icon(p, tab, icon, if active { ACCENT } else { fg });
    p.text(
        Pos2::new(rect.left() + 42.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        FontId::proportional(14.0),
        fg,
    );
    if !badge.is_empty() {
        let galley = p.layout_no_wrap(badge.to_string(), FontId::proportional(11.5), if active { TEXT } else { MUTED });
        let size = Vec2::new(galley.size().x + 14.0, 20.0);
        let br = Rect::from_min_size(Pos2::new(rect.right() - 8.0 - size.x, rect.center().y - 10.0), size);
        p.rect_filled(br, 10.0, if active { ACCENT.gamma_multiply(0.25) } else { SURFACE_HI });
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
                p.line_segment([Pos2::new(x, b.top() + 3.5), Pos2::new(x, b.bottom() - 3.5)], s);
            }
        }
        Tab::Images => {
            let base = Rect::from_min_max(Pos2::new(r.left(), r.bottom() - 7.0), r.max);
            p.rect_stroke(base, 2.0, s, inside);
            let y1 = base.top() - 3.0;
            p.line_segment([Pos2::new(r.left() + 2.5, y1), Pos2::new(r.right() - 2.5, y1)], s);
            let y2 = y1 - 3.5;
            p.line_segment([Pos2::new(r.left() + 5.0, y2), Pos2::new(r.right() - 5.0, y2)], s);
        }
        Tab::Volumes => {
            let rx = r.width() / 2.0 - 0.5;
            let top = Pos2::new(r.center().x, r.top() + 3.0);
            let bot = Pos2::new(r.center().x, r.bottom() - 3.0);
            p.add(egui::Shape::ellipse_stroke(top, Vec2::new(rx, 2.6), s));
            p.line_segment([Pos2::new(r.left() + 0.5, top.y), Pos2::new(r.left() + 0.5, bot.y)], s);
            p.line_segment([Pos2::new(r.right() - 0.5, top.y), Pos2::new(r.right() - 0.5, bot.y)], s);
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
            p.line_segment([Pos2::new(b.center().x, b.top()), Pos2::new(b.center().x, r.top() + 1.5)], s);
            p.circle_filled(Pos2::new(b.center().x, r.top() + 1.5), 1.2, c);
        }
        Tab::Logs => {
            for (i, f) in [1.0, 0.65, 0.85, 0.5].iter().enumerate() {
                let y = r.top() + 2.0 + i as f32 * 4.0;
                p.line_segment([Pos2::new(r.left(), y), Pos2::new(r.left() + r.width() * f, y)], s);
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
                p.line_segment([Pos2::new(t, b.bottom()), Pos2::new(t, b.bottom() + pin)], s);
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
            p.line_segment([Pos2::new(b.center().x, b.bottom()), Pos2::new(b.center().x, r.bottom() + 1.5)], s);
        }
        Tab::Settings => {
            // a gear: ring, hub, six teeth
            let ring = r.shrink2(Vec2::new(3.5, 3.5));
            p.circle_stroke(ring.center(), ring.width().min(ring.height()) / 2.0 - 1.0, s);
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
            p.line_segment([Pos2::new(b.center().x, b.top()), Pos2::new(b.center().x, b.top() - pin)], s);
            p.line_segment([Pos2::new(b.center().x, b.bottom()), Pos2::new(b.center().x, b.bottom() + pin)], s);
            p.line_segment([Pos2::new(b.left(), b.center().y), Pos2::new(b.left() - pin, b.center().y)], s);
            p.line_segment([Pos2::new(b.right(), b.center().y), Pos2::new(b.right() + pin, b.center().y)], s);
        }
    }
}

/// Rounded icon tile at the left edge of a list row.
fn icon_tile(p: &egui::Painter, row: Rect, tab: Tab, color: Color32) {
    let tile = Rect::from_center_size(Pos2::new(row.left() + 26.0, row.center().y), Vec2::splat(34.0));
    p.rect_filled(tile, 9.0, color.gamma_multiply(0.12));
    nav_icon(p, tab, Rect::from_center_size(tile.center(), Vec2::splat(16.0)), color);
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
fn elide(p: &egui::Painter, text: String, font: FontId, color: Color32, max_w: f32) -> Arc<egui::Galley> {
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
    let r = Rect::from_min_size(Pos2::new(right_center.x - size.x, right_center.y - size.y / 2.0), size);
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
    let galley = ui.painter().layout_no_wrap(text.to_string(), FontId::proportional(11.5), color);
    let size = Vec2::new(galley.size().x + 30.0, 22.0);
    let (r, _) = ui.allocate_exact_size(size, Sense::hover());
    let p = ui.painter();
    p.rect_filled(r, 11.0, color.gamma_multiply(0.12));
    p.circle_filled(Pos2::new(r.left() + 12.0, r.center().y), 3.0, color);
    p.galley(Pos2::new(r.left() + 21.0, r.center().y - galley.size().y / 2.0), galley, color);
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
    p.rect_stroke(rect, 11.0, Stroke::new(1.0_f32, stroke), egui::epaint::StrokeKind::Inside);
    let knob = 16.0_f32;
    let cx = if on { rect.right() - knob / 2.0 - 3.0 } else { rect.left() + knob / 2.0 + 3.0 };
    p.circle_filled(Pos2::new(cx, rect.center().y), knob / 2.0, if on { Color32::WHITE } else { MUTED });
    resp
}

fn toggle_chip(ui: &mut Ui, label: &str, on: bool) -> egui::Response {
    let (fill, fg, stroke) = if on {
        (ACCENT.gamma_multiply(0.16), ACCENT, ACCENT.gamma_multiply(0.5))
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
    ui.painter().line_segment([c + Vec2::new(3.3, 3.3), c + Vec2::new(6.8, 6.8)], s);
    if !text.is_empty() {
        let r = Rect::from_center_size(Pos2::new(resp.rect.right() - 16.0, resp.rect.center().y), Vec2::splat(18.0));
        let clear = ui.interact(r, resp.id.with("clear"), Sense::click());
        let col = if clear.hovered() { TEXT } else { FAINT };
        let d = 3.5;
        ui.painter().line_segment([r.center() - Vec2::splat(d), r.center() + Vec2::splat(d)], Stroke::new(1.5_f32, col));
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
    ui.painter()
        .rect(rect, 9.0, SURFACE, Stroke::new(1.0_f32, BORDER), egui::epaint::StrokeKind::Inside);
    let w = (rect.width() - 6.0) / opts.len() as f32;
    let mut out = None;
    for (i, (v, label)) in opts.iter().enumerate() {
        let r = Rect::from_min_size(rect.min + Vec2::new(3.0 + i as f32 * w, 3.0), Vec2::new(w, h - 6.0));
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
    let t = if mono { t.font(FontId::monospace(12.5)) } else { t.size(13.5) };
    ui.add(egui::Label::new(t).truncate());
    ui.add_space(8.0);
}

fn progress(p: &egui::Painter, bar: Rect, frac: f32, color: Color32) {
    let r = bar.height() / 2.0;
    p.rect_filled(bar, r, SURFACE_HI);
    let frac = frac.clamp(0.0, 1.0);
    if frac > 0.0 {
        let w = (bar.width() * frac).max(bar.height());
        p.rect_filled(Rect::from_min_size(bar.min, Vec2::new(w, bar.height())), r, color);
    }
}

/// Tiny inline trend line.
fn sparkline(p: &egui::Painter, rect: Rect, series: impl Iterator<Item = f64>, color: Color32, floor: f64) {
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
    p.line_segment([rect.left_bottom(), rect.right_bottom()], Stroke::new(1.0_f32, BORDER));
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
        p.line_segment([Pos2::new(q.x, rect.top()), Pos2::new(q.x, rect.bottom())], Stroke::new(1.0_f32, BORDER));
        p.circle_filled(q, 4.0, color);
        p.circle_stroke(q, 4.0, Stroke::new(1.5_f32, SURFACE));
        let label = fmt(series[i]);
        let ago_s = pts.len() - 1 - i;
        let text = if ago_s == 0 { label } else { format!("{label} · {ago_s}s ago") };
        let galley = p.layout_no_wrap(text, FontId::proportional(11.5), TEXT);
        let mut tip = Rect::from_min_size(q + Vec2::new(8.0, -26.0), galley.size() + Vec2::new(12.0, 6.0));
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
fn log_line_job(l: &LogLine, source: Option<&str>, font: &FontId, alpha: f32) -> egui::text::LayoutJob {
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
    const PALETTE: [Color32; 6] = [ACCENT, GREEN, VIOLET, AMBER, CYAN, Color32::from_rgb(244, 140, 190)];
    let h = name.bytes().fold(5381u32, |h, b| h.wrapping_mul(33) ^ b as u32);
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
    let (a, d) = (r.min + Vec2::new(60.0, 52.0) * k, Vec2::new(136.0, 152.0) * k);
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
    p.add(egui::Shape::line(pts, egui::epaint::PathStroke::new_uv(w, ink)));
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
    pts.extend([(100.0, 128.0), (110.0, 128.0), (120.0, 98.0), (134.0, 158.0), (144.0, 128.0), (156.0, 128.0)]);
    pts.extend(bowl(156.0, 160.0, -90.0, 90.0));
    pts.extend([(156.0, 192.0), (84.0, 192.0)]);
    let k = r.width() / 256.0;
    pts.into_iter().map(|(x, y)| r.min + Vec2::new(x, y) * k).collect()
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
            let h = n.bytes().fold(5381u32, |h, b| h.wrapping_mul(33) ^ b as u32);
            const PALETTE: [Color32; 5] = [ACCENT, GREEN, VIOLET, AMBER, Color32::from_rgb(120, 160, 120)];
            PALETTE[h as usize % PALETTE.len()]
        }
        None => {
            let public = p.addr == "0.0.0.0" || p.addr == "[::]";
            if public { AMBER } else { FAINT }
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

/// Swap usage note for the per-core card footer.
fn swap_note(s: &HwSample) -> Option<String> {
    if s.swap_total == 0 {
        return None;
    }
    let used = s.swap_total - s.swap_free;
    Some(format!(
        "swap {} of {} in use",
        human_size(used as i64),
        human_size(s.swap_total as i64)
    ))
}

/// Labelled horizontal bar: "Used ███░░ 12.3 GB of 64 GB".
fn bar_row(ui: &mut Ui, label: &str, used: u64, total: u64, color: Color32) {
    if total == 0 {
        ui.label(RichText::new(format!("{label}: none")).size(12.5).color(FAINT));
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
        format!("{} of {}", human_size(used as i64), human_size(total as i64)),
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

fn spawn_info(
    rt: &tokio::runtime::Handle,
    docker: &Arc<Docker>,
    tx: &mpsc::UnboundedSender<Msg>,
) {
    let docker = Arc::clone(docker);
    let tx = tx.clone();
    rt.spawn(async move {
        let _ = tx.send(Msg::Info(docker.system_info().await.map_err(err_str)));
    });
}

fn spawn_df(
    rt: &tokio::runtime::Handle,
    docker: &Arc<Docker>,
    tx: &mpsc::UnboundedSender<Msg>,
) {
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
                Ok(Some(line)) => {
                    match serde_json::from_str::<StatsSample>(&line) {
                        Ok(s) => {
                            if tx.send(Msg::Stats(id.clone(), Box::new(s))).is_err() {
                                return;
                            }
                        }
                        Err(e) => {
                            eprintln!("stats decode error: {e} in {:?}", &line[..line.len().min(160)]);
                            continue;
                        }
                    }
                }
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
            if !collected.is_empty() && tx.send(Msg::AllLogs(tag(collected, &Some(label)))).is_err() {
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
                let line = String::from_utf8_lossy(&self.buf[..pos]).trim_end().to_string();
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
                    self.chunked = head
                        .windows(7)
                        .any(|w| w.eq_ignore_ascii_case(b"chunked"));
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

pub fn run(rt: tokio::runtime::Runtime, socket: &str, refresh: f64) -> Result<(), Box<dyn std::error::Error>> {
    run_with(rt, socket, refresh, None)
}

/// `tab_shot`: open directly on a tab and auto-close after that many
/// seconds — used by the throwaway screenshot harness for visual checks.
pub fn run_with(rt: tokio::runtime::Runtime, socket: &str, refresh: f64, tab_shot: Option<f64>) -> Result<(), Box<dyn std::error::Error>> {
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
            if let Some(secs) = tab_shot {
                app.gui.tab = Tab::Agents;
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
        let mut eof = false;
        'head: loop {
            let mut b = [0u8; 512];
            let n = match stream.read(&mut b).await {
                Ok(0) => {
                    eof = true;
                    break 'head;
                }
                Ok(n) => n,
                Err(e) => {
                    let _ = tx.send(Msg::LogDone(gen, id.clone(), Err(err_str(DockerError::Io(e.to_string())))));
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
            let _ = tx.send(Msg::LogDone(gen, id.clone(), Err("unexpected response from daemon".into())));
            return;
        }
        if let Some(code) = head.get(9..12).and_then(|c| std::str::from_utf8(c).ok()) {
            if !code.starts_with('2') {
                let _ = tx.send(Msg::LogDone(gen, id.clone(), Err(format!("daemon returned {code}"))));
                return;
            }
        }
        let chunked = head.windows(7).any(|w| w.eq_ignore_ascii_case(b"chunked"));

        let mut fr = Chunked::default();
        let mut chunk = [0u8; 16 * 1024];
        if !body_prefix.is_empty() {
            let payload = if chunked { fr.feed(&body_prefix) } else { body_prefix.clone() };
            let lines = dec.push(&payload);
            if !lines.is_empty() && tx.send(Msg::LogLines(gen, tag(lines, &label))).is_err() {
                return;
            }
        }
        while !eof {
            match stream.read(&mut chunk).await {
                Ok(0) => break,
                Ok(n) => {
                    let payload = if chunked { fr.feed(&chunk[..n]) } else { chunk[..n].to_vec() };
                    let lines = dec.push(&payload);
                    if !lines.is_empty() && tx.send(Msg::LogLines(gen, tag(lines, &label))).is_err() {
                        return;
                    }
                    if chunked && fr.done {
                        break;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Msg::LogDone(gen, id.clone(), Err(err_str(DockerError::Io(e.to_string())))));
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
        let rt = Arc::new(tokio::runtime::Builder::new_current_thread().build().unwrap());
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
        raw.screen_rect = Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(1200.0, 800.0)));
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
        assert!(all.contains("probe-model"), "model name never painted — page is blank");
        assert!(all.contains("123"), "throughput stat never painted");
        assert!(!all.contains("Collecting"), "stats present but page still says Collecting");
    }

    /// Without metrics yet the page must still paint *something* (status
    /// card + empty state), never a blank panel.
    #[test]
    fn strata_page_never_goes_blank_before_first_pull() {
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let (tx, rx) = mpsc::unbounded_channel();
        let rt = Arc::new(tokio::runtime::Builder::new_current_thread().build().unwrap());
        let mut app = SysKamlingApp::new_offscreen(&ctx, tx, rx, rt);
        app.gui.tab = Tab::Strata;
        app.gui.strata_cfg.api_key = "k".into();
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(1200.0, 800.0)));
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
        assert!(painted > 8, "strata waiting state painted only {painted} text shapes — page is blank");
    }

    /// The Settings tab must paint its General + Strata cards.
    #[test]
    fn settings_tab_renders_both_cards() {
        let ctx = egui::Context::default();
        apply_theme(&ctx);
        let (tx, rx) = mpsc::unbounded_channel();
        let rt = Arc::new(tokio::runtime::Builder::new_current_thread().build().unwrap());
        let mut app = SysKamlingApp::new_offscreen(&ctx, tx, rx, rt);
        app.gui.tab = Tab::Settings;
        let mut raw = egui::RawInput::default();
        raw.screen_rect = Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(1200.0, 800.0)));
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
        assert!(painted > 15, "settings tab painted only {painted} text shapes — cards missing");
    }
}

#[cfg(test)]
mod settings_tab_tests {
    use super::*;

    fn settings_probe() -> (egui::Context, SysKamlingApp) {
        let ctx = egui::Context::default();
        let (tx, rx) = mpsc::unbounded_channel();
        let rt = Arc::new(tokio::runtime::Builder::new_current_thread().build().unwrap());
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
        let dir = std::env::temp_dir().join(format!(
            "syskamling-gui-settings-{}",
            std::process::id()
        ));
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

    fn paint(ctx: &egui::Context, app: &mut SysKamlingApp, events: Vec<egui::Event>) -> Vec<String> {
        let mut raw = egui::RawInput::default();
        raw.screen_rect =
            Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(1200.0, 800.0)));
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
        let _home = isolate_home();
        let (ctx, mut app) = settings_probe();
        // Pass 1 paints the tab; the view records the endpoint field's rect
        // into the probe field (cfg(test) seam), so Pass 2 can click it.
        let mut raw = egui::RawInput::default();
        raw.screen_rect =
            Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(1200.0, 800.0)));
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
        assert!(all.contains("Strata server"), "settings card title never painted");
    }

    /// Layout sanity for the redesigned tab: the Test & save button must sit
    /// tight under the key field (no dead gap), and the whole card column
    /// must fit a short window without sideways overflow.
    #[test]
    fn save_button_sits_close_under_the_fields() {
        let _home = isolate_home();
        let (ctx, mut app) = settings_probe();
        let mut raw = egui::RawInput::default();
        raw.screen_rect =
            Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::new(1100.0, 420.0)));
        ctx.begin_pass(raw.clone());
        egui::CentralPanel::default().show(&ctx, |ui| app.settings_view(ui));
        let _ = ctx.end_pass();
        let key = app.gui.probe_endpoint_rect.expect("endpoint field");
        let save = app.gui.probe_save_rect.expect("save button");
        // second field sits ~10px below the first; the button ~12px below that
        let gap = save.top() - (key.bottom() + 48.0); // 48 ≈ key field height+spacing
        assert!(gap.abs() < 20.0, "dead gap between fields and save: {gap:.1}px");
        assert!(save.left() >= key.left() - 1.0, "save button not aligned to card");
    }

    /// Changing the refresh interval must retune the poller at once.
    #[test]
    fn refresh_interval_drag_takes_effect() {
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
        assert!((app.refresh - 7.5).abs() < 1e-6, "refresh not applied: {}", app.refresh);
    }
}

#[cfg(test)]
mod hw_scroll_tests {
    use super::*;

    #[test]
    fn hardware_page_body_scrolls_instead_of_clipping() {
        let (viewport_h, content_h) = SysKamlingApp::probe_hardware_scroll(egui::Vec2::new(1100.0, 420.0));
        assert!(viewport_h > 0.0, "probe never ran");
        assert!(content_h > viewport_h, "content {content_h} must exceed viewport {viewport_h} so the body scrolls");
        // the body must not be collapsed to nothing either
        assert!(content_h > 1000.0, "content {content_h} looks collapsed/blank");
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
