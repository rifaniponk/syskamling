//! Graphical desktop front-end (egui/eframe).
//!
//! Same engine facade as the TUI: all Docker calls run on the tokio runtime
//! and report back through an mpsc channel, so the window never blocks.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use egui::{
    Color32, FontData, FontFamily, FontId, Frame, Margin, Pos2, Rect, RichText, ScrollArea, Sense,
    Stroke, TextStyle, Ui, Vec2,
};

use tokio::sync::mpsc;

use crate::agents::{AgentScanner, AgentSession};
use crate::docker::{
    self, ContainerSummary, DiskSummary, Docker, DockerError, EventMessage, ImageSummary,
    LogLine, StatsSample, StreamKind, SystemInfo,
};
use crate::hw::{HwSample, Sampler};
use crate::ports::{self as portsmod, PortEntry};

/// One page of container logs held by the UI.
const LOG_CAP: usize = 5_000;
/// How many lines to pull when first opening a container's logs.
const LOG_TAIL: usize = 1_000;
/// Lines per container when refreshing the consolidated Logs tab.
const ALL_LOG_TAIL: usize = 200;
/// History kept per running container for the dashboard charts.
const HIST: usize = 60;

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Tab {
    Dashboard,
    Containers,
    Images,
    Volumes,
    Agents,
    Hardware,
    Ports,
    Logs,
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
    AllLogs(Vec<LogLine>),
    AgentsScanned(Vec<AgentSession>),
    PortsScanned(Vec<PortEntry>),
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
    hw_disk_r_hist: VecDeque<f64>,
    hw_disk_w_hist: VecDeque<f64>,
    hw_net_rx_hist: VecDeque<f64>,
    hw_net_tx_hist: VecDeque<f64>,
    // --- ports tab ---
    ports: Vec<PortEntry>,
    /// (proto, addr, port) of the selected socket
    selected_port: Option<(String, String, u16)>,
    port_filter: String,
    /// only sockets with a discoverable owning process
    ports_owned_only: bool,
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
            hw_disk_r_hist: VecDeque::new(),
            hw_disk_w_hist: VecDeque::new(),
            hw_net_rx_hist: VecDeque::new(),
            hw_net_tx_hist: VecDeque::new(),
            ports: Vec::new(),
            selected_port: None,
            port_filter: String::new(),
            ports_owned_only: false,
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

    fn push_hw(&mut self, s: HwSample) {
        let push = |q: &mut VecDeque<f64>, v: f64| {
            q.push_back(v);
            while q.len() > HIST {
                q.pop_front();
            }
        };
        push(&mut self.hw_cpu_hist, s.cpu_pct);
        push(&mut self.hw_mem_hist, (s.mem_total - s.mem_avail) as f64);
        push(&mut self.hw_disk_r_hist, s.disk_read_bps);
        push(&mut self.hw_disk_w_hist, s.disk_write_bps);
        push(&mut self.hw_net_rx_hist, s.net_rx_bps);
        push(&mut self.hw_net_tx_hist, s.net_tx_bps);
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
        let app = Self {
            gui: Gui::new(),
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
        };
        app
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
        if self.tick.elapsed() >= Duration::from_millis(250) {
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
            ctx.request_repaint();
        }

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
    /// /proc reads are cheap (microseconds); no thread needed.
    fn sample_hardware(&mut self) {
        if self.last_hw_sample.elapsed() < Duration::from_secs_f64(self.refresh.max(1.0)) {
            return;
        }
        self.last_hw_sample = Instant::now();
        let s = self.sampler.sample();
        self.gui.push_hw(s);
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
            (Tab::Logs, "Logs", String::new()),
        ];
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
    // Dashboard
    // ------------------------------------------------------------------

    fn dashboard_view(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        // The dashboard leads with the machine, not the engine: host vitals
        // first, Docker as one sub-section below.
        let sub = match &g.hw {
            Some(h) => format!(
                "{} · {} cores · {} RAM · up {}",
                host_os_name(),
                h.ncpu,
                human_size(h.mem_total as i64),
                uptime_str(h.uptime as i64)
            ),
            None => "Collecting host vitals…".to_string(),
        };
        let ok = g.daemon_ok;
        ui.horizontal(|ui| {
            ui.vertical(|ui| page_header(ui, "Dashboard", &sub));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                live_pill(ui, true);
            });
        });
        ui.add_space(16.0);
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.kpi_cards(ui);
                ui.add_space(14.0);
                self.system_charts(ui);
                ui.add_space(14.0);
                self.system_summary(ui);
                ui.add_space(14.0);
                section_heading(ui, "Docker");
                ui.add_space(6.0);
                self.docker_strip(ui, ok);
                ui.add_space(14.0);
                self.docker_charts(ui);
                ui.add_space(14.0);
                if ui.available_width() > 860.0 {
                    ui.scope(|ui| {
                        ui.spacing_mut().item_spacing.x = 14.0;
                        ui.columns(2, |cols| {
                            self.top_containers(&mut cols[0]);
                            self.event_feed(&mut cols[1]);
                        });
                    });
                } else {
                    self.top_containers(ui);
                    ui.add_space(14.0);
                    self.event_feed(ui);
                }
                ui.add_space(14.0);
                self.disk_section(ui);
                ui.add_space(8.0);
            });
    }

    /// Host-wide series summed over every container with live stats:
    /// (cpu % of host, memory bytes, net bytes/s in+out).
    fn host_series(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize) {
        let g = &self.gui;
        let ncpu = g.info.as_ref().map(|i| i.ncpu.max(1)).unwrap_or(1) as f64;
        let mut cpu = vec![0.0; HIST];
        let mut mem = vec![0.0; HIST];
        let mut net = vec![0.0; HIST];
        let mut samples = 0;
        for (id, h) in &g.hist {
            if !g.stats.contains_key(id) {
                continue;
            }
            samples = samples.max(h.cpu.len());
            let off = HIST - h.cpu.len();
            for (i, v) in h.cpu.iter().enumerate() {
                cpu[off + i] += v / ncpu;
            }
            let off = HIST - h.mem.len();
            for (i, v) in h.mem.iter().enumerate() {
                mem[off + i] += *v as f64;
            }
            let off = HIST - h.rx.len();
            for (i, (rx, tx)) in h.rx.iter().zip(h.tx.iter()).enumerate() {
                net[off + i] += (*rx + *tx) as f64;
            }
        }
        // Only the window that actually has samples, so charts don't start
        // with a flat run of zeros.
        let skip = HIST - samples;
        (
            cpu[skip..].to_vec(),
            mem[skip..].to_vec(),
            net[skip..].to_vec(),
            samples,
        )
    }

    fn kpi_cards(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        let n_run = g.containers.iter().filter(|c| c.state == "running").count();
        let n_listen = g.ports.iter().filter(|p| p.state == "LISTEN").count();
        let n_agents = g.agents.len();
        let (disk_total, disk_reclaim) = g
            .df
            .as_ref()
            .map(|d| {
                [&d.image_usage, &d.container_usage, &d.volume_usage, &d.build_cache_usage]
                    .into_iter()
                    .flatten()
                    .fold((0, 0), |(t, r), u| (t + u.total_size, r + u.reclaimable))
            })
            .unwrap_or((0, 0));
        let cpu_hist: Vec<f64> = g.hw_cpu_hist.iter().copied().collect();
        let mem_hist: Vec<f64> = g.hw_mem_hist.iter().copied().collect();

        let kpis = match &g.hw {
            Some(h) => {
                let mem_used = h.mem_total - h.mem_avail;
                [
                    Kpi {
                        label: "CPU",
                        value: format!("{:.1}%", h.cpu_pct),
                        sub: format!("{} cores · load {}", h.ncpu, h.load1),
                        color: ACCENT,
                        viz: Viz::Spark(cpu_hist),
                    },
                    Kpi {
                        label: "Memory",
                        value: human_size(mem_used as i64),
                        sub: format!(
                            "of {} ({:.0}%)",
                            human_size(h.mem_total as i64),
                            mem_used as f64 / h.mem_total.max(1) as f64 * 100.0
                        ),
                        color: VIOLET,
                        viz: Viz::Spark(mem_hist),
                    },
                    Kpi {
                        label: "Ports",
                        value: n_listen.to_string(),
                        sub: format!("listening · {} containers", n_run),
                        color: CYAN,
                        viz: Viz::None,
                    },
                    Kpi {
                        label: "Agents",
                        value: n_agents.to_string(),
                        sub: if n_agents == 0 {
                            "no AI sessions running".into()
                        } else {
                            "AI coding sessions".into()
                        },
                        color: GREEN,
                        viz: Viz::None,
                    },
                ]
            }
            None => [
                Kpi {
                    label: "CPU",
                    value: "—".into(),
                    sub: "collecting host vitals…".into(),
                    color: ACCENT,
                    viz: Viz::Spark(cpu_hist),
                },
                Kpi {
                    label: "Memory",
                    value: "—".into(),
                    sub: "collecting host vitals…".into(),
                    color: VIOLET,
                    viz: Viz::Spark(mem_hist),
                },
                Kpi {
                    label: "Ports",
                    value: n_listen.to_string(),
                    sub: format!("listening · {} containers", n_run),
                    color: CYAN,
                    viz: Viz::None,
                },
                Kpi {
                    label: "Disk",
                    value: if g.df.is_some() { human_size(disk_total) } else { "—".into() },
                    sub: format!("{} reclaimable", human_size(disk_reclaim)),
                    color: AMBER,
                    viz: Viz::Ratio(disk_reclaim as f32 / disk_total.max(1) as f32),
                },
            ],
        };
        let per_row = if ui.available_width() > 860.0 { 4 } else { 2 };
        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.x = 14.0;
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

    /// Host-wide charts fed by the /proc sampler: CPU, memory, network.
    fn system_charts(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        let cpu: Vec<f64> = g.hw_cpu_hist.iter().copied().collect();
        let mem: Vec<f64> = g.hw_mem_hist.iter().copied().collect();
        let net: Vec<f64> = g
            .hw_net_rx_hist
            .iter()
            .zip(g.hw_net_tx_hist.iter())
            .map(|(r, t)| r + t)
            .collect();
        let samples = cpu.len();
        card(ui, |ui| {
            card_title(
                ui,
                "System vitals",
                &if samples > 0 {
                    format!("last {samples}s · /proc")
                } else {
                    String::new()
                },
            );
            if samples < 2 {
                ui.add_space(4.0);
                ui.label(RichText::new("Collecting samples…").color(FAINT));
                ui.add_space(70.0);
                return;
            }
            let last = |s: &[f64]| s.last().copied().unwrap_or(0.0);
            let charts: [(&str, String, Vec<f64>, Color32, fn(f64) -> String); 3] = [
                ("CPU", format!("{:.1}%", last(&cpu)), cpu, ACCENT, |v| format!("{v:.1}%")),
                ("Memory", human_size(last(&mem) as i64), mem, VIOLET, |v| human_size(v as i64)),
                ("Network", format!("{}/s", human_size(last(&net) as i64)), net, CYAN, |v| {
                    format!("{}/s", human_size(v as i64))
                }),
            ];
            let per_row = if ui.available_width() > 720.0 { 3 } else { 1 };
            ui.scope(|ui| {
                ui.spacing_mut().item_spacing.x = 20.0;
                for row in charts.chunks(per_row) {
                    ui.columns(per_row, |cols| {
                        for (col, (label, now, series, color, fmt)) in cols.iter_mut().zip(row) {
                            col.horizontal(|ui| {
                                ui.label(RichText::new(*label).size(12.5).color(MUTED));
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    ui.label(RichText::new(now).font(bold(14.0)).color(TEXT));
                                });
                            });
                            let w = col.available_width();
                            area_chart(col, series, *color, Vec2::new(w, 110.0), *fmt);
                        }
                    });
                }
            });
        });
    }

    /// One-line host summary: uptime, load, temperature, ports, agents,
    /// disk throughput — the "system at a glance" strip.
    fn system_summary(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        let Some(h) = &g.hw else { return };
        let n_listen = g.ports.iter().filter(|p| p.state == "LISTEN").count();
        let n_agents = g.agents.len();
        let temp = h
            .temp_c
            .map(|t| {
                let c = if t > 85.0 { RED } else if t > 70.0 { AMBER } else { GREEN };
                (format!("{t:.0} °C"), c)
            })
            .unwrap_or_else(|| ("n/a".to_string(), FAINT));
        let cells: Vec<(&str, String, Color32)> = vec![
            ("Uptime", uptime_str(h.uptime as i64), TEXT),
            (
                "Load",
                format!("{} · {} · {}", h.load1, h.load5, h.load15),
                TEXT,
            ),
            ("Temp", temp.0, temp.1),
            ("Listening", format!("{n_listen} ports"), CYAN),
            (
                "Agents",
                format!("{n_agents} running"),
                if n_agents > 0 { GREEN } else { FAINT },
            ),
            (
                "Disk I/O",
                format!("↓ {} ↑ {}", human_rate(h.disk_read_bps), human_rate(h.disk_write_bps)),
                TEXT,
            ),
        ];
        card(ui, |ui| {
            let per_row = if ui.available_width() > 980.0 { 6 } else if ui.available_width() > 640.0 { 3 } else { 2 };
            ui.scope(|ui| {
                ui.spacing_mut().item_spacing.x = 14.0;
                for (n, row) in cells.chunks(per_row).enumerate() {
                    if n > 0 {
                        ui.add_space(8.0);
                    }
                    ui.columns(per_row, |cols| {
                        for (col, (label, value, color)) in cols.iter_mut().zip(row) {
                            col.vertical(|ui| {
                                ui.spacing_mut().item_spacing.y = 1.0;
                                ui.label(RichText::new(*label).size(11.5).color(FAINT));
                                ui.add(
                                    egui::Label::new(
                                        RichText::new(value).font(FontId::monospace(13.5)).color(*color),
                                    )
                                    .truncate(),
                                );
                            });
                        }
                    });
                }
            });
        });
    }

    /// Docker one-liner under the section heading: engine state, container
    /// counts, or the offline reason.
    fn docker_strip(&mut self, ui: &mut Ui, ok: bool) {
        let g = &self.gui;
        let n_all = g.containers.len();
        let n_run = g.containers.iter().filter(|c| c.state == "running").count();
        let text = if !ok {
            "Docker engine offline — containers, images and volumes are unavailable".to_string()
        } else {
            match &g.info {
                Some(i) => format!(
                    "Docker {} · {} containers, {} running · {} images · {} volumes",
                    i.server_version,
                    n_all,
                    n_run,
                    g.images.len(),
                    g.volumes.len()
                ),
                None => "Connecting to the Docker engine…".to_string(),
            }
        };
        let color = if ok { MUTED } else { RED };
        ui.horizontal(|ui| {
            let (r, _) = ui.allocate_exact_size(Vec2::splat(8.0), Sense::hover());
            ui.painter().circle_filled(r.center(), 3.5, if ok { GREEN } else { RED });
            ui.label(RichText::new(text).size(12.5).color(color));
        });
    }

    /// Per-container resource charts (CPU/mem/net summed over containers).
    fn docker_charts(&mut self, ui: &mut Ui) {
        let (cpu, mem, net, samples) = self.host_series();
        let n = self.gui.stats.len();
        card(ui, |ui| {
            card_title(
                ui,
                "Resource usage",
                &if samples > 0 {
                    format!("last {samples}s · {n} containers")
                } else {
                    String::new()
                },
            );
            if samples < 2 {
                ui.add_space(4.0);
                ui.label(RichText::new("Collecting samples…").color(FAINT));
                ui.add_space(70.0);
                return;
            }
            let last = |s: &[f64]| s.last().copied().unwrap_or(0.0);
            let charts: [(&str, String, &[f64], Color32, fn(f64) -> String); 3] = [
                ("CPU", format!("{:.1}%", last(&cpu)), &cpu, ACCENT, |v| format!("{v:.1}%")),
                ("Memory", human_size(last(&mem) as i64), &mem, VIOLET, |v| human_size(v as i64)),
                ("Network", format!("{}/s", human_size(last(&net) as i64)), &net, CYAN, |v| {
                    format!("{}/s", human_size(v as i64))
                }),
            ];
            let per_row = if ui.available_width() > 720.0 { 3 } else { 1 };
            ui.scope(|ui| {
                ui.spacing_mut().item_spacing.x = 20.0;
                for row in charts.chunks(per_row) {
                    ui.columns(per_row, |cols| {
                        for (col, (label, now, series, color, fmt)) in cols.iter_mut().zip(row) {
                            col.horizontal(|ui| {
                                ui.label(RichText::new(*label).size(12.5).color(MUTED));
                                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                    ui.label(RichText::new(now).font(bold(14.0)).color(TEXT));
                                });
                            });
                            let w = col.available_width();
                            area_chart(col, series, *color, Vec2::new(w, 110.0), *fmt);
                        }
                    });
                }
            });
        });
    }

    fn top_containers(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        let mut rows: Vec<(String, f64, i64)> = g
            .containers
            .iter()
            .filter(|c| c.state == "running")
            .filter_map(|c| {
                let s = g.stats.get(&c.id)?;
                let cpu = StatsSample::cpu_percent(g.prev_stats.get(&c.id), s).unwrap_or(0.0);
                Some((c.short_name(), cpu.max(0.0), s.mem_usage()))
            })
            .collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let peak = rows.first().map(|r| r.1).unwrap_or(0.0).max(1.0);
        card(ui, |ui| {
            ui.set_min_height(286.0);
            card_title(ui, "Top containers", "CPU · 100% = 1 core");
            if rows.is_empty() {
                ui.label(RichText::new("No running containers with stats yet.").color(FAINT));
                return;
            }
            for (name, cpu, mem) in rows.iter().take(7) {
                let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 30.0), Sense::hover());
                let p = ui.painter();
                let name_w = rect.width() * 0.42;
                let n = elide(p, name.clone(), FontId::proportional(13.0), TEXT, name_w - 8.0);
                p.galley(Pos2::new(rect.left(), rect.center().y - n.size().y / 2.0), n, TEXT);
                let bar = Rect::from_min_max(
                    Pos2::new(rect.left() + name_w, rect.center().y - 3.0),
                    Pos2::new(rect.right() - 128.0, rect.center().y + 3.0),
                );
                // Bars scale to the busiest container so small loads stay visible.
                progress(p, bar, (*cpu / peak) as f32, ACCENT);
                p.text(
                    Pos2::new(rect.right() - 66.0, rect.center().y),
                    egui::Align2::RIGHT_CENTER,
                    format!("{cpu:.1}%"),
                    FontId::monospace(12.0),
                    TEXT,
                );
                p.text(
                    Pos2::new(rect.right(), rect.center().y),
                    egui::Align2::RIGHT_CENTER,
                    human_size(*mem),
                    FontId::monospace(12.0),
                    FAINT,
                );
            }
        });
    }

    /// Recent container lifecycle events, newest first.
    fn event_feed(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        let now = epoch();
        card(ui, |ui| {
            ui.set_min_height(286.0);
            card_title(ui, "Activity", if g.events.is_empty() { "listening" } else { "live" });
            if g.events.is_empty() {
                ui.label(RichText::new("No container events yet. Start or stop something to see it here.").color(FAINT));
                return;
            }
            ScrollArea::vertical()
                .max_height(236.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 0.0;
                    for e in g.events.iter().rev().take(60) {
                        let action = e.action.split(':').next().unwrap_or(&e.action);
                        let color = match action {
                            "start" | "create" | "attach" | "unpause" => GREEN,
                            "die" | "kill" | "stop" | "oom" => RED,
                            "destroy" => AMBER,
                            _ => MUTED,
                        };
                        let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 30.0), Sense::hover());
                        let p = ui.painter();
                        p.circle_filled(Pos2::new(rect.left() + 5.0, rect.center().y), 3.5, color);
                        let when = ago(now - e.time);
                        let when_w = 70.0;
                        let name = elide(
                            p,
                            e.container_name(),
                            FontId::proportional(13.0),
                            TEXT,
                            rect.width() - 20.0 - when_w - 70.0,
                        );
                        let nw = name.size().x;
                        p.galley(Pos2::new(rect.left() + 18.0, rect.center().y - name.size().y / 2.0), name, TEXT);
                        p.text(
                            Pos2::new(rect.left() + 26.0 + nw, rect.center().y),
                            egui::Align2::LEFT_CENTER,
                            action,
                            FontId::proportional(12.5),
                            color,
                        );
                        p.text(
                            Pos2::new(rect.right() - 4.0, rect.center().y),
                            egui::Align2::RIGHT_CENTER,
                            when,
                            FontId::proportional(12.0),
                            FAINT,
                        );
                    }
                });
        });
    }

    fn disk_section(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        card(ui, |ui| {
            let Some(df) = &g.df else {
                card_title(ui, "Disk usage", "");
                ui.label(RichText::new("Collecting…").color(FAINT));
                return;
            };
            let sections: Vec<(&str, &docker::UsageTotals, Color32)> = [
                ("Images", df.image_usage.as_ref(), ACCENT),
                ("Build cache", df.build_cache_usage.as_ref(), AMBER),
                ("Volumes", df.volume_usage.as_ref(), VIOLET),
                ("Containers", df.container_usage.as_ref(), GREEN),
            ]
            .into_iter()
            .filter_map(|(l, u, c)| Some((l, u?, c)))
            .collect();
            let total: i64 = sections.iter().map(|(_, u, _)| u.total_size).sum();
            let reclaim: i64 = sections.iter().map(|(_, u, _)| u.reclaimable).sum();
            card_title(
                ui,
                "Disk usage",
                &format!("{} total · {} reclaimable", human_size(total), human_size(reclaim)),
            );
            // One stacked bar, segments proportional to each category.
            let (bar, _) = ui.allocate_exact_size(Vec2::new(ui.available_width(), 10.0), Sense::hover());
            let p = ui.painter();
            p.rect_filled(bar, 5.0, SURFACE_HI);
            let mut x = bar.left();
            for (_, u, color) in &sections {
                let w = bar.width() * (u.total_size as f32 / total.max(1) as f32);
                if w >= 1.0 {
                    let seg = Rect::from_min_max(Pos2::new(x, bar.top()), Pos2::new((x + w - 2.0).max(x + 1.0), bar.bottom()));
                    p.rect_filled(seg, 3.0, *color);
                }
                x += w;
            }
            ui.add_space(12.0);
            let per_row = if ui.available_width() > 640.0 { 4 } else { 2 };
            ui.scope(|ui| {
                ui.spacing_mut().item_spacing.x = 16.0;
                for row in sections.chunks(per_row) {
                    ui.columns(per_row, |cols| {
                        for (col, (label, u, color)) in cols.iter_mut().zip(row) {
                            col.spacing_mut().item_spacing.y = 2.0;
                            col.horizontal(|ui| {
                                let (r, _) = ui.allocate_exact_size(Vec2::splat(10.0), Sense::hover());
                                ui.painter().rect_filled(r.shrink(1.0), 2.0, *color);
                                ui.label(RichText::new(*label).size(12.5).color(MUTED));
                            });
                            col.label(RichText::new(human_size(u.total_size)).font(bold(16.0)).color(TEXT));
                            col.label(
                                RichText::new(format!("{} reclaimable", human_size(u.reclaimable)))
                                    .size(11.5)
                                    .color(FAINT),
                            );
                        }
                    });
                }
            });
        });
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
            },
            Kpi {
                label: "Network",
                value: format!("↓ {}/s", human_size(rx)),
                sub: format!("↑ {}/s · {} pids", human_size(txb), s.pids_stats.current),
                color: CYAN,
                viz: Viz::Spark(series(&|h| h.rx.iter().zip(&h.tx).map(|(a, b)| (a + b) as f64).collect())),
            },
            Kpi {
                label: "Disk I/O",
                value: format!("{}/s", human_size(br + bw)),
                sub: format!("read {} · write {}", human_size(s.blk_read()), human_size(s.blk_write())),
                color: AMBER,
                viz: Viz::Spark(series(&|h| h.blk_r.iter().zip(&h.blk_w).map(|(a, b)| (a + b) as f64).collect())),
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
                },
                Kpi {
                    label: "Unique",
                    value: human_size((im.size - im.shared_size.max(0)).max(0)),
                    sub: format!("{} shared with other images", human_size(im.shared_size.max(0))),
                    color: VIOLET,
                    viz: Viz::Ratio((im.size - im.shared_size.max(0)).max(0) as f32 / im.size.max(1) as f32),
                },
                Kpi {
                    label: "Containers",
                    value: im.containers.max(0).to_string(),
                    sub: if im.created > 0 { format!("created {}", fmt_epoch(im.created)) } else { String::new() },
                    color: GREEN,
                    viz: Viz::None,
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
            },
            Kpi {
                label: "Memory",
                value: human_size(a.mem_bytes as i64),
                sub: "resident, summed".into(),
                color: VIOLET,
                viz: Viz::None,
            },
            Kpi {
                label: "Uptime",
                value: ago(uptime),
                sub: format!("lead pid {}", a.lead_pid),
                color: ACCENT,
                viz: Viz::None,
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
                "{} cores · {} RAM · up {}",
                s.ncpu,
                human_size(s.mem_total as i64),
                ago(s.uptime as i64)
            ),
        );
        ui.add_space(16.0);

        let mem_used = s.mem_total - s.mem_avail;
        let swap_used = s.swap_total.saturating_sub(s.swap_free);
        let cpu_hist: Vec<f64> = self.gui.hw_cpu_hist.iter().copied().collect();
        let dr_hist: Vec<f64> = self.gui.hw_disk_r_hist.iter().copied().collect();
        let dw_hist: Vec<f64> = self.gui.hw_disk_w_hist.iter().copied().collect();
        let rx_hist: Vec<f64> = self.gui.hw_net_rx_hist.iter().copied().collect();
        let tx_hist: Vec<f64> = self.gui.hw_net_tx_hist.iter().copied().collect();

        let kpis = [
            Kpi {
                label: "CPU",
                value: format!("{:.1}%", s.cpu_pct),
                sub: format!("load {} · {} · {}", s.load1, s.load5, s.load15),
                color: ACCENT,
                viz: Viz::Spark(cpu_hist),
            },
            Kpi {
                label: "Memory",
                value: human_size(mem_used as i64),
                sub: format!("of {} ({:.0}%)", human_size(s.mem_total as i64), mem_used as f64 / s.mem_total.max(1) as f64 * 100.0),
                color: VIOLET,
                viz: Viz::Ratio(mem_used as f32 / s.mem_total.max(1) as f32),
            },
            Kpi {
                label: "Disk I/O",
                value: format!("{} · {}", human_rate(s.disk_read_bps), human_rate(s.disk_write_bps)),
                sub: "read · write per second".into(),
                color: GREEN,
                viz: Viz::Spark(dr_hist.into_iter().zip(dw_hist).map(|(r, w)| r + w).collect()),
            },
            Kpi {
                label: "Network",
                value: format!("{} · {}", human_rate(s.net_rx_bps), human_rate(s.net_tx_bps)),
                sub: "down · up per second".into(),
                color: AMBER,
                viz: Viz::Spark(rx_hist.into_iter().zip(tx_hist).map(|(r, t)| r + t).collect()),
            },
        ];
        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.x = 14.0;
            ui.columns(4, |cols| {
                for (col, k) in cols.iter_mut().zip(&kpis) {
                    kpi_card(col, k);
                }
            });
        });
        ui.add_space(14.0);

        card(ui, |ui| {
            card_title(
                ui,
                &format!("Per-core load ({})", s.per_core.len()),
                &s.temp_c.map(|t| format!("{t:.0} °C")).unwrap_or_default(),
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

        ui.scope(|ui| {
            ui.spacing_mut().item_spacing.x = 14.0;
            ui.columns(2, |cols| {
                card(&mut cols[0], |ui| {
                    card_title(ui, "Memory & swap", "");
                    bar_row(ui, "Used", mem_used, s.mem_total, VIOLET);
                    bar_row(ui, "Swap", swap_used, s.swap_total, AMBER);
                });
                card(&mut cols[1], |ui| {
                    card_title(ui, "Throughput", "");
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
                    if let Some(t) = s.temp_c {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Temp ").size(12.5).color(MUTED).weak());
                            let c = if t > 85.0 { RED } else if t > 70.0 { AMBER } else { GREEN };
                            ui.label(RichText::new(format!("{t:.0} °C")).font(FontId::monospace(12.5)).color(c));
                            ui.label(RichText::new("hottest sensor").size(11.5).color(FAINT));
                        });
                    }
                });
            });
        });
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
            },
            Kpi {
                label: "Protocol",
                value: p.proto.to_uppercase(),
                sub: format!("IPv{}", p.family),
                color: ACCENT,
                viz: Viz::None,
            },
            Kpi {
                label: "Bind",
                value: if p.addr == "0.0.0.0" || p.addr == "[::]" { "public" } else if p.addr.starts_with("127.") || p.addr == "[::1]" { "localhost" } else { "interface" }.into(),
                sub: p.addr.clone(),
                color: if p.addr == "0.0.0.0" || p.addr == "[::]" { AMBER } else { GREEN },
                viz: Viz::None,
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
            },
            Kpi {
                label: "References",
                value: refs.map(|n| n.to_string()).unwrap_or_else(|| "—".into()),
                sub: if refs.is_some() { "containers mounting it".into() } else { "not reported by the daemon".into() },
                color: GREEN,
                viz: Viz::None,
            },
            Kpi {
                label: "Created",
                value: v.created.get(..10).unwrap_or("—").to_string(),
                sub: v.created.get(11..19).unwrap_or("").to_string(),
                color: ACCENT,
                viz: Viz::None,
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

/// Dashboard / detail metric tile.
struct Kpi {
    label: &'static str,
    value: String,
    sub: String,
    color: Color32,
    viz: Viz,
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

/// Short OS label for the dashboard header: prefers the kernel name from
/// /proc (Linux + release) over the Docker-reported string.
fn host_os_name() -> String {
    if let Ok(s) = std::fs::read_to_string("/proc/sys/kernel/ostype") {
        let name = s.trim();
        if let Ok(rel) = std::fs::read_to_string("/proc/sys/kernel/osrelease") {
            return format!("{name} {}", rel.trim());
        }
        return name.to_string();
    }
    "Linux".to_string()
}

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
                    let _ = tx.send(Msg::Events(Err("events: idle".into())));
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
                        let _ = tx.send(Msg::Events(Err("event stream ended".into())));
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
