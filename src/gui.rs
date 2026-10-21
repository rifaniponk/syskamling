//! Graphical desktop front-end (egui/eframe).
//!
//! Same engine facade as the TUI: all Docker calls run on the tokio runtime
//! and report back through an mpsc channel, so the window never blocks.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::{
    Color32, FontData, FontFamily, FontId, Frame, Pos2, Rect, RichText, ScrollArea, Sense, Stroke,
    TextStyle, Ui, Vec2, WidgetText,
};

use tokio::sync::mpsc;

use crate::docker::{
    self, ContainerSummary, DiskSummary, Docker, DockerError, EventMessage, ImageSummary,
    LogLine, StatsSample, StreamKind, SystemInfo,
};

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
    selected_image: Option<String>,
    selected_volume: Option<String>,
    last_refresh: String,
    daemon_ok: bool,
    docker_version: String,
    // --- dashboard data ---
    info: Option<SystemInfo>,
    df: Option<DiskSummary>,
    events: Vec<EventMessage>,
    events_active: bool,
    stats: HashMap<String, StatsSample>,
    prev_stats: HashMap<String, StatsSample>,
    hist: HashMap<String, StatHist>,
    stats_gen: u64,
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
            selected_image: None,
            selected_volume: None,
            last_refresh: String::new(),
            daemon_ok: false,
            docker_version: String::new(),
            info: None,
            df: None,
            events: Vec::new(),
            events_active: false,
            stats: HashMap::new(),
            prev_stats: HashMap::new(),
            hist: HashMap::new(),
            stats_gen: 0,
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
struct SddApp {
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
    /// when each stats stream task was spawned (grace for first sample)
    stats_spawned: HashMap<String, Instant>,
    /// container ids with a live log-follow task feeding the consolidated page
    all_logs_follow: Vec<String>,
}

impl SddApp {
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
        };
        app
    }
}

/// Dark palette tuned to the Docker-blue accent, roomier spacing.
fn apply_theme(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();

    style.visuals.dark_mode = true;
    style.visuals.extreme_bg_color = Color32::from_rgb(12, 16, 22);
    style.visuals.panel_fill = Color32::from_rgb(18, 22, 30);
    style.visuals.window_fill = Color32::from_rgb(24, 29, 39);
    style.visuals.faint_bg_color = Color32::from_rgb(26, 31, 41);
    style.visuals.override_text_color = Some(Color32::from_rgb(214, 220, 228));
    style.visuals.window_corner_radius = egui::CornerRadius::same(10);
    style.visuals.widgets.hovered.weak_bg_fill = Color32::from_rgb(38, 46, 60);
    style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(46, 56, 74);
    style.visuals.widgets.inactive.weak_bg_fill = Color32::from_rgb(30, 36, 48);
    style.visuals.widgets.inactive.bg_fill = Color32::from_rgb(42, 52, 70);
    style.visuals.widgets.active.weak_bg_fill = Color32::from_rgb(34, 60, 92);
    style.visuals.widgets.active.bg_fill = Color32::from_rgb(48, 90, 140);
    style.visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(18, 22, 30);
    style.visuals.widgets.noninteractive.fg_stroke.color = Color32::from_rgb(214, 220, 228);
    style.visuals.selection.bg_fill = Color32::from_rgba_premultiplied(40, 90, 150, 140);
    style.visuals.selection.stroke.color = accent();
    style.visuals.window_shadow = egui::epaint::Shadow {
        offset: [0, 6],
        blur: 18,
        spread: 2,
        color: Color32::from_rgba_premultiplied(0, 0, 0, 140),
    };
    style.visuals.popup_shadow = style.visuals.window_shadow;

    // egui's built-in open/close animations (button press, slider handle,
    // window fade) — cheap micro-interactions that come free with repaints.
    style.animation_time = 0.18;

    style.spacing.item_spacing = Vec2::new(8.0, 6.0);
    style.spacing.button_padding = Vec2::new(12.0, 7.0);

    style.text_styles.insert(
        TextStyle::Heading,
        FontId::new(21.0, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Body,
        FontId::new(14.5, FontFamily::Proportional),
    );
    style.text_styles.insert(
        TextStyle::Button,
        FontId::new(14.0, FontFamily::Proportional),
    );
    style.text_styles.insert(TextStyle::Monospace, FontId::new(13.0, FontFamily::Monospace));
    style.text_styles.insert(TextStyle::Small, FontId::new(12.0, FontFamily::Proportional));

    ctx.set_style(style);

    // Prefer a system monospace (DejaVu Sans Mono covers the box-drawing and
    // geometric glyphs we use for status dots and button icons).
    let mut fonts = egui::FontDefinitions::default();
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ] {
        if let Ok(bytes) = std::fs::read(path) {
            let name = format!(
                "sys-{}",
                std::path::Path::new(path).file_stem().unwrap().to_string_lossy()
            );
            fonts.font_data.insert(name.clone(), FontData::from_owned(bytes).into());
            fonts
                .families
                .entry(FontFamily::Monospace)
                .or_default()
                .insert(0, name);
        }
    }
    ctx.set_fonts(fonts);
}

impl eframe::App for SddApp {
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
            ctx.request_repaint();
        }

        // Animation bookkeeping: ease status dots toward their target,
        // expire the spin, and schedule fast repaints while anything moves.
        let animating = self.advance_animations();
        if animating {
            ctx.request_repaint_after(Duration::from_millis(33));
        }

        self.top_bar(ctx);
        self.side_bar(ctx);
        self.center(ctx);
        self.status_bar(ctx);
        if let Some(id) = self.gui.confirm_remove.clone() {
            self.remove_dialog(ctx, &id);
        }
    }
}

impl SddApp {
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
        // containers get a grace period so a task that hasn't delivered its
        // first sample yet isn't reaped (respawning would churn sockets).
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

impl SddApp {
    fn apply(&mut self, m: Msg, ctx: &egui::Context) {
        let g = &mut self.gui;
        match m {
            Msg::Refreshed(Ok(data)) => {
                g.error = None;
                g.daemon_ok = true;
                if g.status.contains("refreshing") {
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
        }
    }

    fn top_bar(&mut self, ctx: &egui::Context) {
        let g = &mut self.gui;
        let mut new_tab: Option<Tab> = None;
        let mut refresh_now = false;
        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading(RichText::new("SDD").color(accent()));
                ui.label(
                    RichText::new(if g.daemon_ok {
                        format!("Docker {}", g.docker_version)
                    } else {
                        "daemon unreachable".to_string()
                    })
                    .weak(),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.button("⟳").on_hover_text("refresh now").clicked() {
                        refresh_now = true;
                    }
                    ui.separator();
                    // right-to-left: declare in reverse so they render L→R
                    if let Some(t) = tab_button(ui, g.tab, "Logs", Tab::Logs) {
                        new_tab = Some(t);
                    }
                    if let Some(t) = tab_button(ui, g.tab, "Volumes", Tab::Volumes) {
                        new_tab = Some(t);
                    }
                    if let Some(t) = tab_button(ui, g.tab, "Images", Tab::Images) {
                        new_tab = Some(t);
                    }
                    if let Some(t) = tab_button(ui, g.tab, "Containers", Tab::Containers) {
                        new_tab = Some(t);
                    }
                    if let Some(t) = tab_button(ui, g.tab, "Dashboard", Tab::Dashboard) {
                        new_tab = Some(t);
                    }
                });
            });
        });
        if let Some(t) = new_tab {
            if t != self.gui.tab {
                self.on_tab_change(t);
            }
        }
        if refresh_now {
            spawn_refresh(&self.rt, &self.docker, &self.tx);
            self.spawn_dashboard_pulls();
            self.gui.status = "refreshing...".to_string();
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

    fn side_bar(&mut self, ctx: &egui::Context) {
        let tab = self.gui.tab;
        egui::SidePanel::left("side").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let w = ui.available_width() - 8.0;
                ui.add(
                    egui::TextEdit::singleline(&mut self.gui.filter)
                        .hint_text("filter")
                        .desired_width(w),
                );
            });
            if tab == Tab::Containers {
                let n_all = self.gui.containers.len();
                let n_run = self
                    .gui
                    .containers
                    .iter()
                    .filter(|c| c.state == "running")
                    .count();
                let cur = self.gui.state_filter;
                let mut set = None;
                ui.horizontal(|ui| {
                    if chip(ui, cur == StateFilter::Running, format!("● running {n_run}")) {
                        set = Some(StateFilter::Running);
                    }
                    if chip(
                        ui,
                        cur == StateFilter::Stopped,
                        format!("○ stopped {}", n_all - n_run),
                    ) {
                        set = Some(StateFilter::Stopped);
                    }
                    if chip(ui, cur == StateFilter::All, format!("all {n_all}")) {
                        set = Some(StateFilter::All);
                    }
                });
                if let Some(f) = set {
                    self.gui.state_filter = f;
                    self.gui.selected = None;
                }
            }
            ui.separator();
            match tab {
                Tab::Dashboard => self.event_feed(ui),
                Tab::Containers => self.container_list(ui),
                Tab::Images => self.image_list(ui),
                Tab::Volumes => self.volume_list(ui),
                Tab::Logs => self.logs_summary(ui),
            }
        });
    }

    /// Compact lifecycle feed shown in the dashboard sidebar.
    fn event_feed(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        ui.label(RichText::new("recent events").strong().small());
        if !g.events_active {
            ui.label(RichText::new("listening…").weak().small());
        }
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for e in g.events.iter().rev().take(80) {
                    let action = e.action.split(':').next().unwrap_or(&e.action);
                    let colored = match action {
                        "start" | "create" | "attach" => Color32::from_rgb(110, 210, 150),
                        "die" | "kill" | "stop" => Color32::from_rgb(230, 130, 110),
                        _ => Color32::from_rgb(150, 155, 165),
                    };
                    let t = ((e.time % 86400) / 3600, (e.time % 3600) / 60, e.time % 60);
                    ui.label(
                        RichText::new(format!(
                            "{:02}:{:02}:{:02}  {} · {}",
                            t.0,
                            t.1,
                            t.2,
                            e.container_name(),
                            action
                        ))
                        .small()
                        .color(colored),
                    );
                }
            });
    }

    fn container_list(&mut self, ui: &mut Ui) {
        // Snapshot what the row widgets need so the closure never borrows self.
        let rows = self.gui.filtered();
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
                for (i, c) in rows {
                    let running = c.state == "running";
                    let is_sel = sel == Some(i);
                    let busy = !busy_id.is_empty() && busy_id == c.id;
                    let pulse = dot_anim.get(&c.id).copied().unwrap_or(0.0);
                    let frame = Frame::NONE.fill(row_fill(is_sel, pulse));
                    // Make the whole row a click target: lay the widgets out
                    // first, then overlay an invisible click rect across the
                    // full row so clicking anywhere (dot, name, state label, or
                    // empty space) selects it.
                    let row_id = ui.id().with(("row", &c.id));
                    frame.show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(dot(running, pulse));
                            if busy {
                                ui.add(egui::Spinner::new().size(14.0).color(accent()));
                            }
                            let mk = || {
                                let t = RichText::new(c.short_name()).strong();
                                if busy {
                                    RichText::new(t.text()).weak()
                                } else {
                                    t
                                }
                            };
                            ui.add(tx(mk()));
                            // live mini sparkline + cpu/mem for running rows
                            if running {
                                if let Some(h) = hist.get(&c.id) {
                                    if h.cpu.len() >= 2 {
                                        sparkline(ui, &h.cpu, accent(), Vec2::new(46.0, 14.0));
                                    }
                                }
                                if let Some(s) = stats.get(&c.id) {
                                    let cpu = StatsSample::cpu_percent(
                                        prev_stats.get(&c.id),
                                        s,
                                    )
                                    .unwrap_or(0.0);
                                    ui.label(
                                        RichText::new(format!("{cpu:.0}%"))
                                            .small()
                                            .monospace(),
                                    );
                                    if let Some(mp) = s.mem_percent() {
                                        ui.label(
                                            RichText::new(format!("{:.0}%", mp))
                                                .small()
                                                .monospace()
                                                .weak(),
                                        );
                                    }
                                }
                            }
                            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                                let state = RichText::new(&c.state).small().color(if running {
                                    Color32::from_rgb(110, 210, 150)
                                } else {
                                    Color32::from_rgb(150, 155, 165)
                                });
                                ui.label(state);
                            });
                        });
                        let row_rect = ui.min_rect();
                        let click = ui.interact(row_rect, row_id, Sense::click());
                        if click.hovered() {
                            click.clone().on_hover_text(format!(
                                "{}\n{}\n{}\n{}",
                                c.short_name(),
                                c.image,
                                c.status,
                                c.port_summary()
                            ));
                        }
                        if click.clicked() {
                            new_sel = Some(i);
                        }
                    });
                }
            });

        if let Some(i) = new_sel {
            self.gui.selected = Some(i);
        }
    }

    fn image_list(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        let f = g.filter.to_lowercase();
        let sel = g.selected_image.clone();
        let mut new_sel: Option<Option<String>> = None;
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for im in &g.images {
                    let tag = im.tag();
                    if !f.is_empty() && !tag.to_lowercase().contains(&f) {
                        continue;
                    }
                    let is_sel = sel.as_deref() == Some(im.id.as_str());
                    let frame = Frame::NONE.fill(if is_sel {
                        Color32::from_rgb(34, 60, 92)
                    } else {
                        Color32::TRANSPARENT
                    });
                    let row_id = ui.id().with(("img", &im.id));
                    frame.show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(&tag).strong());
                        });
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(format!("{}  {}", im.short_id(), human_size(im.size)))
                                    .weak(),
                            );
                            if im.containers > 0 {
                                ui.label(
                                    RichText::new(format!("{} containers", im.containers))
                                        .small()
                                        .weak(),
                                );
                            }
                        });
                        ui.separator();
                        let click = ui.interact(ui.min_rect(), row_id, Sense::click());
                        if click.clicked() {
                            new_sel = Some(if is_sel { None } else { Some(im.id.clone()) });
                        }
                    });
                }
            });
        if let Some(s) = new_sel {
            self.gui.selected_image = s;
        }
    }

    fn volume_list(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        let f = g.filter.to_lowercase();
        let sel = g.selected_volume.clone();
        let mut new_sel: Option<Option<String>> = None;
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for v in &g.volumes {
                    if !f.is_empty() && !v.name.to_lowercase().contains(&f) {
                        continue;
                    }
                    let is_sel = sel.as_deref() == Some(v.name.as_str());
                    let frame = Frame::NONE.fill(if is_sel {
                        Color32::from_rgb(34, 60, 92)
                    } else {
                        Color32::TRANSPARENT
                    });
                    let row_id = ui.id().with(("vol", &v.name));
                    frame.show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.label(RichText::new(&v.name).strong());
                        });
                        ui.horizontal(|ui| {
                            let size = v
                                .usage_data
                                .as_ref()
                                .and_then(|u| u.size)
                                .map(human_size)
                                .unwrap_or_else(|| "?".into());
                            ui.label(
                                RichText::new(format!(
                                    "driver={} · {} · {} refs",
                                    v.driver,
                                    size,
                                    v.usage_data.as_ref().map(|u| u.ref_count).unwrap_or(0)
                                ))
                                .weak(),
                            );
                        });
                        ui.separator();
                        let click = ui.interact(ui.min_rect(), row_id, Sense::click());
                        if click.clicked() {
                            new_sel = Some(if is_sel { None } else { Some(v.name.clone()) });
                        }
                    });
                }
            });
        if let Some(s) = new_sel {
            self.gui.selected_volume = s;
        }
    }

    fn logs_summary(&mut self, ui: &mut Ui) {
        let g = &mut self.gui;
        if g.logs_container.is_none() {
            ui.label(
                RichText::new(
                    "Select a container to see its logs here, or open this tab to load all containers.",
                )
                .weak(),
            );
            return;
        }
        let all = g.logs_container.as_deref() == Some("__all__");
        ui.label(RichText::new(&g.logs_container_name).strong());
        ui.label(
            RichText::new(format!(
                "{} lines · {}",
                g.logs.len(),
                if g.logs_active { "streaming" } else { "stopped" }
            ))
            .weak(),
        );
        ui.add_space(6.0);
        if ui
            .button(if g.logs_follow { "Pause follow" } else { "Resume follow" })
            .clicked()
        {
            g.logs_follow = !g.logs_follow;
        }
        let reload_all = all && ui.button("Reload all logs").clicked();
        let reopen = !all && g.logs_container.is_some() && ui.button("Reopen stream").clicked();
        let to_all = !all && ui.button("Show all containers").clicked();
        if reload_all || to_all {
            self.load_all_logs();
        } else if reopen {
            let id = g.logs_container.clone().unwrap();
            self.open_logs(&id);
        }
    }

    fn image_detail(&mut self, ui: &mut Ui) {
        let Some(id) = self.gui.selected_image.clone() else {
            ui.label(RichText::new("Select an image in the list to see details.").weak());
            return;
        };
        let Some(im) = self.gui.images.iter().find(|i| i.id == id).cloned() else {
            ui.label(RichText::new("That image no longer exists.").weak());
            return;
        };
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(im.tag()).heading());
                    ui.label(
                        RichText::new(format!("· {}", im.short_id()))
                            .weak()
                            .monospace(),
                    );
                });
                ui.add_space(4.0);
                let avail = ui.available_width();
                let col_w = (avail - 16.0) / 2.0;
                ui.horizontal(|ui| {
                    ui.vertical(|ui| {
                        ui.set_width(col_w);
                        detail_row(ui, "size", &human_size(im.size));
                        detail_row(ui, "shared size", &human_size(im.shared_size));
                        detail_row(ui, "unique size", &human_size(im.size - im.shared_size));
                        detail_row(ui, "used by", &format!("{} containers", im.containers));
                        if im.created > 0 {
                            detail_row(ui, "created", &fmt_epoch(im.created));
                        }
                    });
                    ui.vertical(|ui| {
                        ui.set_width(col_w);
                        ui.label(RichText::new("tags").strong().small());
                        if im.repo_tags.is_empty() {
                            ui.label(RichText::new("<none>").weak().monospace());
                        } else {
                            for t in &im.repo_tags {
                                ui.label(RichText::new(t).monospace().small());
                            }
                        }
                        ui.add_space(6.0);
                        ui.label(RichText::new("used by containers").strong().small());
                        let users: Vec<String> = self
                            .gui
                            .containers
                            .iter()
                            .filter(|c| c.image == im.tag() || c.image.contains(im.short_id()))
                            .map(|c| c.short_name())
                            .collect();
                        if users.is_empty() {
                            ui.label(RichText::new("none").weak().small());
                        } else {
                            for u in users.iter().take(20) {
                                ui.label(RichText::new(u).monospace().small());
                            }
                        }
                    });
                });
                ui.add_space(8.0);
                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(RichText::new("disk share").strong().small());
                    let total = self
                        .gui
                        .df
                        .as_ref()
                        .map(|d| d.layers_size)
                        .filter(|v| *v > 0)
                        .unwrap_or_else(|| self.gui.images.iter().map(|i| i.size).sum::<i64>().max(1));
                    let frac = (im.size as f64 / total.max(1) as f64).clamp(0.0, 1.0);
                    gauge(ui, "", frac * 100.0, accent(), 100.0);
                    ui.label(
                        RichText::new(format!(
                            "of {} total layers",
                            human_size(total)
                        ))
                        .small()
                        .weak(),
                    );
                });
            });
    }

    fn volume_detail(&mut self, ui: &mut Ui) {
        let Some(name) = self.gui.selected_volume.clone() else {
            ui.label(RichText::new("Select a volume in the list to see details.").weak());
            return;
        };
        let Some(v) = self.gui.volumes.iter().find(|v| v.name == name).cloned() else {
            ui.label(RichText::new("That volume no longer exists.").weak());
            return;
        };
        ui.horizontal(|ui| {
            ui.label(RichText::new(&v.name).heading());
        });
        ui.add_space(4.0);
        detail_row(ui, "driver", &v.driver);
        detail_row(
            ui,
            "size",
            &v.usage_data
                .as_ref()
                .and_then(|u| u.size)
                .map(human_size)
                .unwrap_or_else(|| "not reported by driver".into()),
        );
        detail_row(
            ui,
            "references",
            &v.usage_data
                .as_ref()
                .map(|u| u.ref_count.to_string())
                .unwrap_or_else(|| "?".into()),
        );
        if !v.created.is_empty() {
            detail_row(ui, "created", &v.created);
        }
        ui.add_space(8.0);
        ui.separator();
        ui.label(RichText::new("in use by").strong().small());
        // Which containers mount this volume: scan the container list's
        // mount info if present, else fall back to name-prefix matching.
        let users: Vec<String> = self
            .gui
            .containers
            .iter()
            .filter(|c| c.names.iter().any(|n| n.contains(&v.name)))
            .map(|c| c.short_name())
            .collect();
        if users.is_empty() {
            ui.label(
                RichText::new(
                    "no direct container matches (list endpoint omits mounts; \
                     inspect via docker CLI for exact mounts)",
                )
                .weak()
                .small(),
            );
        } else {
            for u in users {
                ui.label(RichText::new(u).monospace().small());
            }
        }
    }

    fn center(&mut self, ctx: &egui::Context) {
        let tab = self.gui.tab;
        egui::CentralPanel::default().show(ctx, |ui| match tab {
            Tab::Dashboard => self.dashboard_view(ui),
            Tab::Logs => self.logs_view(ui),
            Tab::Containers => self.container_detail(ui),
            Tab::Images => self.image_detail(ui),
            Tab::Volumes => self.volume_detail(ui),
        });
    }

    // ------------------------------------------------------------------
    // Dashboard
    // ------------------------------------------------------------------

    fn dashboard_view(&mut self, ui: &mut Ui) {
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.stat_cards(ui);
                ui.add_space(10.0);
                self.system_charts(ui);
                ui.add_space(10.0);
                self.top_containers(ui);
                ui.add_space(10.0);
                self.disk_section(ui);
            });
    }

    fn stat_cards(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        let n_run = g.containers.iter().filter(|c| c.state == "running").count();
        let n_stop = g.containers.len() - n_run;
        let n_img = g.images.len();
        let n_vol = g.volumes.len();
        // cpu% is per-core normalized (0..100 per container), so divide the
        // sum by core count to get a host-wide percentage.
        let ncpu = g.info.as_ref().map(|i| i.ncpu.max(1)).unwrap_or(1) as f64;
        let cpu_sum: f64 = g
            .containers
            .iter()
            .filter(|c| c.state == "running")
            .filter_map(|c| {
                StatsSample::cpu_percent(g.prev_stats.get(&c.id), g.stats.get(&c.id)?)
            })
            .sum();
        let cpu_total = cpu_sum / ncpu;
        let mem_total: i64 = g
            .containers
            .iter()
            .filter(|c| c.state == "running")
            .filter_map(|c| g.stats.get(&c.id).map(|s| s.mem_usage()))
            .sum();
        let host_mem = g
            .info
            .as_ref()
            .map(|i| human_size(i.mem_total))
            .unwrap_or_else(|| "host".into());

        let avail = ui.available_width();
        let card_w = ((avail - 4.0 * 8.0) / 5.0).max(140.0);
        ui.horizontal(|ui| {
            ui.set_min_height(74.0);
            stat_card(ui, "running", &n_run.to_string(), "containers", card_w, Color32::from_rgb(90, 200, 130));
            stat_card(ui, "stopped", &n_stop.to_string(), "containers", card_w, Color32::from_rgb(150, 155, 165));
            stat_card(ui, "cpu", &format!("{:.1}%", cpu_total), &format!("of {ncpu:.0} cores"), card_w, accent());
            stat_card(ui, "memory", &human_size(mem_total), &host_mem, card_w, Color32::from_rgb(190, 140, 240));
            stat_card(ui, "images", &n_img.to_string(), &format!("{} volumes", n_vol), card_w, Color32::from_rgb(230, 170, 90));
        });
    }

    fn system_charts(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        let mut cpu_series: Vec<f64> = vec![0.0; HIST];
        let mut mem_series: Vec<f64> = vec![0.0; HIST];
        let mut net_series: Vec<f64> = vec![0.0; HIST];
        let mut any = false;
        for (id, h) in &g.hist {
            if !g.stats.contains_key(id) {
                continue;
            }
            any = true;
            let off = HIST - h.cpu.len();
            for (i, v) in h.cpu.iter().enumerate() {
                cpu_series[off + i] += v;
            }
            let off = HIST - h.mem.len();
            for (i, v) in h.mem.iter().enumerate() {
                mem_series[off + i] += *v as f64;
            }
            let off = HIST - h.rx.len();
            for (i, v) in h.rx.iter().enumerate() {
                net_series[off + i] += *v as f64;
            }
        }
        ui.label(RichText::new("cluster activity").strong());
        if !any {
            ui.label(
                RichText::new("collecting samples… (charts appear once stats stream in)")
                    .weak(),
            );
            return;
        }
        let avail = ui.available_width();
        let w = (avail - 8.0) / 2.0;
        let h = 130.0;
        let cpu_peak = cpu_series.iter().cloned().fold(0.0_f64, f64::max);
        let mem_peak = mem_series.iter().cloned().fold(0.0_f64, f64::max);
        let net_peak = net_series.iter().cloned().fold(0.0_f64, f64::max);
        let samples = g.hist.values().map(|h| h.cpu.len()).max().unwrap_or(0);
        ui.label(
            RichText::new(format!("last {samples}s across {} containers", g.hist.len()))
                .small()
                .weak(),
        );
        ui.label(
            RichText::new("cpu chart sums per-core-normalized usage (32 cores = 3200%)")
                .small()
                .weak(),
        );
        ui.horizontal(|ui| {
            area_chart(ui, "cpu %", &cpu_series, accent(), Vec2::new(w, h));
            ui.label(RichText::new(format!("peak {cpu_peak:.0}%")).small().weak());
        });
        ui.horizontal(|ui| {
            area_chart(ui, "memory", &mem_series, Color32::from_rgb(190, 140, 240), Vec2::new(w, h));
            ui.label(RichText::new(format!("peak {}", human_size(mem_peak as i64))).small().weak());
        });
        ui.horizontal(|ui| {
            area_chart(ui, "net rx/s", &net_series, Color32::from_rgb(90, 200, 130), Vec2::new(w, h));
            ui.label(RichText::new(format!("peak {}/s", human_size(net_peak as i64))).small().weak());
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
                Some((c.short_name(), cpu, s.mem_usage()))
            })
            .collect();
        rows.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ui.label(RichText::new("top containers by cpu").strong());
        if rows.is_empty() {
            ui.label(RichText::new("no running containers with stats yet").weak());
            return;
        }
        let avail = ui.available_width() - 16.0;
        for (name, cpu, mem) in rows.iter().take(8) {
            ui.horizontal(|ui| {
                ui.label(RichText::new(name).monospace().small());
                let frac = (cpu / 100.0).clamp(0.0, 1.0) as f32;
                let bar_w = (avail * 0.45).max(80.0);
                let (rect, _) = ui.allocate_exact_size(Vec2::new(bar_w, 10.0), Sense::hover());
                let paint = ui.painter();
                paint.rect_filled(rect, 3.0, Color32::from_rgb(30, 36, 48));
                if frac > 0.0 {
                    let r = Rect::from_min_size(rect.min, Vec2::new(rect.width() * frac, rect.height()));
                    paint.rect_filled(r, 3.0, accent());
                }
                ui.label(RichText::new(format!("{cpu:.1}%")).monospace().small());
                ui.label(RichText::new(human_size(*mem)).monospace().small().weak());
            });
        }
    }

    fn disk_section(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        ui.label(RichText::new("disk usage").strong());
        let Some(df) = &g.df else {
            ui.label(RichText::new("collecting…").weak());
            return;
        };
        let sections: [(&str, Option<&docker::UsageTotals>, Color32); 4] = [
            ("images", df.image_usage.as_ref(), accent()),
            ("build cache", df.build_cache_usage.as_ref(), Color32::from_rgb(230, 170, 90)),
            ("volumes", df.volume_usage.as_ref(), Color32::from_rgb(190, 140, 240)),
            ("containers", df.container_usage.as_ref(), Color32::from_rgb(90, 200, 130)),
        ];
        let total: i64 = sections.iter().filter_map(|(_, u, _)| u.map(|u| u.total_size)).sum();
        let total = total.max(1);
        let avail = ui.available_width() - 16.0;
        for (label, u, color) in sections {
            let Some(u) = u else { continue };
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("{label:<12}")).monospace().small());
                let (rect, _) = ui.allocate_exact_size(Vec2::new(avail * 0.55, 12.0), Sense::hover());
                let paint = ui.painter();
                paint.rect_filled(rect, 3.0, Color32::from_rgb(30, 36, 48));
                let frac = (u.total_size as f64 / total as f64).clamp(0.0, 1.0) as f32;
                let r = Rect::from_min_size(rect.min, Vec2::new(rect.width() * frac, rect.height()));
                paint.rect_filled(r, 3.0, color);
                ui.label(
                    RichText::new(format!(
                        "{} · {} reclaimable",
                        human_size(u.total_size),
                        human_size(u.reclaimable)
                    ))
                    .small()
                    .weak(),
                );
            });
        }
    }


    // ------------------------------------------------------------------
    // Container detail: identity, live stats, inline logs
    // ------------------------------------------------------------------

    fn container_detail(&mut self, ui: &mut Ui) {
        // Header + actions + stats snapshot, then the inline log page below.
        let c = match self.gui.selected_container().cloned() {
            Some(c) => c,
            None => {
                ui.label(RichText::new("Select a container to see details, live stats and logs.").weak());
                return;
            }
        };
        let g = &self.gui;
        let running = c.state == "running";
        let busy = g.pending_action.as_ref().is_some_and(|(i, _)| *i == c.id);
        let pulse = self.dot_anim.get(&c.id).copied().unwrap_or(0.0);

        ui.horizontal(|ui| {
            ui.label(dot(running, pulse));
            ui.label(RichText::new(c.short_name()).heading());
            if let Some(p) = c.project() {
                ui.label(RichText::new(format!("· {p}")).weak());
            }
            if busy {
                ui.add(egui::Spinner::new().size(16.0).color(accent()));
            }
        });
        ui.horizontal_wrapped(|ui| {
            let start_stop = if running {
                action_button(ui, "■ Stop", Color32::from_rgb(230, 140, 80), busy)
            } else {
                action_button(ui, "▶ Start", Color32::from_rgb(90, 200, 130), busy)
            };
            let mut action = None;
            if start_stop.clicked() {
                action = Some(if running { Pending::Stop } else { Pending::Start });
            }
            if action_button(ui, "↻ Restart", accent(), busy).clicked() {
                action = Some(Pending::Restart);
            }
            if action_button(ui, "✕ Remove", Color32::from_rgb(220, 90, 90), busy).clicked() {
                self.gui.confirm_remove = Some(c.id.clone());
            }
            if let Some(p) = action {
                self.gui.pending_action = Some((c.id.clone(), p));
                self.spin = Some((c.id.clone(), Instant::now()));
                spawn_action(&self.rt, &self.docker, &self.tx, &c, p);
            }
        });
        ui.add_space(4.0);

        // Two columns: facts on the left, live stats on the right.
        let avail = ui.available_width();
        let col_w = (avail - 16.0) / 2.0;
        ui.horizontal(|ui| {
            ui.vertical(|ui| {
                ui.set_width(col_w);
                detail_row(ui, "image", &c.image);
                detail_row(ui, "state", &c.status);
                detail_row(ui, "id", &c.id[..12.min(c.id.len())]);
                if !c.port_summary().is_empty() {
                    detail_row(ui, "ports", &c.port_summary());
                }
                if let Some(sr) = c.size_rw {
                    detail_row(ui, "rw size", &human_size(sr));
                }
                if c.created > 0 {
                    detail_row(ui, "created", &fmt_epoch(c.created));
                }
            });
            ui.vertical(|ui| {
                ui.set_width(col_w);
                self.detail_stats(ui, &c);
            });
        });

        ui.add_space(6.0);
        ui.separator();
        ui.label(RichText::new("logs").strong());
        self.logs_page(ui);
    }

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
            if c.state == "running" {
                ui.label(RichText::new("collecting stats…").weak());
            } else {
                ui.label(RichText::new("not running — no live stats").weak());
            }
            return;
        };
        let cpu = StatsSample::cpu_percent(g.prev_stats.get(&c.id), s).unwrap_or(0.0);
        let mem_p = s.mem_percent().unwrap_or(0.0);
        ui.horizontal(|ui| {
            gauge(ui, "cpu", cpu, accent(), 100.0);
            gauge(ui, "mem", mem_p, Color32::from_rgb(190, 140, 240), 100.0);
        });
        detail_row(ui, "mem", &format!("{} / {}", human_size(s.mem_usage()), human_size(s.mem_limit())));
        detail_row(ui, "pids", &s.pids_stats.current.to_string());
        detail_row(ui, "net", &format!("↓ {}  ↑ {}", human_size(s.net_rx()), human_size(s.net_tx())));
        detail_row(ui, "disk", &format!("r {}  w {}", human_size(s.blk_read()), human_size(s.blk_write())));
        if let Some(h) = g.hist.get(&c.id) {
            if h.cpu.len() >= 2 {
                ui.label(RichText::new("cpu history").small().weak());
                let cpu: Vec<f64> = h.cpu.iter().copied().collect();
                area_chart(ui, "", &cpu, accent(), Vec2::new(ui.available_width(), 70.0));
            }
            if h.rx.len() >= 2 {
                ui.label(RichText::new("net rx/s").small().weak());
                let rx: Vec<f64> = h.rx.iter().map(|v| *v as f64).collect();
                area_chart(ui, "", &rx, Color32::from_rgb(90, 200, 130), Vec2::new(ui.available_width(), 55.0));
            }
        }
    }

    fn logs_view(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        if g.logs_container.is_none() {
            ui.label(RichText::new("Loading… pick a container, or open this tab to see all containers.").weak());
            return;
        }
        let name = g.logs_container_name.clone();
        let n = g.logs.len();
        let streaming = g.logs_active;
        let is_all = g.logs_container.as_deref() == Some("__all__");
        let mut back = false;
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("Logs: {name}")).strong());
            ui.label(RichText::new(format!("{n} lines")).weak());
            if streaming {
                ui.add(egui::Spinner::new().size(13.0).color(accent()));
                ui.label(RichText::new("streaming").weak());
            }
            if !is_all && ui.button("back to container").clicked() {
                back = true;
            }
        });
        if back {
            self.gui.tab = Tab::Containers;
        }
        ui.separator();
        self.logs_page(ui);
    }

    /// Shared log renderer with fade-in and fresh-line flash.
    fn logs_page(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        let age = self.logs_opened.elapsed().as_secs_f32();
        let page_alpha = (age / 0.45).clamp(0.0, 1.0);
        let fresh_start = self.prev_log_len.min(g.logs.len());
        let fresh_age = self.logs_opened.elapsed().as_secs_f32().min(2.0);
        let all = g.logs_container.as_deref() == Some("__all__");

        let follow = g.logs_follow;
        let lines = &g.logs;
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(follow)
            .show(ui, |ui| {
                for (idx, l) in lines.iter().enumerate() {
                    let base = match l.stream {
                        StreamKind::Stderr => Color32::from_rgb(230, 120, 120),
                        StreamKind::Stdout => Color32::from_rgb(206, 212, 220),
                    };
                    let mut a = page_alpha;
                    if idx >= fresh_start {
                        let t = (fresh_age / 0.35).clamp(0.0, 1.0);
                        a = a.max(0.35 + 0.65 * t);
                    }
                    let mut row = RichText::new(&l.text)
                        .monospace()
                        .size(13.0)
                        .color(base.linear_multiply(a));
                    if all {
                        if let Some(ref src) = l.source {
                            row = RichText::new(format!("{}  {}", src, l.text))
                                .monospace()
                                .size(13.0)
                                .color(base.linear_multiply(a));
                        }
                    }
                    ui.label(row);
                }
            });
        self.prev_log_len = g.logs.len();
    }

    fn status_bar(&mut self, ctx: &egui::Context) {
        let g = &self.gui;
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(e) = &g.error {
                    ui.label(RichText::new(format!("error: {e}")).color(Color32::from_rgb(230, 110, 110)));
                } else if !g.status.is_empty() {
                    ui.label(RichText::new(&g.status).weak());
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let info_bits = g
                        .info
                        .as_ref()
                        .map(|i| format!(" · {} cores · {}", i.ncpu, human_size(i.mem_total)))
                        .unwrap_or_default();
                    ui.label(
                        RichText::new(format!(
                            "containers: {} · images: {} · volumes: {} · updated {}{}",
                            g.containers.len(),
                            g.images.len(),
                            g.volumes.len(),
                            if g.last_refresh.is_empty() { "-" } else { &g.last_refresh },
                            info_bits
                        ))
                        .weak(),
                    );
                });
            });
        });
    }

    fn remove_dialog(&mut self, ctx: &egui::Context, id: &str) {
        let name = self
            .gui
            .containers
            .iter()
            .find(|c| c.id == id)
            .map(|c| c.short_name())
            .unwrap_or_else(|| id.to_string());
        egui::Window::new("Remove container")
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, Vec2::ZERO)
            .show(ctx, |ui| {
                ui.label(format!("Remove {name}? This deletes the container."));
                ui.horizontal(|ui| {
                    if ui
                        .button(RichText::new("Remove").color(Color32::from_rgb(220, 90, 90)))
                        .clicked()
                    {
                        if let Some(c) = self.gui.containers.iter().find(|c| c.id == id).cloned() {
                            self.gui.pending_action = Some((id.to_string(), Pending::Remove));
                            spawn_action(&self.rt, &self.docker, &self.tx, &c, Pending::Remove);
                        }
                        self.gui.confirm_remove = None;
                    }
                    if ui.button("Cancel").clicked() {
                        self.gui.confirm_remove = None;
                        self.gui.status = "remove cancelled".to_string();
                    }
                });
            });
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

fn tab_button(ui: &mut Ui, current: Tab, label: &'static str, tab: Tab) -> Option<Tab> {
    let active = tab == current;
    let btn = if active {
        egui::Button::new(RichText::new(label).strong().color(accent()))
            .fill(Color32::from_rgb(35, 50, 75))
    } else {
        egui::Button::new(RichText::new(label).weak()).fill(Color32::TRANSPARENT)
    };
    if ui.add(btn).clicked() {
        Some(tab)
    } else {
        None
    }
}

/// Toggleable filter chip; true when clicked.
fn chip(ui: &mut Ui, active: bool, label: String) -> bool {
    let btn = if active {
        egui::Button::new(RichText::new(label).small().strong().color(accent()))
            .fill(Color32::from_rgb(34, 60, 92))
    } else {
        egui::Button::new(RichText::new(label).small().weak()).fill(Color32::from_rgb(24, 29, 39))
    };
    ui.add(btn).clicked()
}

fn dot(running: bool, pulse: f32) -> RichText {
    // pulse eases 0..1; running dots glow, stopped ones sit grey.
    if running {
        RichText::new("●")
            .color(Color32::from_rgb(60 + 40 * pulse as u8, 190, 110 + 40 * pulse as u8))
            .size(15.0 + 3.0 * pulse)
    } else {
        RichText::new("○").color(Color32::from_rgb(140, 146, 156)).size(14.0)
    }
}

/// Selected rows ride a blue wash; unselected stay clear.
fn row_fill(selected: bool, pulse: f32) -> Color32 {
    if selected {
        Color32::from_rgba_premultiplied(36, 58, 88, (140.0 * (0.4 + 0.6 * pulse.max(0.5))) as u8)
    } else {
        Color32::TRANSPARENT
    }
}

/// Colored pill button; dims and ignores clicks while an action is in flight.
fn action_button(ui: &mut Ui, label: &str, color: Color32, busy: bool) -> egui::Response {
    let txt = RichText::new(label).color(if busy {
        color.gamma_multiply(0.4)
    } else {
        color
    });
    let btn = egui::Button::new(txt)
        .fill(Color32::from_rgba_premultiplied(28, 34, 46, 220))
        .stroke(Stroke::new(1.0_f32, color.gamma_multiply(0.55)))
        .min_size(Vec2::new(84.0, 30.0));
    let resp = ui.add_enabled(!busy, btn);
    if resp.hovered() {
        resp.clone().on_hover_text(label);
    }
    resp
}

fn tx(t: impl Into<WidgetText>) -> egui::Label {
    egui::Label::new(t).selectable(false)
}

fn detail_row(ui: &mut Ui, k: &str, v: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{k}:")).weak());
        ui.label(RichText::new(v).monospace());
    });
}

/// Dashboard stat card: label, big number, sub-caption.
fn stat_card(ui: &mut Ui, label: &str, value: &str, sub: &str, w: f32, color: Color32) {
    let frame = Frame::NONE
        .fill(Color32::from_rgb(24, 29, 39))
        .stroke(Stroke::new(1.0_f32, Color32::from_rgb(40, 48, 62)))
        .corner_radius(10)
        .inner_margin(egui::Margin::same(10));
    frame.show(ui, |ui| {
        ui.set_width(w);
        ui.label(RichText::new(label).small().weak());
        ui.label(RichText::new(value).heading().color(color).size(24.0));
        ui.label(RichText::new(sub).small().weak());
    });
}

/// Horizontal gauge: label, bar, percent.
fn gauge(ui: &mut Ui, label: &str, pct: f64, color: Color32, full: f64) {
    let frac = (pct / full).clamp(0.0, 1.0) as f32;
    ui.vertical(|ui| {
        ui.label(RichText::new(format!("{label} {pct:.1}%")).small().weak());
        let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width().max(60.0), 8.0), Sense::hover());
        let paint = ui.painter();
        paint.rect_filled(rect, 4.0, Color32::from_rgb(30, 36, 48));
        if frac > 0.0 {
            let r = Rect::from_min_size(rect.min, Vec2::new(rect.width() * frac, rect.height()));
            paint.rect_filled(r, 4.0, color);
        }
    });
}

/// Tiny inline sparkline.
fn sparkline(ui: &mut Ui, series: &VecDeque<f64>, color: Color32, size: Vec2) {
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    if series.len() < 2 {
        return;
    }
    let max = series.iter().cloned().fold(0.0_f64, f64::max).max(1.0);
    let n = series.len();
    let paint = ui.painter();
    let stroke = Stroke::new(1.2_f32, color);
    let mut prev: Option<Pos2> = None;
    for (i, v) in series.iter().enumerate() {
        let x = rect.left() + rect.width() * (i as f64 / (n - 1) as f64) as f32;
        let y = rect.bottom() - rect.height() * ((*v / max).clamp(0.0, 1.0)) as f32;
        let p = Pos2::new(x, y);
        if let Some(q) = prev {
            paint.line_segment([q, p], stroke);
        }
        prev = Some(p);
    }
}

/// Filled area chart with grid.
fn area_chart(ui: &mut Ui, title: &str, series: &[f64], color: Color32, size: Vec2) {
    let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
    let paint = ui.painter();
    paint.rect_filled(rect, 6.0, Color32::from_rgb(14, 18, 25));
    paint.rect_stroke(rect, 6.0, Stroke::new(1.0_f32, Color32::from_rgb(36, 44, 58)), egui::epaint::StrokeKind::Outside);
    // gridlines
    for g in 1..4 {
        let y = rect.top() + rect.height() * g as f32 / 4.0;
        paint.line_segment(
            [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
            Stroke::new(0.5_f32, Color32::from_rgb(28, 34, 46)),
        );
    }
    if series.len() < 2 {
        return;
    }
    let max = series.iter().cloned().fold(0.0_f64, f64::max).max(1.0);
    let n = series.len();
    let pts: Vec<Pos2> = series
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let x = rect.left() + rect.width() * (i as f64 / (n - 1) as f64) as f32;
            let y = rect.bottom() - rect.height() * ((*v / max).clamp(0.0, 1.0)) as f32;
            Pos2::new(x, y)
        })
        .collect();
    // filled area
    let mut mesh = egui::epaint::Mesh {
        indices: Vec::new(),
        vertices: Vec::new(),
        texture_id: egui::TextureId::Managed(0),
    };
    let base = Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), 70);
    mesh.vertices.push(egui::epaint::Vertex {
        pos: Pos2::new(rect.left(), rect.bottom()),
        uv: egui::epaint::WHITE_UV,
        color: base,
    });
    for p in &pts {
        mesh.vertices.push(egui::epaint::Vertex {
            pos: *p,
            uv: egui::epaint::WHITE_UV,
            color: base,
        });
    }
    mesh.vertices.push(egui::epaint::Vertex {
        pos: Pos2::new(rect.right(), rect.bottom()),
        uv: egui::epaint::WHITE_UV,
        color: base,
    });
    for i in 1..mesh.vertices.len() - 1 {
        mesh.indices.extend_from_slice(&[0, i as u32, (i + 1) as u32]);
    }
    paint.add(mesh);
    // top line
    let stroke = Stroke::new(1.4_f32, color);
    for w in pts.windows(2) {
        paint.line_segment([w[0], w[1]], stroke);
    }
    if !title.is_empty() {
        paint.text(
            Pos2::new(rect.left() + 6.0, rect.top() + 4.0),
            egui::Align2::LEFT_TOP,
            title,
            FontId::new(11.0, FontFamily::Monospace),
            Color32::from_rgb(150, 155, 165),
        );
    }
}

fn accent() -> Color32 {
    Color32::from_rgb(60, 150, 240)
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
    let icon = load_icon();
    let mut vb = egui::ViewportBuilder::default()
        .with_inner_size([1100.0, 700.0])
        .with_min_inner_size([800.0, 500.0])
        .with_title("Simple Docker Desktop")
        .with_app_id("io.sdd.SimpleDockerDesktop");
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
        "sdd",
        opts,
        Box::new(move |cc| {
            let app = SddApp::new(cc, Arc::clone(&rt), &socket, refresh);
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
