//! Graphical desktop front-end (egui/eframe).
//!
//! Same engine facade as the TUI: all Docker calls run on the tokio runtime
//! and report back through an mpsc channel, so the window never blocks.

use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::{
    Color32, FontData, FontFamily, FontId, Frame, RichText, ScrollArea, Sense, TextStyle, Ui,
    Vec2, WidgetText,
};

use tokio::sync::mpsc;

use crate::docker::{self, ContainerSummary, Docker, DockerError, LogLine, StreamKind};

/// One page of container logs held by the UI.
const LOG_CAP: usize = 5_000;
/// How many lines to pull when first opening a container's logs.
const LOG_TAIL: usize = 1_000;

#[derive(PartialEq, Eq, Clone, Copy)]
enum Tab {
    Containers,
    Images,
    Volumes,
    Logs,
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
    LogDone(u64, Result<(), String>),
    ActionDone(Result<(&'static str, String), String>),
}

struct RefreshData {
    containers: Result<Vec<ContainerSummary>, String>,
    images: Result<Vec<docker::ImageSummary>, String>,
    volumes: Result<Vec<docker::VolumeEntry>, String>,
}

struct Gui {
    tab: Tab,
    containers: Vec<ContainerSummary>,
    selected: Option<usize>,
    images: Vec<docker::ImageSummary>,
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
    last_refresh: String,
    daemon_ok: bool,
    docker_version: String,
}

impl Gui {
    fn new() -> Self {
        Self {
            tab: Tab::Containers,
            containers: Vec::new(),
            selected: None,
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
            last_refresh: String::new(),
            daemon_ok: false,
            docker_version: String::new(),
        }
    }

    fn filtered(&self) -> Vec<(usize, &ContainerSummary)> {
        let f = self.filter.to_lowercase();
        self.containers
            .iter()
            .enumerate()
            .filter(|(_, c)| {
                f.is_empty()
                    || c.names.join(",").to_lowercase().contains(&f)
                    || c.image.to_lowercase().contains(&f)
                    || c.state.to_lowercase().contains(&f)
            })
            .collect()
    }

    fn selected_container(&self) -> Option<&ContainerSummary> {
        let list = self.filtered();
        list.into_iter().find(|(i, _)| Some(*i) == self.selected).map(|(_, c)| c)
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
}

/// egui application wired to a tokio runtime.
struct SddApp {
    gui: Gui,
    docker: Arc<Docker>,
    tx: mpsc::UnboundedSender<Msg>,
    rx: mpsc::UnboundedReceiver<Msg>,
    rt: tokio::runtime::Handle,
    refresh: f64,
    last_refresh: Instant,
    tick: Instant,
    // --- micro animations ---
    /// per-container-id eased 0..1 for the status dot pulse
    dot_anim: std::collections::HashMap<String, f32>,
    /// container id currently running a spin animation, and its start time
    spin: Option<(String, Instant)>,
    /// when the current logs page opened (fade-in)
    logs_opened: Instant,
    /// number of log lines at last paint, to flash freshly arrived lines
    prev_log_len: usize,
}

impl SddApp {
    fn new(
        cc: &eframe::CreationContext<'_>,
        rt: tokio::runtime::Runtime,
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
            refresh,
            last_refresh: Instant::now() - Duration::from_secs(999),
            tick: Instant::now(),
            dot_anim: Default::default(),
            spin: None,
            logs_opened: Instant::now(),
            prev_log_len: 0,
        };
        // Keep the runtime alive for the whole lifetime of the window.
        std::mem::forget(rt);
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
            let name = format!("sys-{}", std::path::Path::new(path).file_stem().unwrap().to_string_lossy());
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
            self.apply(m);
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
    /// Advance all animation clocks; true while something is still moving.
    fn advance_animations(&mut self) -> bool {
        let now = Instant::now();
        let mut animating = false;

        // Status dots: ease each container's pulse toward 1.0 (running) or
        // fade the glow out (stopped). Values live in 0..=1.
        for c in &self.gui.containers {
            let running = c.state == "running";
            let e = self.dot_anim.entry(c.id.clone()).or_insert(0.0);
            let target = if running { 1.0 } else { 0.0 };
            let d = target - *e;
            if d.abs() > 0.01 {
                *e += d * 0.25;
                animating = true;
            } else {
                *e = target;
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
    fn apply(&mut self, m: Msg) {
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
            Msg::LogDone(gen, r) => {
                if gen == g.logs_gen {
                    g.logs_active = false;
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
        }
    }

    fn top_bar(&mut self, ctx: &egui::Context) {
        let g = &mut self.gui;
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
                    let mut refresh_now = false;
                    if ui.button("⟳").on_hover_text("refresh now").clicked() {
                        refresh_now = true;
                    }
                    ui.separator();
                    // right-to-left: declare in reverse so they render L→R
                    let mut new_tab = None;
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
                    if let Some(t) = new_tab {
                        g.tab = t;
                    }
                    if refresh_now {
                        spawn_refresh(&self.rt, &self.docker, &self.tx);
                        g.status = "refreshing...".to_string();
                    }
                });
            });
        });
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
            ui.separator();
            match tab {
                Tab::Containers => self.container_list(ui),
                Tab::Images => self.image_list(ui),
                Tab::Volumes => self.volume_list(ui),
                Tab::Logs => self.logs_summary(ui),
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
        let mut new_sel: Option<usize> = None;
        let mut open_logs: Option<String> = None;

        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
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
                    if click.double_clicked() {
                        open_logs = Some(c.id.clone());
                    }
                });
            }
        });

        if let Some(i) = new_sel {
            self.gui.selected = Some(i);
        }
        if let Some(id) = open_logs {
            self.open_logs(&id);
        }
        // Action buttons for the selected container.
        let sel_id = self
            .gui
            .selected
            .and_then(|i| self.gui.containers.get(i))
            .map(|c| (c.id.clone(), c.state.clone()));
        if let Some((id, state)) = sel_id {
            if self.gui.tab == Tab::Containers {
                ui.add_space(8.0);
                ui.separator();
                let mut action: Option<(String, Pending)> = None;
                let mut confirm = false;
                ui.horizontal_wrapped(|ui| {
                    let running = state == "running";
                    let busy = self.gui.pending_action.as_ref().is_some_and(|(i, _)| *i == id);
                    let start_stop = if running {
                        action_button(ui, "■ Stop", Color32::from_rgb(230, 140, 80), busy)
                    } else {
                        action_button(ui, "▶ Start", Color32::from_rgb(90, 200, 130), busy)
                    };
                    if start_stop.clicked() {
                        action = Some((id.clone(), if running { Pending::Stop } else { Pending::Start }));
                    }
                    if action_button(ui, "↻ Restart", accent(), busy).clicked() {
                        action = Some((id.clone(), Pending::Restart));
                    }
                    if action_button(ui, "✕ Remove", Color32::from_rgb(220, 90, 90), busy).clicked() {
                        confirm = true;
                    }
                });
                if confirm {
                    self.gui.confirm_remove = Some(id);
                }
                if let Some((id, p)) = action {
                    if let Some(c) = self.gui.containers.iter().find(|c| c.id == id).cloned() {
                        self.gui.pending_action = Some((id.clone(), p));
                        self.spin = Some((id.clone(), Instant::now()));
                        spawn_action(&self.rt, &self.docker, &self.tx, &c, p);
                    }
                }
            }
        }
    }

    fn image_list(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        let f = g.filter.to_lowercase();
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            for im in &g.images {
                let tag = im.tag();
                if !f.is_empty() && !tag.to_lowercase().contains(&f) {
                    continue;
                }
                ui.horizontal(|ui| {
                    ui.label(RichText::new(tag).strong());
                });
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{}  {}", im.short_id(), human_size(im.size))).weak());
                });
                ui.separator();
            }
        });
    }

    fn volume_list(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        let f = g.filter.to_lowercase();
        ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            for v in &g.volumes {
                if !f.is_empty() && !v.name.to_lowercase().contains(&f) {
                    continue;
                }
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&v.name).strong());
                });
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("driver={}", v.driver)).weak());
                });
                ui.separator();
            }
        });
    }

    fn logs_summary(&mut self, ui: &mut Ui) {
        let g = &mut self.gui;
        if g.logs_container_name.is_empty() {
            ui.label(RichText::new("Select a container and press the logs button, or double-click it.").weak());
            return;
        }
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
        if ui.button(if g.logs_follow { "Pause follow" } else { "Resume follow" }).clicked() {
            g.logs_follow = !g.logs_follow;
        }
        let reopen = g.logs_container.is_some() && ui.button("Reopen stream").clicked();
        if reopen {
            let id = g.logs_container.clone().unwrap();
            self.open_logs(&id);
        }
    }

    fn center(&mut self, ctx: &egui::Context) {
        let tab = self.gui.tab;
        egui::CentralPanel::default().show(ctx, |ui| match tab {
            Tab::Logs => self.logs_view(ui),
            Tab::Containers => self.container_detail(ui),
            Tab::Images => {
                ui.label(RichText::new("Select an image in the list to see details.").weak());
            }
            Tab::Volumes => {
                ui.label(RichText::new("Select a volume in the list to see details.").weak());
            }
        });
    }

    fn container_detail(&mut self, ui: &mut Ui) {
        let g = &mut self.gui;
        let Some(c) = g.selected_container().cloned() else {
            ui.label(RichText::new("No container selected.").weak());
            return;
        };
        ui.label(RichText::new(c.short_name()).heading());
        ui.add_space(4.0);
        detail_row(ui, "image", &c.image);
        detail_row(ui, "state", &c.status);
        detail_row(ui, "id", &c.id[..12.min(c.id.len())]);
        if !c.port_summary().is_empty() {
            detail_row(ui, "ports", &c.port_summary());
        }
        ui.add_space(10.0);
        if ui.button("View logs").clicked() {
            let id = c.id.clone();
            self.open_logs(&id);
        }
    }

    fn logs_view(&mut self, ui: &mut Ui) {
        let g = &self.gui;
        if g.logs.is_empty() && g.logs_container.is_none() {
            ui.label(RichText::new("Pick a container, then View logs / double-click.").weak());
            return;
        }
        let name = g.logs_container_name.clone();
        let n = g.logs.len();
        ui.horizontal(|ui| {
            ui.label(RichText::new(format!("Logs: {name}")).strong());
            ui.label(RichText::new(format!("{n} lines")).weak());
            if g.logs_active {
                ui.add(egui::Spinner::new().size(13.0).color(accent()));
                ui.label(RichText::new("streaming").weak());
            }
        });
        ui.separator();

        // Page fade-in: the whole block eases in over ~450 ms after opening.
        let age = self.logs_opened.elapsed().as_secs_f32();
        let page_alpha = (age / 0.45).clamp(0.0, 1.0);
        // Freshly arrived lines (since last paint) flash brighter for a beat.
        let fresh_start = self.prev_log_len.min(g.logs.len());
        let fresh_age = self.logs_opened.elapsed().as_secs_f32().min(2.0);

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
                    ui.label(
                        RichText::new(&l.text)
                            .monospace()
                            .size(13.0)
                            .color(base.linear_multiply(a)),
                    );
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
                    ui.label(
                        RichText::new(format!(
                            "containers: {} · images: {} · volumes: {} · updated {}",
                            g.containers.len(),
                            g.images.len(),
                            g.volumes.len(),
                            if g.last_refresh.is_empty() { "-" } else { &g.last_refresh }
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
                    if ui.button(RichText::new("Remove").color(Color32::from_rgb(220, 90, 90))).clicked() {
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
        g.logs_gen += 1;
        g.pending_logs_gen = Some(g.logs_gen);
        g.logs_container = Some(id.to_string());
        g.logs_container_name = c.short_name();
        g.logs.clear();
        g.logs_follow = true;
        g.logs_active = false;
        g.tab = Tab::Logs;
        self.logs_opened = Instant::now();
        self.prev_log_len = 0;
        spawn_logs(&self.rt, &self.docker, &self.tx, id, g.logs_gen);
    }
}

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

fn dot(running: bool, pulse: f32) -> RichText {
    // pulse eases 0..1; running dots glow, stopped ones sit grey.
    let glow = (0.55 + 0.45 * pulse) as u8;
    if running {
        RichText::new("●")
            .color(Color32::from_rgb(60 + 40 * pulse as u8, 190, 110 + 40 * pulse as u8))
            .size(15.0 + 3.0 * pulse)
    } else {
        let _ = glow;
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
        .stroke(egui::Stroke::new(1.0_f32, color.gamma_multiply(0.55)))
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
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (h, m, s) = ((secs / 3600) % 24, (secs / 60) % 60, secs % 60);
    format!("{h:02}:{m:02}:{s:02}")
}

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

fn spawn_logs(
    rt: &tokio::runtime::Handle,
    docker: &Arc<Docker>,
    tx: &mpsc::UnboundedSender<Msg>,
    id: &str,
    gen: u64,
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
                    let _ = tx.send(Msg::LogDone(gen, Err(err_str(DockerError::Io(e.to_string())))));
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
            let _ = tx.send(Msg::LogDone(gen, Err("unexpected response from daemon".into())));
            return;
        }
        if let Some(code) = head.get(9..12).and_then(|c| std::str::from_utf8(c).ok()) {
            if !code.starts_with('2') {
                let _ = tx.send(Msg::LogDone(gen, Err(format!("daemon returned {code}"))));
                return;
            }
        }
        let chunked = head.windows(7).any(|w| w.eq_ignore_ascii_case(b"chunked"));

        let mut fr = Chunked::default();
        let mut chunk = [0u8; 16 * 1024];
        if !body_prefix.is_empty() {
            let payload = if chunked { fr.feed(&body_prefix) } else { body_prefix.clone() };
            let lines = dec.push(&payload);
            if !lines.is_empty() && tx.send(Msg::LogLines(gen, lines)).is_err() {
                return;
            }
        }
        while !eof {
            match stream.read(&mut chunk).await {
                Ok(0) => break,
                Ok(n) => {
                    let payload = if chunked { fr.feed(&chunk[..n]) } else { chunk[..n].to_vec() };
                    let lines = dec.push(&payload);
                    if !lines.is_empty() && tx.send(Msg::LogLines(gen, lines)).is_err() {
                        return;
                    }
                    if chunked && fr.done {
                        break;
                    }
                }
                Err(e) => {
                    let _ = tx.send(Msg::LogDone(gen, Err(err_str(DockerError::Io(e.to_string())))));
                    return;
                }
            }
        }
        let _ = tx.send(Msg::LogDone(gen, Ok(())));
    });
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
                        Ok(0) | Err(_) => {
                            self.done = true;
                            break;
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

pub fn run(
    rt: tokio::runtime::Runtime,
    socket: &str,
    refresh: f64,
) -> Result<(), Box<dyn std::error::Error>> {
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
    eframe::run_native(
        "sdd",
        opts,
        Box::new(move |cc| {
            let app = SddApp::new(cc, rt, &socket, refresh);
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
