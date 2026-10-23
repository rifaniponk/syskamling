use std::io;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Row, Table},
    Frame, Terminal,
};
use tokio::sync::mpsc;

use crate::agents::{AgentScanner, AgentSession};
use crate::docker::{ContainerSummary, Docker, DockerError, LogLine, StreamKind};
use crate::hw::{HwSample, Sampler};
use crate::hw_inventory::{self as hw_inventory, GpuStat};
use crate::ports::{self as portsmod, PortEntry};
use crate::strata::{self, Metrics};

pub fn run(
    rt: tokio::runtime::Runtime,
    socket: &str,
    refresh: f64,
    logs_first: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let res = rt.block_on(run_async(&mut terminal, socket, refresh, logs_first));

    disable_raw_mode()?;
    execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;
    terminal.show_cursor()?;
    res
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum Tab {
    Containers,
    Images,
    Volumes,
    Agents,
    Hardware,
    Ports,
    Processes,
    Strata,
    Logs,
}

impl Tab {
    fn next(self) -> Self {
        match self {
            Tab::Containers => Tab::Images,
            Tab::Images => Tab::Volumes,
            Tab::Volumes => Tab::Agents,
            Tab::Agents => Tab::Hardware,
            Tab::Hardware => Tab::Processes,
            Tab::Processes => Tab::Ports,
            Tab::Ports => Tab::Strata,
            Tab::Strata => Tab::Logs,
            Tab::Logs => Tab::Containers,
        }
    }
    fn prev(self) -> Self {
        match self {
            Tab::Containers => Tab::Logs,
            Tab::Images => Tab::Containers,
            Tab::Volumes => Tab::Images,
            Tab::Agents => Tab::Volumes,
            Tab::Hardware => Tab::Agents,
            Tab::Processes => Tab::Hardware,
            Tab::Ports => Tab::Processes,
            Tab::Strata => Tab::Ports,
            Tab::Logs => Tab::Strata,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Tab::Containers => " Containers ",
            Tab::Images => " Images ",
            Tab::Volumes => " Volumes ",
            Tab::Agents => " Agents ",
            Tab::Hardware => " Hardware ",
            Tab::Ports => " Ports ",
            Tab::Processes => " Processes ",
            Tab::Strata => " Strata ",
            Tab::Logs => " Logs ",
        }
    }
}

/// One page of container logs held by the UI.
const LOG_CAP: usize = 5_000;
/// How many lines to pull when first opening a container's logs.
const LOG_TAIL: usize = 1_000;

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

/// Messages the background tokio tasks send back to the UI loop.
enum Msg {
    Refreshed(Result<RefreshData, String>),
    LogStart(u64, Result<(), String>),
    LogLines(u64, Vec<LogLine>),
    LogDone(u64, Result<(), String>),
    ActionDone(Result<(&'static str, String), String>),
    AgentsScanned(Vec<AgentSession>),
    PortsScanned(Vec<PortEntry>),
    ProcsScanned(Vec<crate::procs::ProcInfo>),
    StrataHealth(Result<strata::Health, String>),
    StrataMetrics(Result<Box<Metrics>, String>),
}

struct RefreshData {
    containers: Result<Vec<ContainerSummary>, String>,
    images: Result<Vec<crate::docker::ImageSummary>, String>,
    volumes: Result<Vec<crate::docker::VolumeEntry>, String>,
}

struct App {
    tab: Tab,
    containers: Vec<ContainerSummary>,
    container_state: ListState,
    images: Vec<crate::docker::ImageSummary>,
    images_state: ListState,
    volumes: Vec<crate::docker::VolumeEntry>,
    volumes_state: ListState,
    agents: Vec<AgentSession>,
    agents_state: ListState,
    hw: Option<HwSample>,
    /// rolling CPU history for the Hardware trend line
    hw_cpu_hist: Vec<f64>,
    /// live per-GPU stats, refreshed on the Hardware tab
    gpus: Vec<GpuStat>,
    ports: Vec<PortEntry>,
    ports_state: ListState,
    procs: Vec<crate::procs::ProcInfo>,
    procs_state: ListState,
    proc_sort: crate::procs::SortCol,
    proc_sort_desc: bool,
    /// icon name of the app group expanded in the processes table
    proc_group: Option<String>,
    // strata screen: settings live in the shared config file; the TUI is
    // read-only and points at it when the server is not configured
    strata_cfg: strata::StrataConfig,
    strata_health: Option<strata::Health>,
    strata_metrics: Option<Box<Metrics>>,
    strata_error: Option<String>,
    strata_show_all: bool,
    logs: Vec<LogLine>,
    logs_follow: bool,
    logs_scroll: usize,
    logs_container: Option<String>,
    logs_container_name: String,
    logs_active: bool,
    logs_gen: u64,
    pending_logs_gen: Option<u64>,
    confirm_remove: Option<String>,
    pending_action: Option<(String, Pending)>,
    status: String,
    error: Option<String>,
    last_refresh: String,
    filter: String,
    filtering: bool,
    quit: bool,
}

impl App {
    fn new() -> Self {
        let mut container_state = ListState::default();
        container_state.select(Some(0));
        Self {
            tab: Tab::Containers,
            containers: Vec::new(),
            container_state,
            images: Vec::new(),
            images_state: ListState::default(),
            volumes: Vec::new(),
            volumes_state: ListState::default(),
            agents: Vec::new(),
            agents_state: ListState::default(),
            hw: None,
            hw_cpu_hist: Vec::new(),
            gpus: Vec::new(),
            ports: Vec::new(),
            ports_state: ListState::default(),
            procs: Vec::new(),
            procs_state: ListState::default(),
            proc_sort: crate::procs::SortCol::Mem,
            proc_sort_desc: true,
            proc_group: None,
            strata_cfg: strata::StrataConfig::default(),
            strata_health: None,
            strata_metrics: None,
            strata_error: None,
            strata_show_all: false,
            logs: Vec::new(),
            logs_follow: true,
            logs_scroll: 0,
            logs_container: None,
            logs_container_name: String::new(),
            logs_active: false,
            logs_gen: 0,
            pending_logs_gen: None,
            confirm_remove: None,
            pending_action: None,
            status: String::new(),
            error: None,
            last_refresh: String::new(),
            filter: String::new(),
            filtering: false,
            quit: false,
        }
    }

    fn filtered_containers(&self) -> Vec<&ContainerSummary> {
        if self.filter.is_empty() {
            return self.containers.iter().collect();
        }
        let f = self.filter.to_lowercase();
        self.containers
            .iter()
            .filter(|c| {
                c.names.join(",").to_lowercase().contains(&f)
                    || c.image.to_lowercase().contains(&f)
                    || c.state.to_lowercase().contains(&f)
            })
            .collect()
    }

    fn selected_container(&self) -> Option<&ContainerSummary> {
        let list = self.filtered_containers();
        list.into_iter().nth(self.container_state.selected()?)
    }

    fn move_container(&mut self, delta: i32) {
        let len = self.filtered_containers().len();
        if len == 0 {
            self.container_state.select(Some(0));
            return;
        }
        let cur = self.container_state.selected().unwrap_or(0) as i32;
        self.container_state
            .select(Some((cur + delta).clamp(0, len as i32 - 1) as usize));
    }

    fn move_images(&mut self, delta: i32) {
        let len = self.images.len();
        if len == 0 {
            return;
        }
        let cur = self.images_state.selected().unwrap_or(0) as i32;
        self.images_state
            .select(Some((cur + delta).clamp(0, len as i32 - 1) as usize));
    }

    fn move_volumes(&mut self, delta: i32) {
        let len = self.volumes.len();
        if len == 0 {
            return;
        }
        let cur = self.volumes_state.selected().unwrap_or(0) as i32;
        self.volumes_state
            .select(Some((cur + delta).clamp(0, len as i32 - 1) as usize));
    }

    fn filtered_agents(&self) -> Vec<&AgentSession> {
        if self.filter.is_empty() {
            return self.agents.iter().collect();
        }
        let f = self.filter.to_lowercase();
        self.agents
            .iter()
            .filter(|a| {
                a.agent.to_lowercase().contains(&f)
                    || a.vendor.to_lowercase().contains(&f)
                    || a.cwd.as_deref().map(|c| c.to_lowercase()).unwrap_or_default().contains(&f)
            })
            .collect()
    }

    fn move_agents(&mut self, delta: i32) {
        let len = self.filtered_agents().len();
        if len == 0 {
            self.agents_state.select(None);
            return;
        }
        let cur = self.agents_state.selected().unwrap_or(0) as i32;
        self.agents_state
            .select(Some((cur + delta).clamp(0, len as i32 - 1) as usize));
    }

    fn filtered_ports(&self) -> Vec<&PortEntry> {
        if self.filter.is_empty() {
            return self.ports.iter().collect();
        }
        let f = self.filter.to_lowercase();
        self.ports
            .iter()
            .filter(|p| {
                p.proto.contains(&f)
                    || p.addr.contains(&f)
                    || p.port.to_string().contains(&f)
                    || p.proc_name.as_deref().map(|n| n.to_lowercase()).unwrap_or_default().contains(&f)
                    || p.state.to_lowercase().contains(&f)
            })
            .collect()
    }

    fn sorted_procs(&self) -> Vec<crate::procs::ProcInfo> {
        let f = self.filter.to_lowercase();
        let mut rows: Vec<crate::procs::ProcInfo> = self
            .procs
            .iter()
            .filter(|r| {
                f.is_empty()
                    || r.name.to_lowercase().contains(&f)
                    || r.cmdline.to_lowercase().contains(&f)
                    || r.pid.to_string() == f
            })
            .cloned()
            .collect();
        crate::procs::sort_rows(&mut rows, self.proc_sort, self.proc_sort_desc);
        rows
    }

    fn sort_procs(&mut self, col: crate::procs::SortCol) {
        if self.proc_sort == col {
            self.proc_sort_desc = !self.proc_sort_desc;
        } else {
            self.proc_sort = col;
            self.proc_sort_desc = !matches!(col, crate::procs::SortCol::Name | crate::procs::SortCol::Pid);
        }
    }

    /// The flat row list render_procs draws: app groups, with an expanded
    /// group's members spliced in right behind it.
    fn proc_row_count(&self) -> usize {
        let rows = self.sorted_procs();
        let groups = crate::procs::group_rows(&rows, self.proc_sort, self.proc_sort_desc);
        groups
            .iter()
            .map(|g| {
                1 + if self.proc_group.as_deref() == Some(g.icon.as_str()) {
                    g.members.len()
                } else {
                    0
                }
            })
            .sum()
    }

    fn move_procs(&mut self, delta: i32) {
        let len = self.proc_row_count();
        if len == 0 {
            self.procs_state.select(None);
            return;
        }
        let cur = self.procs_state.selected().unwrap_or(0) as i32;
        self.procs_state
            .select(Some(((cur + delta).clamp(0, len as i32 - 1)) as usize));
    }

    /// Enter on the Processes tab: toggle the app group under the cursor.
    fn toggle_proc_group(&mut self) {
        let sel = match self.procs_state.selected() {
            Some(s) => s,
            None => return,
        };
        let rows = self.sorted_procs();
        let groups = crate::procs::group_rows(&rows, self.proc_sort, self.proc_sort_desc);
        let mut i = 0usize;
        for g in &groups {
            if i == sel {
                // every app row (icon resolved) can expand, even with one
                // process; standalone rows have nothing to expand
                self.proc_group = if self.proc_group.as_deref() == Some(g.icon.as_str()) {
                    None
                } else if !g.icon.is_empty() {
                    Some(g.icon.clone())
                } else {
                    None
                };
                return;
            }
            i += 1;
            if self.proc_group.as_deref() == Some(g.icon.as_str()) {
                i += g.members.len();
            }
        }
    }

    fn move_ports(&mut self, delta: i32) {
        let len = self.filtered_ports().len();
        if len == 0 {
            self.ports_state.select(None);
            return;
        }
        let cur = self.ports_state.selected().unwrap_or(0) as i32;
        self.ports_state
            .select(Some((cur + delta).clamp(0, len as i32 - 1) as usize));
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

async fn run_async(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    socket: &str,
    refresh: f64,
    logs_first: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let rt = tokio::runtime::Handle::current();

    let docker = Arc::new(Docker::new(socket));
    let (tx, mut rx) = mpsc::unbounded_channel::<Msg>();

    let mut app = App::new();
    if logs_first {
        app.tab = Tab::Logs;
    }

    let mut last_refresh = Instant::now() - Duration::from_secs(999);
    let mut last_agent_scan = Instant::now() - Duration::from_secs(999);
    let mut last_port_scan = Instant::now() - Duration::from_secs(999);
    let mut last_proc_scan = Instant::now() - Duration::from_secs(999);
    let mut last_hw_sample = Instant::now() - Duration::from_secs(999);
    let strata_cfg = strata::load();
    app.strata_cfg = strata_cfg.clone();
    let strata_client = Arc::new(strata::HttpClient::new(&strata_cfg));
    let mut last_strata_poll = Instant::now() - Duration::from_secs(999);
    let strata_busy = Arc::new(std::sync::Mutex::new(false));
    let mut sampler = Sampler::default();
    let mut tick = tokio::time::interval(Duration::from_millis(300));
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

    // Dedicated thread for blocking key reads; the async loop never blocks.
    let (key_tx, mut key_rx) = mpsc::unbounded_channel::<KeyEv>();
    std::thread::spawn(move || loop {
        match read_key() {
            Some(ev) => {
                if key_tx.send(KeyEv::Key(ev)).is_err() {
                    break;
                }
            }
            None => {
                // stdin closed / read error: end the UI loop
                let _ = key_tx.send(KeyEv::Eof);
                break;
            }
        }
    });

    // Single-threaded loop: the UI never blocks, every Docker call runs on
    // the runtime and reports back through the channel.
    let result = loop {
        terminal.draw(|f| ui(f, &mut app))?;

        // Drain everything already queued.
        while let Ok(m) = rx.try_recv() {
            apply_msg(&mut app, m);
        }

        if app.quit {
            break Ok(());
        }

        // Periodic container/image/volume refresh.
        if refresh > 0.0 && last_refresh.elapsed() >= Duration::from_secs_f64(refresh) {
            last_refresh = Instant::now();
            spawn_refresh(&rt, &docker, &tx);
        }

        // /proc agent scan on the same cadence, on the blocking pool so the
        // UI loop never waits on filesystem reads.
        if refresh > 0.0 && last_agent_scan.elapsed() >= Duration::from_secs_f64(refresh.max(1.0)) {
            last_agent_scan = Instant::now();
            let tx2 = tx.clone();
            rt.spawn_blocking(move || {
                let mut scanner = AgentScanner::default();
                let sessions = scanner.scan();
                let _ = tx2.send(Msg::AgentsScanned(sessions));
            });
        }

        // Listening-socket scan: /proc/net + fd walk, on the blocking pool.
        if refresh > 0.0 && last_port_scan.elapsed() >= Duration::from_secs_f64(refresh.max(1.0) * 2.0) {
            last_port_scan = Instant::now();
            let tx3 = tx.clone();
            rt.spawn_blocking(move || {
                let _ = tx3.send(Msg::PortsScanned(portsmod::scan()));
            });
        }

        // Process scan: /proc walk per refresh, on every screen (like the
        // agent and port scans) so the table is warm the moment tab 7 opens;
        // the scanner keeps tick counters so CPU% is a real delta.
        if refresh > 0.0 && last_proc_scan.elapsed() >= Duration::from_secs_f64(refresh.max(1.0)) {
            last_proc_scan = Instant::now();
            let tx4 = tx.clone();
            rt.spawn_blocking(move || {
                let mut sc = crate::procs::ProcScanner::new();
                sc.scan(); // seed counters
                std::thread::sleep(std::time::Duration::from_millis(600));
                let _ = tx4.send(Msg::ProcsScanned(sc.scan()));
            });
        }

        // Host vitals: rates need consecutive samples, so the sampler lives
        // here and is read inline (/proc reads are microseconds). GPU stats
        // only run on the Hardware screen: nvidia-smi costs ~50 ms.
        if refresh > 0.0 && last_hw_sample.elapsed() >= Duration::from_secs_f64(refresh.max(1.0)) {
            last_hw_sample = Instant::now();
            let s = sampler.sample();
            app.hw_cpu_hist.push(s.cpu_pct);
            if app.hw_cpu_hist.len() > 240 {
                let excess = app.hw_cpu_hist.len() - 240;
                app.hw_cpu_hist.drain(..excess);
            }
            if app.tab == Tab::Hardware {
                app.gpus = hw_inventory::gpu_stats();
            } else if app.gpus.is_empty() {
                app.gpus = hw_inventory::gpu_stats_hwmon_only();
            }
            app.hw = Some(s);
        }

        // Strata monitor: /health + /metrics once per refresh, only while
        // the Strata tab is on screen; the async task never blocks the loop.
        if refresh > 0.0
            && app.tab == Tab::Strata
            && last_strata_poll.elapsed() >= Duration::from_secs_f64(refresh.max(1.0))
            && !*strata_busy.lock().unwrap()
        {
            last_strata_poll = Instant::now();
            *strata_busy.lock().unwrap() = true;
            let busy = Arc::clone(&strata_busy);
            let client = Arc::clone(&strata_client);
            let tx4 = tx.clone();
            rt.spawn(async move {
                let h = client.health().await;
                let _ = tx4.send(Msg::StrataHealth(h));
                let m = client.metrics().await;
                *busy.lock().unwrap() = false;
                let _ = tx4.send(Msg::StrataMetrics(m.map(Box::new)));
            });
        }

        // Wait for either a key or a tick/message.
        match tokio::select! {
            _ = tick.tick() => LoopEv::Tick,
            m = rx.recv() => LoopEv::Msg(m),
            k = key_rx.recv() => LoopEv::Key(k),
        } {
            LoopEv::Tick => {}
            LoopEv::Msg(Some(m)) => apply_msg(&mut app, m),
            LoopEv::Msg(None) => app.quit = true,
            LoopEv::Key(Some(KeyEv::Key((code, modifiers)))) => {
                handle_key(&mut app, code, modifiers, &rt, &docker, &tx)
            }
            LoopEv::Key(Some(KeyEv::Eof)) | LoopEv::Key(None) => app.quit = true,
        }
    };
    result
}

enum KeyEv {
    Key((KeyCode, crossterm::event::KeyModifiers)),
    Eof,
}

enum LoopEv {
    Tick,
    Msg(Option<Msg>),
    Key(Option<KeyEv>),
}

fn read_key() -> Option<(KeyCode, crossterm::event::KeyModifiers)> {
    if let Ok(Event::Key(k)) = event::read() {
        if k.kind == KeyEventKind::Press {
            return Some((k.code, k.modifiers));
        }
    }
    None
}

fn apply_msg(app: &mut App, m: Msg) {
    match m {
        Msg::Refreshed(Ok(data)) => {
            app.error = None;
            if app.status.contains("refreshing") {
                app.status.clear();
            }
            if let Ok(c) = data.containers {
                app.containers = c;
                let len = app.filtered_containers().len();
                if let Some(sel) = app.container_state.selected() {
                    if sel >= len.max(1) {
                        app.container_state.select(Some(len.saturating_sub(1)));
                    }
                }
            }
            if let Ok(i) = data.images {
                app.images = i;
            }
            if let Ok(v) = data.volumes {
                app.volumes = v;
            }
            app.last_refresh = now_string();
        }
        Msg::Refreshed(Err(e)) => app.error = Some(format!("refresh: {e}")),
        Msg::LogStart(gen, Ok(())) => {
            if Some(gen) == app.pending_logs_gen {
                app.pending_logs_gen = None;
                app.logs_active = true;
                app.status = format!("streaming logs from {}", app.logs_container_name);
            }
        }
        Msg::LogStart(gen, Err(e)) => {
            if Some(gen) == app.pending_logs_gen {
                app.pending_logs_gen = None;
                app.error = Some(format!("logs: {e}"));
            }
        }
        Msg::LogLines(gen, lines) => {
            if gen == app.logs_gen {
                app.push_logs(lines);
            }
        }
        Msg::LogDone(gen, r) => {
            if gen == app.logs_gen {
                app.logs_active = false;
                if let Err(e) = r {
                    app.error = Some(format!("logs: {e}"));
                }
            }
        }
        Msg::ActionDone(Ok((verb, name))) => {
            app.status = format!("{verb} {name}");
            app.pending_action = None;
        }
        Msg::ActionDone(Err(e)) => {
            app.pending_action = None;
            app.error = Some(e);
        }
        Msg::AgentsScanned(sessions) => {
            app.agents = sessions;
            let len = app.filtered_agents().len();
            if let Some(sel) = app.agents_state.selected() {
                if sel >= len {
                    app.agents_state.select(if len == 0 { None } else { Some(len - 1) });
                }
            } else if len > 0 && app.agents_state.selected().is_none() {
                app.agents_state.select(Some(0));
            }
        }
        Msg::PortsScanned(entries) => {
            app.ports = entries;
            let len = app.filtered_ports().len();
            if let Some(sel) = app.ports_state.selected() {
                if sel >= len {
                    app.ports_state.select(if len == 0 { None } else { Some(len - 1) });
                }
            } else if len > 0 && app.ports_state.selected().is_none() {
                app.ports_state.select(Some(0));
            }
        }
        Msg::ProcsScanned(rows) => {
            app.procs = rows;
            if app.proc_group.as_deref().is_some_and(|g| {
                !app.procs.iter().any(|r| r.app_icon == g)
            }) {
                app.proc_group = None;
            }
            let len = app.proc_row_count();
            if let Some(sel) = app.procs_state.selected() {
                if sel >= len {
                    app.procs_state.select(if len == 0 { None } else { Some(len - 1) });
                }
            } else if len > 0 && app.procs_state.selected().is_none() {
                app.procs_state.select(Some(0));
            }
        }
        Msg::StrataHealth(res) => match res {
            Ok(h) => {
                app.strata_health = Some(h);
                app.strata_error = None;
            }
            Err(e) => {
                app.strata_health = None;
                app.strata_error = Some(e);
            }
        },
        Msg::StrataMetrics(res) => match res {
            Ok(m) => app.strata_metrics = Some(m),
            Err(e) => app.strata_error = Some(e),
        },
    }
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
        let mut dec = crate::docker::LogStreamDecoder::new();

        // 1. Read the HTTP response head (stop at the first double CRLF).
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
                    let _ = tx.send(Msg::LogDone(gen, Err(err_str(DockerError::Io(
                        e.to_string(),
                    )))));
                    return;
                }
            };
            head.extend_from_slice(&b[..n]);
            if let Some(pos) = find(&head, b"\r\n\r\n") {
                // Bytes already read past the head belong to the body.
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

        // 2. Stream the body: de-chunk if needed, then decode the Docker
        //    multiplexed framing into lines.
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
                    let _ = tx.send(Msg::LogDone(gen, Err(err_str(DockerError::Io(
                        e.to_string(),
                    )))));
                    return;
                }
            }
        }
        let _ = tx.send(Msg::LogDone(gen, Ok(())));
    });
}

/// Minimal HTTP/1.1 chunked-transfer decoder: feeds raw socket bytes in,
/// returns the de-chunked payload.
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
                    // Need the chunk payload plus its trailing CRLF before we
                    // can consume the CRLF; wait until both are present.
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

fn handle_key(
    app: &mut App,
    code: KeyCode,
    modifiers: crossterm::event::KeyModifiers,
    rt: &tokio::runtime::Handle,
    docker: &Arc<Docker>,
    tx: &mpsc::UnboundedSender<Msg>,
) {
    use crossterm::event::KeyModifiers;

    // Confirmation prompt takes priority over everything.
    if let Some(id) = app.confirm_remove.clone() {
        match code {
            KeyCode::Char('y') | KeyCode::Char('Y') => {
                if let Some(c) = app.containers.iter().find(|c| c.id == id).cloned() {
                    app.pending_action = Some((id.clone(), Pending::Remove));
                    spawn_action(rt, docker, tx, &c, Pending::Remove);
                }
                app.confirm_remove = None;
            }
            KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                app.confirm_remove = None;
                app.status = "remove cancelled".to_string();
            }
            _ => {}
        }
        return;
    }

    if app.filtering {
        match code {
            KeyCode::Esc => {
                app.filtering = false;
                app.filter.clear();
            }
            KeyCode::Enter => app.filtering = false,
            KeyCode::Char(c) if !c.is_control() && !modifiers.contains(KeyModifiers::CONTROL) => {
                app.filter.push(c);
            }
            KeyCode::Backspace => {
                app.filter.pop();
            }
            _ => {}
        }
        return;
    }

    let rt = rt;
    match code {
        KeyCode::Char('q') | KeyCode::Esc => app.quit = true,
        KeyCode::Tab | KeyCode::Right => app.tab = app.tab.next(),
        KeyCode::BackTab | KeyCode::Left => app.tab = app.tab.prev(),
        KeyCode::Char('1') => app.tab = Tab::Containers,
        KeyCode::Char('2') => app.tab = Tab::Images,
        KeyCode::Char('3') => app.tab = Tab::Volumes,
        KeyCode::Char('4') => app.tab = Tab::Agents,
        KeyCode::Char('5') => app.tab = Tab::Hardware,
        KeyCode::Char('6') => app.tab = Tab::Ports,
        KeyCode::Char('7') => app.tab = Tab::Processes,
        KeyCode::Char('8') => app.tab = Tab::Strata,
        KeyCode::Char('9') => app.tab = Tab::Logs,
        KeyCode::Char('a') if app.tab == Tab::Strata => app.strata_show_all = !app.strata_show_all,
        KeyCode::Char('c') if app.tab == Tab::Processes => app.sort_procs(crate::procs::SortCol::Cpu),
        KeyCode::Char('m') if app.tab == Tab::Processes => app.sort_procs(crate::procs::SortCol::Mem),
        KeyCode::Char('n') if app.tab == Tab::Processes => app.sort_procs(crate::procs::SortCol::Name),
        KeyCode::Char('p') if app.tab == Tab::Processes => app.sort_procs(crate::procs::SortCol::Pid),
        KeyCode::Char('/') => {
            app.filtering = true;
            app.filter.clear();
        }
        KeyCode::Char('R') => {
            spawn_refresh(rt, docker, tx);
            app.status = "refreshing...".to_string();
        }
        KeyCode::Down | KeyCode::Char('j') => match app.tab {
            Tab::Containers => app.move_container(1),
            Tab::Images => app.move_images(1),
            Tab::Volumes => app.move_volumes(1),
            Tab::Agents => app.move_agents(1),
            Tab::Ports => app.move_ports(1),
            Tab::Processes => app.move_procs(1),
            Tab::Hardware | Tab::Strata => {}
            Tab::Logs => {
                app.logs_follow = false;
                app.logs_scroll = (app.logs_scroll + 1).min(app.logs.len());
            }
        },
        KeyCode::Up | KeyCode::Char('k') => match app.tab {
            Tab::Containers => app.move_container(-1),
            Tab::Images => app.move_images(-1),
            Tab::Volumes => app.move_volumes(-1),
            Tab::Agents => app.move_agents(-1),
            Tab::Ports => app.move_ports(-1),
            Tab::Processes => app.move_procs(-1),
            Tab::Hardware | Tab::Strata => {}
            Tab::Logs => {
                app.logs_follow = false;
                app.logs_scroll = app.logs_scroll.saturating_sub(1);
            }
        },
        KeyCode::PageDown => match app.tab {
            Tab::Logs => {
                app.logs_follow = false;
                app.logs_scroll = (app.logs_scroll + 20).min(app.logs.len());
            }
            Tab::Containers => app.move_container(20),
            Tab::Images => app.move_images(20),
            Tab::Volumes => app.move_volumes(20),
            Tab::Agents => app.move_agents(20),
            Tab::Ports => app.move_ports(20),
            Tab::Processes => app.move_procs(20),
            Tab::Hardware | Tab::Strata => {}
        },
        KeyCode::PageUp => match app.tab {
            Tab::Logs => {
                app.logs_follow = false;
                app.logs_scroll = app.logs_scroll.saturating_sub(20);
            }
            Tab::Containers => app.move_container(-20),
            Tab::Images => app.move_images(-20),
            Tab::Volumes => app.move_volumes(-20),
            Tab::Agents => app.move_agents(-20),
            Tab::Ports => app.move_ports(-20),
            Tab::Processes => app.move_procs(-20),
            Tab::Hardware | Tab::Strata => {}
        },
        KeyCode::Char('g') if app.tab == Tab::Logs => {
            app.logs_follow = false;
            app.logs_scroll = 0;
        }
        KeyCode::Char('G') | KeyCode::Char('$') if app.tab == Tab::Logs => {
            app.logs_follow = true;
        }
        KeyCode::Enter => {
            if app.tab == Tab::Processes {
                app.toggle_proc_group();
            }
            if app.tab == Tab::Containers {
                if let Some(c) = app.selected_container().cloned() {
                    app.logs_gen += 1;
                    app.pending_logs_gen = Some(app.logs_gen);
                    app.logs_container = Some(c.id.clone());
                    app.logs_container_name = c.short_name();
                    app.logs.clear();
                    app.logs_scroll = 0;
                    app.logs_follow = true;
                    app.logs_active = false;
                    app.tab = Tab::Logs;
                    let id = c.id.clone();
                    spawn_logs(rt, docker, tx, &id, app.logs_gen);
                }
            }
        }
        KeyCode::Char('f') if app.tab == Tab::Logs => {
            app.logs_follow = !app.logs_follow;
        }
        KeyCode::Char('c') if app.tab == Tab::Logs => {
            app.logs.clear();
            app.logs_scroll = 0;
        }
        KeyCode::Char('s') if app.tab == Tab::Containers => {
            if let Some(c) = app.selected_container().cloned() {
                let act = if c.state == "running" { Pending::Stop } else { Pending::Start };
                app.pending_action = Some((c.id.clone(), act));
                spawn_action(rt, docker, tx, &c, act);
            }
        }
        KeyCode::Char('t') if app.tab == Tab::Containers => {
            if let Some(c) = app.selected_container().cloned() {
                app.pending_action = Some((c.id.clone(), Pending::Stop));
                spawn_action(rt, docker, tx, &c, Pending::Stop);
            }
        }
        KeyCode::Char('r') if app.tab == Tab::Containers => {
            if let Some(c) = app.selected_container().cloned() {
                app.pending_action = Some((c.id.clone(), Pending::Restart));
                spawn_action(rt, docker, tx, &c, Pending::Restart);
            }
        }
        KeyCode::Char('x') if app.tab == Tab::Containers => {
            if let Some(c) = app.selected_container() {
                app.confirm_remove = Some(c.id.clone());
            }
        }
        KeyCode::Char('l') if modifiers.contains(KeyModifiers::CONTROL) => {
            // repaint
        }
        _ => {}
    }
}

fn now_string() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", d.as_secs())
}

fn ui(f: &mut Frame, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // tabs
            Constraint::Min(1),    // content
            Constraint::Length(1), // status line
        ])
        .split(f.area());

    let tabs: Vec<Span> = [
        Tab::Containers,
        Tab::Images,
        Tab::Volumes,
        Tab::Agents,
        Tab::Hardware,
        Tab::Processes,
        Tab::Ports,
        Tab::Strata,
        Tab::Logs,
    ]
    .iter()
        .flat_map(|t| {
            let active = *t == app.tab;
            vec![
                Span::styled(
                    t.label(),
                    if active {
                        Style::default()
                            .fg(Color::Black)
                            .bg(Color::Cyan)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(Color::Gray)
                    },
                ),
                Span::raw(" "),
            ]
        })
        .collect();
    let tabs = Paragraph::new(Line::from(tabs)).block(Block::default().borders(Borders::ALL));
    f.render_widget(tabs, chunks[0]);

    let status = if let Some(e) = &app.error {
        format!(" error: {} ", e)
    } else if let Some(id) = &app.confirm_remove {
        format!(
            " remove {}? press y to confirm, n/Esc to cancel ",
            name_of(app, id)
        )
    } else if let Some((id, p)) = &app.pending_action {
        format!(" {} {}... ", p.verb(), name_of(app, id))
    } else if !app.status.is_empty() {
        format!(" {} ", app.status)
    } else {
        format!(
            " containers: {}  images: {}  volumes: {}  agents: {}  ports: {}  |  q quit  1-9 tab  / filter  enter logs  s start/stop  t stop  r restart  x remove  R refresh ",
            app.containers.len(),
            app.images.len(),
            app.volumes.len(),
            app.agents.len(),
            app.ports.len()
        )
    };
    f.render_widget(Paragraph::new(status), chunks[2]);

    match app.tab {
        Tab::Containers => render_containers(f, app, chunks[1]),
        Tab::Images => render_images(f, app, chunks[1]),
        Tab::Volumes => render_volumes(f, app, chunks[1]),
        Tab::Agents => render_agents(f, app, chunks[1]),
        Tab::Hardware => render_hardware(f, app, chunks[1]),
        Tab::Ports => render_ports(f, app, chunks[1]),
        Tab::Processes => render_procs(f, app, chunks[1]),
        Tab::Strata => render_strata(f, app, chunks[1]),
        Tab::Logs => render_logs(f, app, chunks[1]),
    }
}

fn name_of(app: &App, id: &str) -> String {
    app.containers
        .iter()
        .find(|c| c.id == id)
        .map(|c| c.short_name())
        .unwrap_or_else(|| id.to_string())
}

fn render_containers(f: &mut Frame, app: &mut App, area: ratatui::layout::Rect) {
    let list = app.filtered_containers();
    let sel = app.container_state.selected().unwrap_or(0);
    let rows: Vec<Row> = list
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let state_style = if c.state == "running" {
                Style::default().fg(Color::Green)
            } else if c.state.starts_with("exited") || c.state == "exited" {
                Style::default().fg(Color::Red)
            } else {
                Style::default().fg(Color::Yellow)
            };
            let row_style = if i == sel {
                Style::default()
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD | Modifier::SLOW_BLINK)
            } else {
                Style::default()
            };
            Row::new(vec![
                c.short_name(),
                c.image.clone(),
                Span::styled(c.state.clone(), state_style).to_string(),
                c.status.clone(),
                c.port_summary(),
            ])
            .style(row_style)
        })
        .collect();

    let title = if app.filtering {
        format!(" Containers (filter: {}) ", app.filter)
    } else {
        " Containers ".to_string()
    };

    let table = Table::new(
        rows,
        [
            Constraint::Length(24),
            Constraint::Length(28),
            Constraint::Length(10),
            Constraint::Min(20),
            Constraint::Length(24),
        ],
    )
    .header(
        Row::new(vec!["NAME", "IMAGE", "STATE", "STATUS", "PORTS"])
            .style(Style::default().fg(Color::Cyan)),
    )
    .block(Block::default().borders(Borders::ALL).title(title))
    .column_spacing(2);

    f.render_widget(table, area);
}

fn render_images(f: &mut Frame, app: &mut App, area: ratatui::layout::Rect) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let items: Vec<ListItem> = app
        .images
        .iter()
        .map(|i| {
            ListItem::new(Line::from(vec![
                Span::styled(
                    i.tag(),
                    Style::default().add_modifier(Modifier::BOLD),
                ),
                Span::raw(format!(
                    "  {}  {}  {}",
                    i.short_id(),
                    human_bytes(i.size),
                    ago(now - i.created)
                )),
            ]))
        })
        .collect();
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" Images "))
        .highlight_style(Style::default().bg(Color::DarkGray))
        .highlight_symbol("> ");
    f.render_stateful_widget(list, area, &mut app.images_state);
}

fn render_volumes(f: &mut Frame, app: &mut App, area: ratatui::layout::Rect) {
    let items: Vec<ListItem> = app
        .volumes
        .iter()
        .map(|v| {
            ListItem::new(Line::from(format!(
                "{}  driver={}  {}",
                v.name, v.driver, v.created
            )))
        })
        .collect();
    let list = List::new(items)
        .block(Block::default().borders(Borders::ALL).title(" Volumes "))
        .highlight_style(Style::default().bg(Color::DarkGray))
        .highlight_symbol("> ");
    f.render_stateful_widget(list, area, &mut app.volumes_state);
}

fn render_agents(f: &mut Frame, app: &mut App, area: ratatui::layout::Rect) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;
    let list = app.filtered_agents();
    let rows: Vec<Row> = list
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let started = now
                - a.started
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_secs() as i64)
                    .unwrap_or(0);
            let row_style = if Some(i) == app.agents_state.selected() {
                Style::default()
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            Row::new(vec![
                a.agent.to_string(),
                a.cwd
                    .as_deref()
                    .map(|c| c.rsplit('/').next().unwrap_or(c).to_string())
                    .unwrap_or_else(|| a.vendor.to_string()),
                a.lead_pid.to_string(),
                a.pids.len().to_string(),
                format!("{:.1}", a.cpu_pct),
                human_bytes(a.mem_bytes as i64),
                ago(started),
            ])
            .style(row_style)
        })
        .collect();

    let title = if app.filtering {
        format!(" Agents (filter: {}) ", app.filter)
    } else {
        " Agents ".to_string()
    };

    let table = Table::new(
        rows,
        [
            Constraint::Length(22),
            Constraint::Min(20),
            Constraint::Length(7),
            Constraint::Length(5),
            Constraint::Length(6),
            Constraint::Length(9),
            Constraint::Length(9),
        ],
    )
    .header(
        Row::new(vec!["AGENT", "PROJECT", "PID", "PROCS", "CPU%", "MEM", "UPTIME"])
            .style(Style::default().fg(Color::Cyan)),
    )
    .block(Block::default().borders(Borders::ALL).title(title))
    .column_spacing(2);

    f.render_widget(table, area);
}

/// ASCII sparkline of a series, scaled to [min..max].
fn sparkline(s: &[f64], width: usize) -> String {
    const CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    if s.is_empty() {
        return String::new();
    }
    let take: Vec<f64> = s.iter().rev().take(width).rev().copied().collect();
    let max = take.iter().cloned().fold(0.0_f64, f64::max).max(1.0);
    take.iter()
        .map(|v| CHARS[((*v / max) * 7.0).clamp(0.0, 7.0) as usize])
        .collect()
}

fn render_hardware(f: &mut Frame, app: &mut App, area: ratatui::layout::Rect) {
    let Some(s) = app.hw.clone() else {
        f.render_widget(
            Paragraph::new("collecting first sample…")
                .block(Block::default().borders(Borders::ALL).title(" Hardware ")),
            area,
        );
        return;
    };
    let top = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(13), Constraint::Min(4)])
        .split(area);

    let mem_used = s.mem_total.saturating_sub(s.mem_avail);
    let swap_used = s.swap_total.saturating_sub(s.swap_free);
    let mem_frac = mem_used as f64 / s.mem_total.max(1) as f64;
    let swap_frac = swap_used as f64 / s.swap_total.max(1) as f64;
    let gauge = |label: &str, frac: f64| -> Line {
        let filled = (frac * 30.0).round() as usize;
        let color = if frac > 0.9 {
            Color::Red
        } else if frac > 0.7 {
            Color::Yellow
        } else {
            Color::Green
        };
        Line::from(vec![
            Span::styled(format!("{label:<8}"), Style::default().fg(Color::Gray)),
            Span::styled(
                "█".repeat(filled) + &"░".repeat(30 - filled),
                Style::default().fg(color),
            ),
            Span::raw(format!(" {:>4.0}%", frac * 100.0)),
        ])
    };
    let inv = &hw_inventory::INVENTORY.cpu;
    let mut lines = vec![
        Line::from(vec![
            Span::styled(inv, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
            Span::raw(format!(
                "   {} cores   up {}",
                s.ncpu,
                ago(s.uptime as i64)
            )),
        ]),
        Line::from(vec![
            Span::styled(
                format!("CPU {:>5.1}%", s.cpu_pct),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
            Span::raw(format!("   load {} {} {}", s.load1, s.load5, s.load15)),
            s.cpu_temp_c
                .or(s.temp_c)
                .map(|t| Span::raw(format!("   cpu {}°C", t as f64)))
                .unwrap_or_default(),
        ]),
        gauge("memory", mem_frac),
        gauge("swap", swap_frac),
        Line::from(vec![
            Span::styled(format!("disk  {} ", human_rate(s.disk_read_bps)), Style::default().fg(Color::Gray)),
            Span::raw(format!("read   {} write/s", human_rate(s.disk_write_bps))),
        ]),
        Line::from(vec![
            Span::styled(format!("net   {} ", human_rate(s.net_rx_bps)), Style::default().fg(Color::Gray)),
            Span::raw(format!("rx     {} tx/s", human_rate(s.net_tx_bps))),
        ]),
        Line::from(vec![
            Span::styled("trend   ", Style::default().fg(Color::Gray)),
            Span::styled(
                sparkline(&app.hw_cpu_hist, 40),
                Style::default().fg(Color::Cyan),
            ),
        ]),
    ];
    // one line per GPU, one per disk
    for g in &app.gpus {
        let busy = g.busy_pct.map(|b| format!("{b:.0}% busy")).unwrap_or_else(|| "busy n/a".into());
        let temp = g.temp_c.map(|t| format!("{t:.0}°C")).unwrap_or_else(|| "n/a".into());
        let vram = if g.vram_total > 0 {
            format!(" {} of {} VRAM", human_bytes(g.vram_used as i64), human_bytes(g.vram_total as i64))
        } else {
            String::new()
        };
        lines.push(Line::from(vec![
            Span::styled("gpu     ", Style::default().fg(Color::Gray)),
            Span::styled(g.name.clone(), Style::default().fg(Color::Magenta)),
            Span::raw(format!("  {busy}  {temp}{vram}")),
        ]));
    }
    for d in &s.disks {
        // only drives the inventory can name; internal/OEM disks (sda…) hide
        let Some(nm) = hw_inventory::INVENTORY.disk_name(&d.name) else {
            continue;
        };
        let temp = d.temp_c.map(|t| format!("{t:.0}°C")).unwrap_or_else(|| "n/a".into());
        lines.push(Line::from(vec![
            Span::styled("ssd     ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{nm}  {} read  {} write  {temp}", human_rate(d.read_bps), human_rate(d.write_bps)),
                Style::default().fg(Color::White),
            ),
        ]));
        for u in hw_inventory::disk_usage().iter().filter(|u| u.kern == d.name) {
            let frac = u.used as f64 / u.total.max(1) as f64;
            let filled = (frac * 20.0).round() as usize;
            let color = if frac > 0.9 {
                Color::Red
            } else if frac > 0.75 {
                Color::Yellow
            } else {
                Color::Green
            };
            lines.push(Line::from(vec![
                Span::styled("         ", Style::default().fg(Color::Gray)),
                Span::styled(format!("{:<10}", u.mount), Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "█".repeat(filled) + &"░".repeat(20 - filled),
                    Style::default().fg(color),
                ),
                Span::raw(format!(
                    " {} / {}",
                    human_bytes(u.used as i64),
                    human_bytes(u.total as i64)
                )),
            ]));
        }
    }
    let vitals = Paragraph::new(lines)
        .block(Block::default().borders(Borders::ALL).title(" Hardware "));
    f.render_widget(vitals, top[0]);

    // sensor readings instead of per-core bars: every hwmon temperature
    let mut rows: Vec<Line> = Vec::new();
    if let Some(t) = s.cpu_temp_c.or(s.temp_c) {
        rows.push(Line::from(vec![
            Span::styled("cpu temp ", Style::default().fg(Color::Gray)),
            Span::styled(format!("{t:.0}°C"), Style::default().fg(Color::Green)),
        ]));
    }
    for x in &s.sensors {
        let color = if x.c > 85.0 {
            Color::Red
        } else if x.c > 70.0 {
            Color::Yellow
        } else {
            Color::Green
        };
        rows.push(Line::from(vec![
            Span::styled(format!("{:<8}", x.group), Style::default().fg(Color::Gray)),
            Span::styled(format!("{:<12}", x.label), Style::default().fg(Color::DarkGray)),
            Span::styled(format!("{:.0}°C", x.c), Style::default().fg(color)),
        ]));
    }
    if rows.is_empty() {
        rows.push(Line::from("no hwmon temperature sensors found"));
    }
    let body = Paragraph::new(rows).block(
        Block::default()
            .borders(Borders::ALL)
            .title(" Sensors "),
    );
    f.render_widget(body, top[1]);
}

fn render_ports(f: &mut Frame, app: &mut App, area: ratatui::layout::Rect) {
    let list = app.filtered_ports();
    let sel = app.ports_state.selected().unwrap_or(0);
    let rows: Vec<Row> = list
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let state_color = match p.state {
                "LISTEN" => Color::Green,
                "ESTABLISHED" => Color::Cyan,
                "-" => Color::DarkGray,
                _ => Color::Yellow,
            };
            let row_style = if i == sel {
                Style::default()
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            Row::new(vec![
                p.proto.to_string(),
                format!("v{}", p.family),
                p.addr.clone(),
                p.port.to_string(),
                Span::styled(p.state.to_string(), Style::default().fg(state_color)).to_string(),
                p.pid.map(|x| x.to_string()).unwrap_or_else(|| "-".into()),
                p.proc_name.clone().unwrap_or_else(|| "-".into()),
                p.cwd
                    .as_deref()
                    .map(|c| c.rsplit('/').next().unwrap_or(c).to_string())
                    .unwrap_or_else(|| "-".into()),
            ])
            .style(row_style)
        })
        .collect();

    let title = if app.filtering {
        format!(" Ports (filter: {}) ", app.filter)
    } else {
        " Ports ".to_string()
    };

    // Header goes in the block title so it survives even when the terminal
    // is too short to render any table rows.
    let header = "PROTO  IP  ADDRESS  PORT  STATE  PID  PROCESS  CWD";
    let table = Table::new(
        rows,
        [
            Constraint::Length(4),
            Constraint::Length(3),
            Constraint::Length(15),
            Constraint::Length(6),
            Constraint::Length(12),
            Constraint::Length(7),
            Constraint::Length(16),
            Constraint::Min(14),
        ],
    )
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(title)
            .title_bottom(Line::from(format!(" {header} ")).style(Style::default().fg(Color::Cyan))),
    )
    .column_spacing(2);

    f.render_widget(table, area);
}

fn render_procs(f: &mut Frame, app: &mut App, area: ratatui::layout::Rect) {
    // one row per app (processes sharing a desktop icon collapse), then the
    // standalone processes; Enter on an app row expands it to its members.
    let rows_data = app.sorted_procs();
    let groups = crate::procs::group_rows(&rows_data, app.proc_sort, app.proc_sort_desc);
    let sel = app.procs_state.selected().unwrap_or(0);
    let mut rows: Vec<Row> = Vec::new();
    let mut i = 0usize;
    for g in &groups {
        let open = app.proc_group.as_deref() == Some(g.icon.as_str());
        let name = if g.sub.is_empty() {
            g.title.clone()
        } else {
            format!("{} [{}]", g.title, g.sub)
        };
        let cpu = g.cpu;
        let cpu_color = if cpu > 80.0 {
            Color::Red
        } else if cpu > 30.0 {
            Color::Yellow
        } else {
            Color::White
        };
        let cpu_cell = Span::styled(format!("{cpu:>5.1}"), Style::default().fg(cpu_color)).to_string();
        let row_style = if i == sel {
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
        };
        rows.push(
            Row::new(vec![
                name,
                g.user.clone(),
                g.pid_label.clone(),
                cpu_cell,
                human_bytes(g.mem as i64),
                if g.members.len() > 1 {
                    g.members.len().to_string()
                } else {
                    g.members[0].state.to_string()
                },
                g.members.iter().map(|m| m.threads as u64).sum::<u64>().to_string(),
                if open {
                    g.members
                        .iter()
                        .map(|m| m.cmdline.clone())
                        .collect::<Vec<_>>()
                        .join(" | ")
                } else {
                    g.members[0].cmdline.clone()
                },
            ])
            .style(row_style),
        );
        i += 1;
        if open {
            for m in &g.members {
                let mcpu = m.cpu_pct;
                let mc = if mcpu > 80.0 {
                    Color::Red
                } else if mcpu > 30.0 {
                    Color::Yellow
                } else {
                    Color::Gray
                };
                rows.push(
                    Row::new(vec![
                        format!("  {}", m.name),
                        m.user.clone(),
                        m.pid.to_string(),
                        format!("{mcpu:>5.1}"),
                        human_bytes(m.rss_bytes as i64),
                        m.state.to_string(),
                        m.threads.to_string(),
                        m.cmdline.clone(),
                    ])
                    .style(Style::default().fg(mc)),
                );
                i += 1;
            }
        }
    }

    let title = if app.filtering {
        format!(" Processes (filter: {}) ", app.filter)
    } else {
        format!(
            " Processes: {} apps, {} procs  (c/m/n/p sort, Enter expand) ",
            groups.len(),
            rows_data.len()
        )
    };

    let header = Row::new(vec![
        "APP",
        "USER",
        "PID",
        "CPU",
        "MEMORY",
        "S",
        "TH",
        "CMDLINE",
    ])
    .style(Style::default().fg(Color::Cyan));

    let table = Table::new(
        rows,
        [
            Constraint::Length(20),
            Constraint::Length(9),
            Constraint::Length(7),
            Constraint::Length(6),
            Constraint::Length(9),
            Constraint::Length(2),
            Constraint::Length(3),
            Constraint::Min(16),
        ],
    )
    .header(header)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(title),
    )
    .column_spacing(2);

    f.render_widget(table, area);
}

fn render_logs(f: &mut Frame, app: &mut App, area: ratatui::layout::Rect) {
    let title = format!(
        " Logs: {}  ({} lines, follow: {}{}) ",
        if app.logs_container_name.is_empty() {
            "(no container selected)"
        } else {
            &app.logs_container_name
        },
        app.logs.len(),
        if app.logs_follow { "on" } else { "off" },
        if app.logs_active { ", streaming" } else { "" },
    );

    let visible_height = area.height.saturating_sub(2) as usize;
    let start = if app.logs_follow {
        app.logs.len().saturating_sub(visible_height)
    } else {
        app.logs_scroll.min(app.logs.len().saturating_sub(1))
    };
    let end = (start + visible_height).min(app.logs.len());

    let lines: Vec<Line> = app.logs[start..end]
        .iter()
        .map(|l| {
            let style = if l.stream == StreamKind::Stderr {
                Style::default().fg(Color::Red)
            } else {
                Style::default()
            };
            Line::from(Span::styled(l.text.clone(), style))
        })
        .collect();

    let para = Paragraph::new(lines).block(Block::default().borders(Borders::ALL).title(title));
    f.render_widget(para, area);
}

fn render_strata(f: &mut Frame, app: &mut App, area: ratatui::layout::Rect) {
    if app.strata_cfg.api_key.trim().is_empty() && app.strata_health.is_none() && app.strata_error.is_none() {
        let msg = format!(
            "Strata is not configured yet.\n\n\
             Point SysKamling at a Strata inference server:\n\
             1. open the Strata tab in the GUI and fill in base URL + API key, or\n\
             2. write ~/.config/syskamling/config.json by hand:\n\n\
             {{\n  \"strata\": {{\n    \"endpoint\": \"http://127.0.0.1:9191\",\n    \"api_key\": \"sk-...\"\n  }}\n}}\n\n\
             endpoint in use: {} — press r to retry.",
            app.strata_cfg.base()
        );
        f.render_widget(
            Paragraph::new(msg).block(Block::default().borders(Borders::ALL).title(" Strata ")),
            area,
        );
        return;
    }
    // the poller only runs on this screen, so nudge it right away instead of
    // waiting out the refresh interval after every tab switch
    if app.strata_metrics.is_none() && app.strata_error.is_none() {
        let body = format!("contacting {}…", app.strata_cfg.base());
        f.render_widget(
            Paragraph::new(body).block(Block::default().borders(Borders::ALL).title(" Strata ")),
            area,
        );
        return;
    }

    let top = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(11), Constraint::Min(4)])
        .split(area);

    let m = app.strata_metrics.as_ref().unwrap();
    let last = m.requests.first();

    // model state line, mirroring the web monitor's pill
    let (state, state_color, detail) = match m.state.as_str() {
        "reading" => (
            format!(
                "Reading prompt{}",
                match (m.prompt_read, m.prompt_total) {
                    (Some(r), Some(t)) if t > 0 => format!(" · {}%", (100 * r) / t),
                    _ => String::new(),
                }
            ),
            Color::Cyan,
            format!(
                "{} read · {} tok/s prefill",
                fmt_tokens_tui_i(m.prompt_read.unwrap_or(0)),
                strata::prefill_now(m, last).map(|v| format!("{v:.0}")).unwrap_or_else(|| "—".into())
            ),
        ),
        "generating" => (
            m.phase.clone().unwrap_or_else(|| "Generating".into()),
            Color::Green,
            format!(
                "{} tokens · {} tok/s",
                fmt_tokens_tui_i(m.generated.unwrap_or(0)),
                m.tok_s.map(|v| format!("{v:.1}")).unwrap_or_else(|| "—".into())
            ),
        ),
        _ => (
            "Idle".into(),
            Color::DarkGray,
            last.map(|r| {
                format!(
                    "last: {} tokens{}",
                    fmt_tokens_tui_i(r.output_tokens.unwrap_or(0)),
                    r.decode_tok_s.map(|d| format!(" at {d:.1} tok/s")).unwrap_or_default()
                )
            })
            .unwrap_or_else(|| "waiting for a request".into()),
        ),
    };
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(vec![
        Span::styled("● ", Style::default().fg(state_color).add_modifier(Modifier::BOLD)),
        Span::styled(
            format!("{state}{}", if m.queued > 0 { format!(" · {} queued", m.queued) } else { String::new() }),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!("  {detail}")),
    ]));

    let hist = |k: &str| -> Vec<f64> { m.hist.get(k).cloned().unwrap_or_default() };
    let kpi = |label: &str, val: String, sub: String, series: &[f64]| -> Vec<Span> {
        vec![
            Span::styled(format!("{label:<13}"), Style::default().fg(Color::DarkGray)),
            Span::styled(val, Style::default().add_modifier(Modifier::BOLD)),
            Span::styled(format!(" {sub:<24}"), Style::default().fg(Color::DarkGray)),
            Span::raw(sparkline(series, 12)),
        ]
    };
    lines.push(Line::from({ let mut v = kpi(
            "GPU speed",
            strata::decode_now(m, last).map(|v| format!("{v:.1} tok/s")).unwrap_or_else(|| "—".into()),
            "decode".into(),
            &hist("tok_s"),
        );
        let kpi2 = kpi(
            "GPU prefill",
            strata::prefill_now(m, last).map(|v| format!("{:.0} t/s", v)).unwrap_or_else(|| "—".into()),
            format!("{} slots", m.parallel),
            &hist("prefill_tok_s_mean"),
        );
        v.extend(kpi2);
        v
    }));
    lines.push(Line::from({ let mut v = kpi(
            "GPU load",
            m.gpu_util.map(|v| format!("{v:.0}%")).unwrap_or_else(|| "—".into()),
            m.gpu_name.clone().unwrap_or_default(),
            &hist("gpu_util"),
        );
        let kpi2 = kpi(
            "VRAM",
            m.gpu_mem_used.map(|v| format!("{:.1} GB", v / 1e9)).unwrap_or_else(|| "—".into()),
            m.gpu_mem_total
                .map(|t| format!("of {:.0} GB", t / 1e9))
                .unwrap_or_else(|| "of ?".into()),
            &hist("gpu_mem_used"),
        );
        v.extend(kpi2);
        v
    }));
    lines.push(Line::from({ let mut v = kpi(
            "GPU temp",
            m.gpu_temp.map(|v| format!("{v:.0} °C")).unwrap_or_else(|| "—".into()),
            m.gpu_power
                .map(|p| format!("{} W of {:.0} W", p, m.gpu_power_limit.unwrap_or(0.0)))
                .unwrap_or_else(|| "power ?".into()),
            &hist("gpu_temp"),
        );
        let kpi2 = kpi(
            "PCIe",
            m.gpu_pcie_gen_max.or(m.gpu_pcie_gen).map(|g| format!("Gen{g}")).unwrap_or_else(|| "—".into()),
            format!(
                "x{} · {}",
                m.gpu_pcie_width.unwrap_or(0),
                m.gpu_pcie_rx_mb.map(|v| format!("{v:.1} MB/s")).unwrap_or_else(|| "—".into())
            ),
            &hist("gpu_pcie_rx_mb"),
        );
        v.extend(kpi2);
        v
    }));
    lines.push(Line::from({ let mut v = kpi(
            "Host",
            m.cpu.map(|v| format!("CPU {:.0}%", v)).unwrap_or_else(|| "—".into()),
            format!(
                "disk read {}",
                m.disk_read_mb.map(|v| format!("{v:.1} MB/s")).unwrap_or_else(|| "—".into())
            ),
            &hist("cpu"),
        );
        let kpi2 = kpi(
            "First token",
            m.ttft.p50.map(|v| format!("{v:.2} s")).unwrap_or_else(|| "—".into()),
            format!(
                "p90 {} · {} reqs{}",
                m.ttft.p90.map(|v| format!("{v:.2} s")).unwrap_or_else(|| "—".into()),
                m.ttft.count,
                m.itl.p50.map(|v| format!(" · {:.0} ms/token", v * 1000.0)).unwrap_or_default()
            ),
            &hist("ttft"),
        );
        v.extend(kpi2);
        v
    }));
    let used = strata::context_used(m, last);
    let ctx_pct = if m.max_context > 0 { (100.0 * used as f64 / m.max_context as f64).min(100.0) as usize } else { 0 };
    lines.push(Line::from(vec![
        Span::styled(format!("{:<13}", "Context"), Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{} / {} · {ctx_pct}%", fmt_tokens_tui_i(used), fmt_tokens_tui_i(m.max_context)),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!(
                "  since start: {} reqs · {} tok in · {} tok out",
                fmt_tokens_tui_i(m.tot_requests),
                fmt_tokens_tui_i(m.tot_prompt_tokens),
                fmt_tokens_tui_i(m.tot_output_tokens)
            ),
            Style::default().fg(Color::DarkGray),
        ),
    ]));
    if let Some(c) = &m.conv {
        lines.push(Line::from(vec![
            Span::styled(format!("{:<13}", "Conv cache"), Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!(
                    "{}/{} requests reused",
                    fmt_tokens_tui_i(c.requests_reused),
                    fmt_tokens_tui_i(c.requests)
                ),
                Style::default().add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(
                    "  {} parked/{} slots · {} tok reused · {} parked, {} restored",
                    c.parked.unwrap_or(0),
                    c.slots.unwrap_or(0),
                    fmt_tokens_tui_i(c.reused_tokens),
                    c.parks.unwrap_or(0),
                    c.restores.unwrap_or(0)
                ),
                Style::default().fg(Color::DarkGray),
            ),
        ]));
    }
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(format!(" Strata · {} · {} ", app.strata_cfg.base(), if m.model.is_empty() { "strata" } else { &m.model })),
        ),
        top[0],
    );

    let show_n = if app.strata_show_all { m.requests.len().min(100) } else { m.requests.len().min(10) };
    let rows: Vec<Row> = m
        .requests
        .iter()
        .take(show_n)
        .map(|r| {
            let (fl, fc) = finish_style_tui(Some(&r.finish));
            Row::new(vec![
                clock_label_tui(r.time),
                Span::styled(fl, Style::default().fg(fc)).to_string(),
                r.prompt_tokens.map(fmt_tokens_tui_i).unwrap_or_else(|| "—".into()),
                r.reused.map(fmt_tokens_tui_i).unwrap_or_else(|| "0".into()),
                r.output_tokens.map(fmt_tokens_tui_i).unwrap_or_else(|| "—".into()),
                strata::prefill_rate(r).map(|v| format!("{:.0}", v)).unwrap_or_else(|| "—".into()),
                r.decode_tok_s.map(|v| format!("{v:.1}")).unwrap_or_else(|| "—".into()),
                r.hit_rate.map(|h| format!("{:.1}%", h * 100.0)).unwrap_or_else(|| "—".into()),
                r.duration_s.map(|v| format!("{v:.1} s")).unwrap_or_else(|| "—".into()),
            ])
        })
        .collect();
    let table = Table::new(
        rows,
        [
            Constraint::Length(9),
            Constraint::Length(12),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Length(7),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Length(8),
            Constraint::Min(9),
        ],
    )
    .header(
        Row::new(vec!["TIME", "FINISH", "PROMPT", "REUSED", "OUTPUT", "PRE t/s", "DEC t/s", "HIT", "DUR"])
            .style(Style::default().fg(Color::Cyan)),
    )
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title(format!(" Recent requests · {} kept · a: show all ", m.requests_kept)),
    )
    .column_spacing(2);
    f.render_widget(table, top[1]);
}

fn fmt_tokens_tui(v: f64) -> String {
    if v >= 1e9 {
        format!("{:.1}B", v / 1e9)
    } else if v >= 1e6 {
        format!("{:.1}M", v / 1e6)
    } else if v >= 1e3 {
        format!("{:.1}k", v / 1e3)
    } else {
        format!("{v:.0}")
    }
}

fn fmt_tokens_tui_i(v: i64) -> String {
    fmt_tokens_tui(v as f64)
}

/// Seconds-since-epoch → HH:MM:SS local-ish (UTC is fine for a monitor).
fn clock_label_tui(secs: f64) -> String {
    let t = secs as i64;
    format!("{:02}:{:02}:{:02}", (t / 3600) % 24, (t / 60) % 60, t % 60)
}

fn finish_style_tui(f: Option<&str>) -> (String, Color) {
    let label = match f.unwrap_or("none") {
        "eos" => "eos",
        "stop" => "stop",
        "length" => "length",
        "user_cancel" => "cancelled",
        "context_limit" => "ctx limit",
        other => other,
    };
    let color = match f.unwrap_or("none") {
        "eos" | "stop" => Color::Green,
        "length" => Color::Blue,
        "user_cancel" => Color::Yellow,
        "context_limit" => Color::Red,
        _ => Color::DarkGray,
    };
    (label.into(), color)
}

fn human_bytes(b: i64) -> String {
    const UNITS: [&str; 6] = ["B", "KB", "MB", "GB", "TB", "PB"];
    let mut n = b as f64;
    let mut i = 0;
    while n >= 1024.0 && i < UNITS.len() - 1 {
        n /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{}{}", b, UNITS[i])
    } else {
        format!("{:.1}{}", n, UNITS[i])
    }
}

/// "1.2MB/s"-style rate for the Hardware screen.
fn human_rate(bps: f64) -> String {
    if bps < 1.0 {
        return "0B/s".into();
    }
    let mut n = bps;
    const UNITS: [&str; 4] = ["B", "KB", "MB", "GB"];
    let mut i = 0;
    while n >= 1024.0 && i < UNITS.len() - 1 {
        n /= 1024.0;
        i += 1;
    }
    format!("{:.0}{}/s", n, UNITS[i])
}

fn ago(secs: i64) -> String {
    let secs = secs.max(0);
    if secs < 60 {
        format!("{secs}s ago")
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86400)
    }
}

#[cfg(test)]
mod framing_tests {
    use super::*;

    #[test]
    fn chunked_single_and_split() {
        let mut fr = Chunked::default();
        // one write: two chunks + terminator
        let out = fr.feed(b"5\r\nhello\r\n6\r\nworld!\r\n0\r\n\r\n");
        assert_eq!(out, b"helloworld!");
        assert!(fr.done);
    }

    #[test]
    fn chunked_across_writes() {
        // one chunk whose size line and payload straddle write boundaries,
        // then the terminator split mid-CRLF.
        let body: Vec<u8> = b"abcdefghij".iter().copied().collect(); // 10 bytes
        let mut wire: Vec<u8> = Vec::new();
        wire.extend(b"a\r\n");
        wire.extend(&body);
        wire.extend(b"\r\n0\r\n\r\n");
        let mut fr = Chunked::default();
        let mut acc = Vec::new();
        let cuts = [2usize, 5, 12, 14, wire.len()];
        let mut prev = 0;
        for &c in &cuts {
            acc.extend(fr.feed(&wire[prev..c]));
            prev = c;
        }
        assert_eq!(acc, body);
        assert!(fr.done);
    }

    #[test]
    fn chunked_extension_line() {
        let mut fr = Chunked::default();
        let out = fr.feed(b"4;ext=1\r\ndata\r\n0\r\n\r\n");
        assert_eq!(out, b"data");
    }

    #[test]
    fn multiplexer_frames() {
        let mut dec = crate::docker::LogStreamDecoder::new();
        let frame = |kind: u8, payload: &[u8]| -> Vec<u8> {
            let mut v = vec![kind, 0, 0, 0];
            v.extend_from_slice(&(payload.len() as u32).to_be_bytes());
            v.extend_from_slice(payload);
            v
        };
        let mut bytes = frame(1, b"out line\n");
        bytes.extend(frame(2, b"err line\n"));
        let lines = dec.push(&bytes);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "out line");
        assert_eq!(lines[0].stream, StreamKind::Stdout);
        assert_eq!(lines[1].text, "err line");
        assert_eq!(lines[1].stream, StreamKind::Stderr);
    }

    #[test]
    fn multiplexer_partial_frames() {
        let mut dec = crate::docker::LogStreamDecoder::new();
        let full = {
            let payload = b"multi\nline\n";
            let mut v = vec![1, 0, 0, 0];
            v.extend_from_slice(&(payload.len() as u32).to_be_bytes());
            v.extend_from_slice(payload);
            v
        };
        let (a, b) = full.split_at(6);
        assert!(dec.push(a).is_empty());
        let lines = dec.push(b);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0].text, "multi");
        assert_eq!(lines[1].text, "line");
    }
}

#[cfg(test)]
mod procs_screen_tests {
    use super::*;
    use crate::procs::{ProcInfo, SortCol};
    use ratatui::backend::TestBackend;

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

    /// Render the whole TUI (nav tabs + processes table) to a text buffer.
    fn screen(app: &mut App) -> Vec<String> {
        let backend = TestBackend::new(160, 40);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| ui(f, app)).unwrap();
        let cells = terminal.backend().buffer().clone();
        let h = cells.area().height as usize;
        let w = cells.area().width as usize;
        (0..h)
            .map(|y| {
                (0..w)
                    .map(|x| cells[(x as u16, y as u16)].symbol())
                    .collect::<String>()
                    .trim()
                    .to_owned()
            })
            .collect()
    }

    fn app_with_procs() -> App {
        let mut app = App::new();
        app.tab = Tab::Processes;
        app.procs = vec![
            row(1, "systemd", "/sbin/init", 0.1, 12 << 20),
            row(900, "hog", "hog --big", 71.0, 9 << 30),
            row(901, "busy", "busy-loop", 99.0, 100 << 20),
        ];
        app
    }

    /// The Processes screen is a real table: header row, then one row per
    /// app (grouped), with the USER column populated.
    #[test]
    fn procs_screen_renders_a_table_with_a_user_column() {
        let mut app = app_with_procs();
        let lines = screen(&mut app);
        let joined = lines.join("\n");
        assert!(joined.contains("APP"), "no APP header");
        assert!(joined.contains("USER"), "no USER header");
        assert!(joined.contains("rifan"), "user column never painted");
        assert!(joined.contains("901"), "pid column never painted");
        // default sort is memory desc: hog (9 GB) above busy (100 MB)
        let hog = lines.iter().position(|l| l.contains("hog")).unwrap();
        let busy = lines.iter().position(|l| l.contains("busy")).unwrap();
        assert!(hog < busy, "memory-desc: hog above busy");
    }

    /// The TUI table collapses processes that share a desktop icon into one
    /// app row ("name [N processes]", "N pids"), and Enter expands it to the
    /// member rows; pressing Enter again collapses it.
    #[test]
    fn app_rows_collapse_and_enter_expands_members() {
        let mut app = App::new();
        app.tab = Tab::Processes;
        let mut a = row(100, "chrome", "/opt/google/chrome/chrome", 20.0, 300 << 20);
        a.app_icon = "google-chrome".into();
        let mut b = row(101, "chrome", "/opt/google/chrome/chrome --type=zygote", 5.0, 50 << 20);
        b.app_icon = "google-chrome".into();
        app.procs = vec![a, b, row(1, "systemd", "/sbin/init", 0.1, 12 << 20)];
        let lines = screen(&mut app);
        let joined = lines.join("\n");
        assert!(joined.contains("chrome [2 processes]"), "collapsed app row missing:\n{joined}");
        assert!(joined.contains("2 pids"), "pid column must count the group");
        // collapsed: the zygote cmdline must not show; the title bar counts apps
        assert!(joined.contains("2 apps, 3 procs"), "title must count apps and procs");
        // cursor on the first row (chrome), Enter expands
        app.procs_state.select(Some(0));
        app.toggle_proc_group();
        let lines = screen(&mut app);
        let joined = lines.join("\n");
        assert!(joined.contains("zygote"), "expanded group must show member cmdlines");
        // both member pids appear
        assert!(joined.contains("100") && joined.contains("101"), "member pids paint");
        // cursor navigation counts the expanded rows
        assert_eq!(app.proc_row_count(), 4, "2 group rows + 2 members for the expanded one");
        app.toggle_proc_group();
        let lines = screen(&mut app);
        assert!(!lines.join("\n").contains("zygote"), "Enter collapses again");
    }

    /// Tab cycling must visit Processes, and the nav bar must show it in
    /// order (Hardware, Processes, Ports).
    #[test]
    fn tab_cycle_visits_processes_in_order() {
        assert_eq!(Tab::Hardware.next(), Tab::Processes);
        assert_eq!(Tab::Processes.next(), Tab::Ports);
        assert_eq!(Tab::Ports.prev(), Tab::Processes);
        assert_eq!(Tab::Processes.prev(), Tab::Hardware);
        let mut app = App::new();
        app.tab = Tab::Hardware;
        app.tab = app.tab.next();
        assert_eq!(app.tab, Tab::Processes);
        // the nav bar lists the tabs in order, Processes between the two
        let lines = screen(&mut app);
        let nav = lines
            .iter()
            .find(|l| l.contains("Processes"))
            .expect("nav bar never shows Processes");
        let at = |t: &str| nav.find(t).unwrap();
        assert!(at("Hardware") < at("Processes") && at("Processes") < at("Ports"),
            "nav order: {nav}");
    }

    /// 'c' sorts by CPU descending; pressing it again flips to ascending.
    #[test]
    fn cpu_sort_key_orders_rows_and_flips() {
        let mut app = app_with_procs();
        app.sort_procs(SortCol::Cpu);
        let lines = screen(&mut app);
        let busy = lines.iter().position(|l| l.contains("busy")).unwrap();
        let hog = lines.iter().position(|l| l.contains("hog")).unwrap();
        assert!(busy < hog, "cpu-desc: busy above hog");
        app.sort_procs(SortCol::Cpu);
        let lines = screen(&mut app);
        let sysd = lines.iter().position(|l| l.contains("systemd")).unwrap();
        let busy = lines.iter().position(|l| l.contains("busy")).unwrap();
        assert!(sysd < busy, "cpu-asc: systemd first");
    }

    /// Regression for the GUI report ("50 GB strata-host never moves to the
    /// top when sorting desc"): the shared grouping must rank standalone
    /// processes and app groups together, in the TUI too.
    #[test]
    fn memory_sort_puts_a_big_standalone_process_above_app_groups() {
        let mut app = App::new();
        app.tab = Tab::Processes;
        let mut a = row(100, "chrome", "/opt/google/chrome/chrome", 20.0, 5 << 30);
        a.app_icon = "google-chrome".into();
        let mut b = row(101, "chrome_crashpad", "/opt/google/chrome/chrome", 1.0, 1 << 30);
        b.app_icon = "google-chrome".into();
        app.procs = vec![a, b, row(200, "strata-host", "strata --serve", 3.0, 50 << 30)];
        app.sort_procs(SortCol::Mem); // already mem; flips desc -> asc
        app.sort_procs(SortCol::Mem); // back to desc
        let lines = screen(&mut app);
        let host = lines.iter().position(|l| l.contains("strata-host")).expect("strata-host row");
        let chrome = lines.iter().position(|l| l.contains("chrome [2 processes]")).expect("chrome group");
        assert!(host < chrome, "memory-desc: 50 GiB standalone above the 6 GiB app group:\n{}", lines.join("\n"));
    }
}
