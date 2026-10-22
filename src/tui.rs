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

#[derive(PartialEq, Eq, Clone, Copy)]
enum Tab {
    Containers,
    Images,
    Volumes,
    Agents,
    Logs,
}

impl Tab {
    fn next(self) -> Self {
        match self {
            Tab::Containers => Tab::Images,
            Tab::Images => Tab::Volumes,
            Tab::Volumes => Tab::Agents,
            Tab::Agents => Tab::Logs,
            Tab::Logs => Tab::Containers,
        }
    }
    fn prev(self) -> Self {
        match self {
            Tab::Containers => Tab::Logs,
            Tab::Images => Tab::Containers,
            Tab::Volumes => Tab::Images,
            Tab::Agents => Tab::Volumes,
            Tab::Logs => Tab::Agents,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Tab::Containers => " Containers ",
            Tab::Images => " Images ",
            Tab::Volumes => " Volumes ",
            Tab::Agents => " Agents ",
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
        KeyCode::Char('5') => app.tab = Tab::Logs,
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
        },
        KeyCode::Char('g') if app.tab == Tab::Logs => {
            app.logs_follow = false;
            app.logs_scroll = 0;
        }
        KeyCode::Char('G') | KeyCode::Char('$') if app.tab == Tab::Logs => {
            app.logs_follow = true;
        }
        KeyCode::Enter => {
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

    let tabs: Vec<Span> = [Tab::Containers, Tab::Images, Tab::Volumes, Tab::Agents, Tab::Logs]
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
            " containers: {}  images: {}  volumes: {}  agents: {}  |  q quit  tab switch  / filter  enter logs  s start/stop  t stop  r restart  x remove  R refresh ",
            app.containers.len(),
            app.images.len(),
            app.volumes.len(),
            app.agents.len()
        )
    };
    f.render_widget(Paragraph::new(status), chunks[2]);

    match app.tab {
        Tab::Containers => render_containers(f, app, chunks[1]),
        Tab::Images => render_images(f, app, chunks[1]),
        Tab::Volumes => render_volumes(f, app, chunks[1]),
        Tab::Agents => render_agents(f, app, chunks[1]),
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
