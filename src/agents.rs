//! Detection of running AI coding agents via /proc.
//!
//! Every scan reads each process's executable path, command line and cwd,
//! then matches against a table of known agent signatures (CLI agents, IDE
//! extensions, agent hosts like omp). Raw hits are grouped into sessions:
//! one agent launch usually spawns a small process tree of the same agent
//! kind, and the user thinks of that tree as one session.
//!
//! Claude Code sessions also get an [`Activity`]: their live status and
//! recent steps, read from Claude's own state files under `~/.claude`.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use serde_json::Value;

/// One running agent session: a group of processes sharing an agent kind
/// and a working directory.
#[derive(Clone, Debug)]
pub struct AgentSession {
    pub agent: &'static str,
    /// Human-readable vendor label, e.g. "Anthropic".
    pub vendor: &'static str,
    /// Project directory the session runs in, if readable.
    pub cwd: Option<String>,
    /// Representative process (lowest pid in the group).
    pub lead_pid: u32,
    /// All pids belonging to this session.
    pub pids: Vec<u32>,
    /// CPU percent since the previous scan (0.0 on the first scan).
    pub cpu_pct: f64,
    /// RSS in bytes, summed over the group.
    pub mem_bytes: u64,
    /// Start time of the lead process.
    pub started: SystemTime,
    /// Shortest command line in the group, for display.
    pub cmdline: String,
    /// What the session is doing, for agents that expose it (Claude Code).
    pub activity: Option<Activity>,
}

/// Live state of a Claude Code session, from `~/.claude/sessions/<pid>.json`
/// (status, session id) and the session transcript
/// `~/.claude/projects/<cwd>/<session id>.jsonl` (title, prompt, steps).
#[derive(Clone, Debug, Default)]
pub struct Activity {
    /// Claude's own status: "busy", "shell", "idle" or "waiting".
    pub status: String,
    /// When the status last changed, epoch seconds.
    pub status_since: Option<i64>,
    /// AI-generated session title, else Claude's derived session name.
    pub title: Option<String>,
    /// The prompt the session is working on.
    pub last_prompt: Option<String>,
    pub model: Option<String>,
    pub branch: Option<String>,
    /// Tokens in the context window as of the last model call.
    pub context_tokens: Option<u64>,
    /// Newest transcript entry, epoch seconds.
    pub last_active: Option<i64>,
    /// Recent steps, oldest first.
    pub steps: Vec<Step>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepKind {
    /// The user's prompt.
    Prompt,
    /// Text the agent wrote back.
    Reply,
    /// A tool call; `Step::tool` names it.
    Tool,
}

#[derive(Clone, Debug)]
pub struct Step {
    /// Epoch seconds.
    pub at: Option<i64>,
    pub kind: StepKind,
    /// Display name of the tool, e.g. "Bash" or "Atlassian · getJiraIssue".
    pub tool: String,
    /// One-line summary: the command, file, pattern or text.
    pub text: String,
    /// Tool call with no result yet.
    pub pending: bool,
    /// Tool call whose result came back as an error.
    pub failed: bool,
}

/// How many steps a session keeps for display.
const MAX_STEPS: usize = 60;
/// Transcripts grow to tens of MB; only the tail is read.
const TRANSCRIPT_TAIL: u64 = 1 << 20;

/// How a signature identifies its process.
enum Match {
    /// Basename of the executable (or argv[0]) equals this exactly.
    Exe(&'static str),
    /// Basename starts with this (covers versioned names like `cursor-1.2`).
    ExePrefix(&'static str),
    /// Any argv token equals this exactly.
    ArgToken(&'static str),
    /// Any argv token contains this substring.
    ArgContains(&'static str),
    /// Any argv token starts with this.
    ArgPrefix(&'static str),
}

struct Sig {
    agent: &'static str,
    vendor: &'static str,
    /// Extra condition on the full command line (lowercased).
    need: Option<&'static str>,
    /// Reject when the command line (lowercased) contains this.
    reject: Option<&'static str>,
    m: Match,
}

use Match::*;
/// Order matters: first match wins, so specific signatures come before
/// generic ones (e.g. `code` with the Copilot extension before bare `code`).
const SIGNATURES: &[Sig] = &[
    // --- terminal / CLI agents -------------------------------------------
    Sig { agent: "Claude Code", vendor: "Anthropic", need: None, reject: None, m: Exe("claude") },
    Sig { agent: "Codex CLI", vendor: "OpenAI", need: None, reject: None, m: Exe("codex") },
    // Codex desktop (ships as a ChatGPT-branded Electron shell): group the
    // GUI main processes only; renderers/gpu/zygotes are noise.
    Sig { agent: "Codex Desktop", vendor: "OpenAI", need: Some("/.config/codex"), reject: Some("--type="), m: Exe("chatgpt") },
    Sig { agent: "Gemini CLI", vendor: "Google", need: None, reject: None, m: Exe("gemini") },
    Sig { agent: "OpenCode", vendor: "SST", need: None, reject: None, m: Exe("opencode") },
    Sig { agent: "Aider", vendor: "Aider", need: None, reject: None, m: ArgToken("aider") },
    Sig { agent: "Aider", vendor: "Aider", need: None, reject: None, m: ArgContains("/aider/") },
    Sig { agent: "Goose", vendor: "Block", need: None, reject: None, m: Exe("goose") },
    Sig { agent: "Amp", vendor: "Sourcegraph", need: None, reject: None, m: Exe("amp") },
    Sig { agent: "OpenClaw", vendor: "OpenClaw", need: None, reject: None, m: Exe("openclaw") },
    Sig { agent: "Hermes", vendor: "Nous", need: None, reject: None, m: ExePrefix("hermes") },
    // omp and its fork names; `omp` is also a common word, so argv[0]/exe
    // basename only — never a substring match.
    Sig { agent: "omp", vendor: "omp", need: None, reject: None, m: Exe("omp") },
    // --- agent hosts / gateways ------------------------------------------
    Sig { agent: "Agent Client Protocol", vendor: "ACP host", need: None, reject: None, m: ArgToken("agent-client-protocol") },
    Sig { agent: "MCP server", vendor: "Model Context Protocol", need: None, reject: None, m: ArgContains("mcp-server") },
    Sig { agent: "MCP server", vendor: "Model Context Protocol", need: None, reject: None, m: ArgPrefix("mcp-") },
    // --- IDE-based agents -------------------------------------------------
    // Cursor: binary is `cursor`; also catch extension-host helpers.
    Sig { agent: "Cursor", vendor: "Anysphere", need: None, reject: Some("--type="), m: Exe("cursor") },
    // Windsurf (Codeium): binary `codeium` or Code - OSS fork paths.
    Sig { agent: "Windsurf", vendor: "Codeium", need: None, reject: Some("--type="), m: Exe("codeium") },
    Sig { agent: "Windsurf", vendor: "Codeium", need: None, reject: Some("--type="), m: Exe("windsurf") },
    // VS Code Copilot agent: `code` process whose extensions mention copilot,
    // or the standalone copilot-runtime / copilot-language-server binaries.
    Sig { agent: "Copilot", vendor: "GitHub/Microsoft", need: Some("copilot"), reject: Some("--type="), m: Exe("code") },
    // copilot-runtime / copilot-language-server binaries; reject the VS Code
    // main process (its argv mentions copilot via extensions) so it groups
    // with the `code` rule above instead of forming its own session.
    Sig { agent: "Copilot", vendor: "GitHub/Microsoft", need: None, reject: Some("--user-data-dir"), m: ExePrefix("copilot") },
    Sig { agent: "Copilot", vendor: "GitHub/Microsoft", need: Some("copilot"), reject: None, m: ArgContains("copilot") },
    // JetBrains AI Assistant / Junie: idea processes with the ai extension.
    Sig { agent: "JetBrains AI", vendor: "JetBrains", need: Some("aiassistant"), reject: Some("--type="), m: ExePrefix("idea") },
    Sig { agent: "JetBrains AI", vendor: "JetBrains", need: Some("junie"), reject: Some("--type="), m: ExePrefix("idea") },
    // Zed with agent panel: zed binary plus assistant token in args.
    Sig { agent: "Zed AI", vendor: "Zed", need: Some("assistant"), reject: None, m: Exe("zed") },
    // --- local model runtimes that serve agents ---------------------------
    // ollama serve is the runtime; `ollama run` is the client side of a
    // session, so any ollama process counts.
    Sig { agent: "ollama", vendor: "ollama", need: None, reject: None, m: Exe("ollama") },
    Sig { agent: "ollama", vendor: "ollama", need: None, reject: None, m: ArgToken("ollama") },
    Sig { agent: "llama.cpp server", vendor: "llama.cpp", need: None, reject: None, m: Exe("llama-server") },
    // vLLM runs as `python -m vllm...` or inside a venv: match any path
    // component or module prefix, not a bare argv token.
    Sig { agent: "vLLM", vendor: "vLLM", need: None, reject: None, m: ArgContains("vllm") },
];

/// Raw per-process hit before grouping.
struct Hit {
    pid: u32,
    ppid: u32,
    agent: &'static str,
    vendor: &'static str,
    cwd: Option<String>,
    cpu_ticks: u64,
    mem_bytes: u64,
    start_ticks: u64,
    cmdline: String,
}

/// State carried between scans: CPU-percent deltas and parsed transcripts
/// (reparsed only when the file changes).
#[derive(Default)]
pub struct AgentScanner {
    prev_cpu: HashMap<u32, u64>,
    prev_at: Option<Instant>,
    transcripts: HashMap<PathBuf, (u64, SystemTime, Activity)>,
}

fn clock_ticks() -> u64 {
    // KERNEL_TICKS is 100 on essentially every Linux port.
    100
}

/// Wall-clock seconds since the epoch, read from /proc/stat's boot time so
/// it agrees with the starttime field in /proc/[pid]/stat.
fn wall_now_secs() -> f64 {
    fs::read_to_string("/proc/stat")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("btime "))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|w| w.parse().ok())
        })
        .unwrap_or_else(|| {
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map(|d| d.as_secs() as f64)
                .unwrap_or(0.0)
        })
}

impl AgentScanner {
    /// Scan /proc once and return sessions, sorted by (agent, lead pid).
    pub fn scan(&mut self) -> Vec<AgentSession> {
        let btime = wall_now_secs();
        let ticks = clock_ticks();
        let now = Instant::now();
        let elapsed = self.prev_at.map(|t| now.duration_since(t)).unwrap_or(Duration::ZERO);

        let mut hits: Vec<Hit> = Vec::new();
        // every process's parent, so hits can be grouped by process tree
        let mut parents: HashMap<u32, u32> = HashMap::new();
        let dir = match fs::read_dir("/proc") {
            Ok(d) => d,
            Err(_) => return Vec::new(),
        };
        for entry in dir.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let Ok(pid): Result<u32, _> = name.parse() else { continue };

            let stat = match fs::read(format!("/proc/{pid}/stat")) {
                Ok(b) => String::from_utf8_lossy(&b).into_owned(),
                Err(_) => continue,
            };
            // comm is in parentheses and may contain spaces/parens: find the
            // LAST ')' then parse the fields after it.
            let Some(close) = stat.rfind(')') else { continue };
            let rest: Vec<&str> = stat[close + 1..].split_whitespace().collect();
            // rest[0]=state rest[1]=ppid ... rest[11]=utime rest[12]=stime
            // rest[19]=starttime (fields 14,15,22 overall)
            if rest.len() < 20 {
                continue;
            }
            let ppid: u32 = rest[1].parse().unwrap_or(0);
            parents.insert(pid, ppid);
            let utime: u64 = rest[11].parse().unwrap_or(0);
            let stime: u64 = rest[12].parse().unwrap_or(0);
            let start_ticks: u64 = rest[19].parse().unwrap_or(0);
            let cpu_ticks = utime + stime;

            let cmdline = fs::read(format!("/proc/{pid}/cmdline"))
                .ok()
                .map(|b| {
                    let mut parts: Vec<String> = b
                        .split(|c| *c == 0)
                        .filter(|s| !s.is_empty())
                        .map(|s| String::from_utf8_lossy(s).into_owned())
                        .collect();
                    if parts.is_empty() {
                        // kernel thread or unreadable: fall back to comm
                        parts.push(comm_from_stat(&stat));
                    }
                    parts
                })
                .unwrap_or_default();
            if cmdline.is_empty() {
                continue; // kernel thread
            }

            let exe_base = fs::read_link(format!("/proc/{pid}/exe"))
                .ok()
                .and_then(|p| p.file_name().map(|f| f.to_string_lossy().into_owned()));

            let cwd = fs::read_link(format!("/proc/{pid}/cwd"))
                .ok()
                .map(|p| p.to_string_lossy().into_owned());

            let Some(sig) = classify(exe_base.as_deref(), &cmdline) else { continue };

            let mem_bytes = read_rss_kb(pid).saturating_mul(1024);
            let joined = cmdline.join(" ");
            hits.push(Hit {
                pid,
                ppid,
                agent: sig.agent,
                vendor: sig.vendor,
                cwd,
                cpu_ticks,
                mem_bytes,
                start_ticks,
                cmdline: if joined.len() > 160 { joined.chars().take(157).collect::<String>() + "…" } else { joined },
            });
        }

        // CPU% over the scan interval.
        let dt = elapsed.as_secs_f64();
        let mut pct = HashMap::new();
        for h in &hits {
            let pct_v = match self.prev_cpu.get(&h.pid) {
                Some(&p) if dt > 0.0 && h.cpu_ticks >= p => {
                    (h.cpu_ticks - p) as f64 / ticks as f64 / dt * 100.0
                }
                _ => 0.0,
            };
            pct.insert(h.pid, pct_v);
        }
        self.prev_cpu = hits.iter().map(|h| (h.pid, h.cpu_ticks)).collect();
        self.prev_at = Some(now);

        // Group by process tree: a hit belongs to its topmost ancestor of
        // the same agent kind. Separate launches in one directory (say three
        // editor sessions on one repo) stay separate sessions.
        let agent_of: HashMap<u32, &'static str> = hits.iter().map(|h| (h.pid, h.agent)).collect();
        let mut groups: HashMap<u32, Vec<Hit>> = HashMap::new();
        for h in hits {
            let root = session_root(h.pid, h.ppid, h.agent, &parents, &agent_of);
            groups.entry(root).or_default().push(h);
        }

        let mut sessions: Vec<AgentSession> = groups
            .into_values()
            .map(|mut hs| {
                hs.sort_by_key(|h| h.pid);
                let lead = &hs[0];
                let started = SystemTime::UNIX_EPOCH
                    + Duration::from_secs_f64(btime + lead.start_ticks as f64 / ticks as f64);
                let cmdline = hs
                    .iter()
                    .map(|h| h.cmdline.as_str())
                    .min_by_key(|c| c.len())
                    .unwrap_or("")
                    .to_string();
                AgentSession {
                    agent: lead.agent,
                    vendor: lead.vendor,
                    cwd: lead.cwd.clone(),
                    lead_pid: lead.pid,
                    pids: hs.iter().map(|h| h.pid).collect(),
                    cpu_pct: hs.iter().map(|h| pct.get(&h.pid).copied().unwrap_or(0.0)).sum(),
                    mem_bytes: hs.iter().map(|h| h.mem_bytes).sum(),
                    started,
                    cmdline,
                    activity: None,
                }
            })
            .collect();
        let mut seen = Vec::new();
        for s in sessions.iter_mut().filter(|s| s.agent == "Claude Code") {
            s.activity = self.claude_activity(&s.pids, &mut seen);
        }
        self.transcripts.retain(|p, _| seen.contains(p));
        sessions.sort_by(|a, b| {
            a.agent.cmp(b.agent).then(a.lead_pid.cmp(&b.lead_pid))
        });
        sessions
    }

    /// Status and recent steps of the Claude Code session one of `pids`
    /// registered under `~/.claude/sessions`. `seen` collects transcript
    /// paths still in use, so the cache can drop the rest.
    fn claude_activity(&mut self, pids: &[u32], seen: &mut Vec<PathBuf>) -> Option<Activity> {
        let home = PathBuf::from(std::env::var_os("HOME")?).join(".claude");
        let meta: Value = pids
            .iter()
            .find_map(|pid| fs::read(home.join("sessions").join(format!("{pid}.json"))).ok())
            .and_then(|b| serde_json::from_slice(&b).ok())?;
        let str_of = |k: &str| meta.get(k).and_then(|v| v.as_str()).map(str::to_string);
        let cwd = str_of("cwd");
        let mut act = match (str_of("sessionId"), &cwd) {
            (Some(id), Some(dir)) => match transcript_path(&home.join("projects"), dir, &id) {
                Some(path) => {
                    let parsed = self.transcript(&path, dir);
                    seen.push(path);
                    parsed
                }
                None => Activity::default(),
            },
            _ => Activity::default(),
        };
        act.status = str_of("status").unwrap_or_else(|| "busy".into());
        act.status_since = meta
            .get("statusUpdatedAt")
            .or_else(|| meta.get("updatedAt"))
            .and_then(|v| v.as_i64())
            .map(|ms| ms / 1000);
        if act.title.is_none() {
            act.title = str_of("name");
        }
        Some(act)
    }

    /// Parsed transcript tail, cached until the file's size or mtime moves.
    fn transcript(&mut self, path: &Path, cwd: &str) -> Activity {
        let Ok(md) = fs::metadata(path) else { return Activity::default() };
        let (len, mtime) = (md.len(), md.modified().unwrap_or(SystemTime::UNIX_EPOCH));
        if let Some((l, m, act)) = self.transcripts.get(path) {
            if *l == len && *m == mtime {
                return act.clone();
            }
        }
        let act = read_tail(path, len).map(|t| parse_transcript(&t, Some(cwd))).unwrap_or_default();
        self.transcripts.insert(path.to_path_buf(), (len, mtime, act.clone()));
        act
    }
}

/// Topmost ancestor of `pid` (itself included) that is a hit of `agent`.
fn session_root(
    pid: u32,
    ppid: u32,
    agent: &str,
    parents: &HashMap<u32, u32>,
    agent_of: &HashMap<u32, &'static str>,
) -> u32 {
    let (mut root, mut cur) = (pid, ppid);
    // depth cap guards against a malformed (cyclic) parent table
    for _ in 0..64 {
        if cur <= 1 {
            break;
        }
        if agent_of.get(&cur) == Some(&agent) {
            root = cur;
        }
        match parents.get(&cur) {
            Some(&p) => cur = p,
            None => break,
        }
    }
    root
}

/// `~/.claude/projects/<cwd with every non-alphanumeric char as '-'>/<id>.jsonl`,
/// falling back to a search of every project dir for the session id.
fn transcript_path(projects: &Path, cwd: &str, id: &str) -> Option<PathBuf> {
    let file = format!("{id}.jsonl");
    let enc: String = cwd.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
    let direct = projects.join(enc).join(&file);
    if direct.is_file() {
        return Some(direct);
    }
    fs::read_dir(projects)
        .ok()?
        .flatten()
        .map(|d| d.path().join(&file))
        .find(|p| p.is_file())
}

/// Last [`TRANSCRIPT_TAIL`] bytes of the file, starting at a line boundary.
fn read_tail(path: &Path, len: u64) -> Option<String> {
    let mut f = fs::File::open(path).ok()?;
    let start = len.saturating_sub(TRANSCRIPT_TAIL);
    f.seek(SeekFrom::Start(start)).ok()?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).ok()?;
    let text = String::from_utf8_lossy(&buf).into_owned();
    Some(match (start, text.find('\n')) {
        (0, _) => text,
        (_, Some(nl)) => text[nl + 1..].to_string(),
        (_, None) => String::new(),
    })
}

/// Build an [`Activity`] (minus status) from Claude Code transcript lines.
pub fn parse_transcript(text: &str, cwd: Option<&str>) -> Activity {
    let mut act = Activity::default();
    let mut steps: Vec<Step> = Vec::new();
    // tool_use id -> index in `steps`, to attach results to their call
    let mut calls: HashMap<String, usize> = HashMap::new();
    for line in text.lines() {
        let Ok(d) = serde_json::from_str::<Value>(line) else { continue };
        let at = d.get("timestamp").and_then(|v| v.as_str()).and_then(parse_iso);
        if at.is_some() {
            act.last_active = at.max(act.last_active);
        }
        if let Some(b) = d.get("gitBranch").and_then(|v| v.as_str()).filter(|b| !b.is_empty()) {
            act.branch = Some(b.to_string());
        }
        // subagent and harness-injected entries aren't the session's own steps
        if d.get("isSidechain").and_then(|v| v.as_bool()) == Some(true)
            || d.get("isMeta").and_then(|v| v.as_bool()) == Some(true)
        {
            continue;
        }
        let msg = d.get("message");
        let content = msg.and_then(|m| m.get("content"));
        match d.get("type").and_then(|v| v.as_str()) {
            Some("ai-title") => act.title = d.get("aiTitle").and_then(|v| v.as_str()).map(one_line),
            Some("last-prompt") => {
                act.last_prompt = d.get("lastPrompt").and_then(|v| v.as_str()).map(one_line)
            }
            Some("user") => match content {
                Some(Value::String(t)) => push_prompt(&mut steps, at, t),
                Some(Value::Array(items)) => {
                    for it in items {
                        match it.get("type").and_then(|v| v.as_str()) {
                            Some("text") => {
                                push_prompt(&mut steps, at, it.get("text").and_then(|v| v.as_str()).unwrap_or(""))
                            }
                            Some("tool_result") => {
                                let id = it.get("tool_use_id").and_then(|v| v.as_str()).unwrap_or("");
                                if let Some(&i) = calls.get(id) {
                                    steps[i].pending = false;
                                    steps[i].failed = it.get("is_error").and_then(|v| v.as_bool()) == Some(true);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            },
            Some("assistant") => {
                let m = msg.unwrap_or(&Value::Null);
                if let Some(model) = m.get("model").and_then(|v| v.as_str()).filter(|s| !s.starts_with('<')) {
                    act.model = Some(model.to_string());
                }
                if let Some(u) = m.get("usage") {
                    let n = |k: &str| u.get(k).and_then(|v| v.as_u64()).unwrap_or(0);
                    let ctx = n("input_tokens") + n("cache_creation_input_tokens") + n("cache_read_input_tokens");
                    if ctx > 0 {
                        act.context_tokens = Some(ctx);
                    }
                }
                for it in content.and_then(|c| c.as_array()).into_iter().flatten() {
                    match it.get("type").and_then(|v| v.as_str()) {
                        Some("text") => {
                            let t = one_line(it.get("text").and_then(|v| v.as_str()).unwrap_or(""));
                            if !t.is_empty() {
                                steps.push(Step { at, kind: StepKind::Reply, tool: String::new(), text: t, pending: false, failed: false });
                            }
                        }
                        Some("tool_use") => {
                            let name = it.get("name").and_then(|v| v.as_str()).unwrap_or("tool");
                            let input = it.get("input").unwrap_or(&Value::Null);
                            if let Some(id) = it.get("id").and_then(|v| v.as_str()) {
                                calls.insert(id.to_string(), steps.len());
                            }
                            steps.push(Step {
                                at,
                                kind: StepKind::Tool,
                                tool: tool_label(name),
                                text: tool_summary(name, input, cwd),
                                pending: true,
                                failed: false,
                            });
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    if act.last_prompt.is_none() {
        act.last_prompt = steps.iter().rev().find(|s| s.kind == StepKind::Prompt).map(|s| s.text.clone());
    }
    let skip = steps.len().saturating_sub(MAX_STEPS);
    act.steps = steps.split_off(skip);
    act
}

/// Typed prompts become steps; harness wrappers (`<system-reminder>` and
/// friends) don't, except slash commands, shown as `/name`.
fn push_prompt(steps: &mut Vec<Step>, at: Option<i64>, raw: &str) {
    let t = raw.trim();
    let text = if let Some(rest) = t.strip_prefix("<command-name>") {
        rest.split("</command-name>").next().unwrap_or("").to_string()
    } else if t.starts_with('<') || t.is_empty() {
        return;
    } else {
        one_line(t)
    };
    steps.push(Step { at, kind: StepKind::Prompt, tool: String::new(), text, pending: false, failed: false });
}

/// "mcp__claude_ai_Atlassian__getJiraIssue" -> "Atlassian · getJiraIssue".
fn tool_label(name: &str) -> String {
    match name.strip_prefix("mcp__") {
        Some(rest) => {
            let mut parts = rest.rsplitn(2, "__");
            let tool = parts.next().unwrap_or(rest);
            let server = parts.next().unwrap_or("mcp");
            format!("{} · {tool}", server.trim_start_matches("claude_ai_"))
        }
        None => name.to_string(),
    }
}

/// The one input field that says what a tool call does.
fn tool_summary(name: &str, input: &Value, cwd: Option<&str>) -> String {
    let s = |k: &str| input.get(k).and_then(|v| v.as_str());
    let rel = |p: &str| match cwd.and_then(|c| p.strip_prefix(c)) {
        Some(r) if r.starts_with('/') => r[1..].to_string(),
        _ => p.to_string(),
    };
    let text = match name {
        "Bash" => s("description").or(s("command")).map(str::to_string),
        "Read" | "Edit" | "Write" | "MultiEdit" => s("file_path").map(rel),
        "NotebookEdit" => s("notebook_path").map(rel),
        "Grep" | "Glob" => s("pattern").map(str::to_string),
        "WebFetch" => s("url").map(str::to_string),
        "WebSearch" => s("query").map(str::to_string),
        "Agent" | "Task" => s("description").map(str::to_string),
        "Skill" => s("skill").map(str::to_string),
        _ => None,
    };
    // unknown tools: the first string field is usually the interesting one
    let text = text.or_else(|| input.as_object()?.values().find_map(|v| v.as_str()).map(str::to_string));
    one_line(&text.unwrap_or_default())
}

/// First non-empty line, capped at 200 chars.
fn one_line(s: &str) -> String {
    let line = s.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("");
    if line.chars().count() > 200 {
        line.chars().take(199).collect::<String>() + "…"
    } else {
        line.to_string()
    }
}

/// "2026-10-22T10:19:44.741Z" -> epoch seconds (UTC only, as Claude writes).
fn parse_iso(s: &str) -> Option<i64> {
    let n = |r: std::ops::Range<usize>| s.get(r)?.parse::<i64>().ok();
    let (y, mo, d) = (n(0..4)?, n(5..7)?, n(8..10)?);
    let (h, mi, sec) = (n(11..13)?, n(14..16)?, n(17..19)?);
    // days from civil (Howard Hinnant)
    let y = if mo <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if mo > 2 { mo - 3 } else { mo + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    Some(days * 86_400 + h * 3600 + mi * 60 + sec)
}

fn comm_from_stat(stat: &str) -> String {
    match (stat.find('('), stat.rfind(')')) {
        (Some(o), Some(c)) if c > o => stat[o + 1..c].to_string(),
        _ => String::new(),
    }
}

fn read_rss_kb(pid: u32) -> u64 {
    fs::read_to_string(format!("/proc/{pid}/statm"))
        .ok()
        .and_then(|s| {
            // statm fields are in pages: size resident shared ...
            s.split_whitespace().nth(1).and_then(|w| w.parse::<u64>().ok())
        })
        .map(|pages| pages * (page_size_kb()))
        .unwrap_or(0)
}

fn page_size_kb() -> u64 {
    // statm counts RSS in pages; 4 KiB is the page size on every common
    // Linux port we target.
    4
}

/// Classify one process; first matching signature wins.
fn classify<'a>(exe_base: Option<&str>, cmdline: &[String]) -> Option<&'a Sig> {
    let joined = cmdline.join(" ").to_lowercase();
    let argv0_base = Path::new(cmdline.get(0).map(|s| s.as_str()).unwrap_or(""))
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tokens: Vec<String> = cmdline.iter().map(|a| a.to_lowercase()).collect();

    for sig in SIGNATURES {
        let hit = match &sig.m {
            Exe(e) => exe_base == Some(e) || argv0_base == *e,
            ExePrefix(e) => exe_base
                .map(|b| b.starts_with(e))
                .unwrap_or(false)
                || argv0_base.starts_with(e),
            ArgToken(t) => tokens.iter().any(|a| a == t),
            ArgContains(t) => joined.contains(t),
            ArgPrefix(t) => tokens.iter().any(|a| a.starts_with(t)),
        };
        if !hit {
            continue;
        }
        if let Some(n) = sig.need {
            if !joined.contains(n) {
                continue;
            }
        }
        if let Some(r) = sig.reject {
            if joined.contains(r) {
                continue;
            }
        }
        return Some(sig);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cl(s: &str) -> Vec<String> {
        s.split_whitespace().map(|w| w.to_string()).collect()
    }

    #[test]
    fn detects_claude_code() {
        let c = cl("/home/u/.local/bin/claude --output-format stream-json --resume=abc");
        assert_eq!(classify(Some("claude"), &c).unwrap().agent, "Claude Code");
    }

    #[test]
    fn detects_codex_and_cursor_and_omp() {
        assert_eq!(classify(Some("codex"), &cl("codex --model gpt-5")).unwrap().agent, "Codex CLI");
        assert_eq!(classify(Some("cursor"), &cl("/usr/bin/cursor --user-data-dir=x")).unwrap().agent, "Cursor");
        assert_eq!(classify(Some("omp"), &cl("omp tui")).unwrap().agent, "omp");
    }

    #[test]
    fn detects_copilot_only_with_copilot_context() {
        // bare `code` is just the editor: no agent
        assert!(classify(Some("code"), &cl("/usr/share/code/code --user-data-dir=x")).is_none());
        let c = cl("/usr/share/code/code --extensionDevelopmentPath=anthropic.copilot-chat");
        assert_eq!(classify(Some("code"), &c).unwrap().agent, "Copilot");
    }

    #[test]
    fn rejects_electron_helpers_and_geoclue() {
        // ChatGPT/Codex desktop helper processes: skipped via --type=
        let helper = cl("/usr/lib/chatgpt/ChatGPT --type=renderer --user-data-dir=/home/u/.config/Codex");
        assert!(classify(Some("ChatGPT"), &helper).is_none());
        // geoclue's binary is literally named "agent": must NOT match
        assert!(classify(Some("agent"), &cl("/usr/libexec/geoclue-2.0/demos/agent")).is_none());
        // crashpad handlers must not match anything
        assert!(classify(Some("chrome_crashpad"), &cl("/usr/share/code/chrome_crashpad_handler --monitor-self")).is_none());
    }

    #[test]
    fn detects_mcp_servers_and_local_models() {
        let c = cl("node /usr/lib/node_modules/mcp-server-filesystem/index.js");
        assert_eq!(classify(Some("node"), &c).unwrap().agent, "MCP server");
        assert_eq!(classify(Some("ollama"), &cl("ollama serve")).unwrap().agent, "ollama");
        let c = cl("python3 -m vllm.entrypoints.openai.api_server");
        assert_eq!(classify(Some("python3"), &c).unwrap().agent, "vLLM");
    }

    #[test]
    fn omp_word_in_args_does_not_match() {
        // a process merely mentioning "omp" in an argument is not omp
        assert!(classify(Some("bash"), &cl("bash -c echo omp")).is_none());
    }

    #[test]
    fn parses_iso_timestamps() {
        assert_eq!(parse_iso("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_iso("2026-10-22T10:19:44.741Z"), Some(1_792_664_384));
        assert_eq!(parse_iso("garbage"), None);
    }

    #[test]
    fn groups_by_process_tree_not_directory() {
        // 10 and 20 are two claude launches in one repo; 11 is 10's child
        // via a non-agent shell (15)
        let parents: HashMap<u32, u32> = [(10, 5), (15, 10), (11, 15), (20, 5)].into_iter().collect();
        let agent_of: HashMap<u32, &'static str> =
            [(10, "Claude Code"), (11, "Claude Code"), (20, "Claude Code")].into_iter().collect();
        assert_eq!(session_root(11, 15, "Claude Code", &parents, &agent_of), 10);
        assert_eq!(session_root(10, 5, "Claude Code", &parents, &agent_of), 10);
        assert_eq!(session_root(20, 5, "Claude Code", &parents, &agent_of), 20);
        // a different agent under claude is its own session
        assert_eq!(session_root(11, 15, "MCP server", &parents, &agent_of), 11);
    }

    #[test]
    fn transcript_yields_title_prompt_and_steps() {
        let lines = [
            r#"{"type":"user","timestamp":"2026-10-22T10:00:00Z","gitBranch":"main","message":{"role":"user","content":"fix the login bug"}}"#,
            r#"{"type":"user","isMeta":true,"message":{"content":"<system-reminder>x</system-reminder>"}}"#,
            r#"{"type":"assistant","timestamp":"2026-10-22T10:00:05Z","message":{"model":"claude-opus-5-5","usage":{"input_tokens":10,"cache_read_input_tokens":1000,"cache_creation_input_tokens":90},"content":[{"type":"text","text":"Looking at the auth module.\nmore"},{"type":"tool_use","id":"t1","name":"Read","input":{"file_path":"/repo/src/auth.rs"}},{"type":"tool_use","id":"t2","name":"Bash","input":{"command":"cargo test","description":"Run tests"}}]}}"#,
            r#"{"type":"user","timestamp":"2026-10-22T10:00:09Z","message":{"content":[{"type":"tool_result","tool_use_id":"t1","content":"..."}, {"type":"tool_result","tool_use_id":"t2","is_error":true,"content":"fail"}]}}"#,
            r#"{"type":"assistant","timestamp":"2026-10-22T10:00:10Z","message":{"content":[{"type":"tool_use","id":"t3","name":"mcp__claude_ai_Atlassian__getJiraIssue","input":{"issueKey":"AB-1"}}]}}"#,
            r#"{"type":"ai-title","aiTitle":"Fix login bug"}"#,
            r#"{"type":"user","isSidechain":true,"message":{"content":"subagent chatter"}}"#,
            "not json",
        ]
        .join("\n");
        let a = parse_transcript(&lines, Some("/repo"));
        assert_eq!(a.title.as_deref(), Some("Fix login bug"));
        assert_eq!(a.last_prompt.as_deref(), Some("fix the login bug"));
        assert_eq!(a.branch.as_deref(), Some("main"));
        assert_eq!(a.model.as_deref(), Some("claude-opus-5-5"));
        assert_eq!(a.context_tokens, Some(1100));
        assert_eq!(a.last_active, parse_iso("2026-10-22T10:00:10Z"));
        let st: Vec<(StepKind, &str, &str, bool, bool)> =
            a.steps.iter().map(|s| (s.kind, s.tool.as_str(), s.text.as_str(), s.pending, s.failed)).collect();
        assert_eq!(
            st,
            vec![
                (StepKind::Prompt, "", "fix the login bug", false, false),
                (StepKind::Reply, "", "Looking at the auth module.", false, false),
                (StepKind::Tool, "Read", "src/auth.rs", false, false),
                (StepKind::Tool, "Bash", "Run tests", false, true),
                (StepKind::Tool, "Atlassian · getJiraIssue", "AB-1", true, false),
            ]
        );
    }

    #[test]
    fn scan_finds_this_test_or_returns_cleanly() {
        // Just exercise the real /proc path: must not panic, and must not
        // report the test binary itself as an agent.
        let mut s = AgentScanner::default();
        let sessions = s.scan();
        let me = std::process::id();
        assert!(!sessions.iter().any(|x| x.pids.contains(&me)));
    }
}
