//! Detection of running AI coding agents via /proc.
//!
//! Every scan reads each process's executable path, command line and cwd,
//! then matches against a table of known agent signatures (CLI agents, IDE
//! extensions, agent hosts like omp). Raw hits are grouped into sessions:
//! one agent launch usually spawns a small process tree sharing an agent
//! kind and a working directory, and the user thinks of that as one session.

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime};

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
}

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
    agent: &'static str,
    vendor: &'static str,
    cwd: Option<String>,
    cpu_ticks: u64,
    mem_bytes: u64,
    start_ticks: u64,
    cmdline: String,
}

/// State carried between scans for CPU-percent deltas.
#[derive(Default)]
pub struct AgentScanner {
    prev_cpu: HashMap<u32, u64>,
    prev_at: Option<Instant>,
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

        // Group by (agent, cwd).
        let mut groups: HashMap<(&'static str, Option<String>), Vec<Hit>> = HashMap::new();
        for h in hits {
            let key = (h.agent, h.cwd.clone());
            groups.entry(key).or_default().push(h);
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
                }
            })
            .collect();
        sessions.sort_by(|a, b| {
            a.agent.cmp(b.agent).then(a.lead_pid.cmp(&b.lead_pid))
        });
        sessions
    }
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
    fn scan_finds_this_test_or_returns_cleanly() {
        // Just exercise the real /proc path: must not panic, and must not
        // report the test binary itself as an agent.
        let mut s = AgentScanner::default();
        let sessions = s.scan();
        let me = std::process::id();
        assert!(!sessions.iter().any(|x| x.pids.contains(&me)));
    }
}
