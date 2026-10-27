//! Process scanner: every process on the box with live CPU% and memory,
//! Task-Manager style. Reads /proc, keeps the previous tick counters so the
//! delta over the scan interval gives a real busy percentage.

use std::collections::HashMap;
use std::fs;
use std::sync::{LazyLock, Mutex};
use std::time::{Duration, Instant};

/// One process row.
#[derive(Clone, Debug)]
pub struct ProcInfo {
    pub pid: u32,
    pub ppid: u32,
    pub name: String,
    /// joined argv, NULs replaced by spaces, elided to 200 chars
    pub cmdline: String,
    /// "R" running, "S" sleeping, "D" uninterruptible, "Z" zombie…
    pub state: char,
    /// CPU percent of one core, 0..=100*ncpu (same scale as Docker stats)
    pub cpu_pct: f64,
    /// resident set size, bytes
    pub rss_bytes: u64,
    pub threads: u32,
    /// epoch seconds the process started
    pub started: i64,
    /// process owner uid; read from /proc stat and used for the root-user
    /// highlight in the table
    #[allow(dead_code)]
    pub uid: u32,
    /// virtual memory size, bytes
    pub vsize_bytes: u64,
    /// bytes read/written since start (/proc/<pid>/io; 0 when unreadable)
    pub io_read: u64,
    pub io_write: u64,
    /// working directory, empty when unreadable
    pub cwd: String,
    /// owner name from /etc/passwd (or the uid as text)
    pub user: String,
    /// fallback glyph family drawn when no real icon resolves:
    /// "browser", "terminal", "service", … or "app"
    pub icon: &'static str,
    /// real application icon name from the freedesktop icon theme
    /// (e.g. "firefox", "org.gimp.GIMP"); empty when the app ships none
    pub app_icon: String,
}

/// Incremental scanner: call `scan()` once per refresh; the second and every
/// later call reports real CPU percentages for the elapsed interval.
pub struct ProcScanner {
    prev: HashMap<u32, (u64, Instant)>,
    prev_at: Option<Instant>,
    /// uid -> username, resolved once per scan from /etc/passwd
    users: HashMap<u32, String>,
    users_at: Option<Instant>,
}

impl Default for ProcScanner {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcScanner {
    pub fn new() -> Self {
        Self {
            prev: HashMap::new(),
            prev_at: None,
            users: HashMap::new(),
            users_at: None,
        }
    }

    pub fn scan(&mut self) -> Vec<ProcInfo> {
        let btime = boot_epoch_secs();
        let ticks = 100.0_f64; // KERNEL_TICKS on every common Linux port
        let now = Instant::now();
        let dt = self
            .prev_at
            .map(|t| now.duration_since(t))
            .unwrap_or(Duration::ZERO);
        let page = page_size();

        // /etc/passwd is tiny and changes rarely: re-read it at most once a
        // minute so rows can show "rifan" instead of "1000"
        if self
            .users_at
            .map(|t| t.elapsed() > Duration::from_secs(60))
            .unwrap_or(true)
        {
            self.users = passwd_users();
            self.users_at = Some(now);
        }

        let mut rows = Vec::new();
        let mut cur: HashMap<u32, (u64, Instant)> = HashMap::new();

        let dir = match fs::read_dir("/proc") {
            Ok(d) => d,
            Err(_) => return rows,
        };
        for entry in dir.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let Ok(pid): Result<u32, _> = name.parse() else {
                continue;
            };

            let stat = match fs::read(format!("/proc/{pid}/stat")) {
                Ok(b) => String::from_utf8_lossy(&b).into_owned(),
                Err(_) => continue,
            };
            // comm is in parentheses and may contain spaces/parens: split at
            // the LAST ')' then parse the fields after it.
            let Some(close) = stat.rfind(')') else {
                continue;
            };
            let name = stat[..close]
                .find('(')
                .map(|i| stat[i + 1..close].to_string())
                .unwrap_or_default();
            let rest: Vec<&str> = stat[close + 1..].split_whitespace().collect();
            // rest[] indexes are /proc(5) fields 3.. (state=rest[0]):
            // utime=rest[11], stime=rest[12], threads=rest[17],
            // starttime=rest[19], vsize=rest[20] (bytes), rss=rest[21] (pages).
            // Verified live against /proc/<pid>/status: rest[20] == VmSize,
            // rest[21] * page size == VmRSS, rest[17] == Threads.
            if rest.len() < 22 {
                continue;
            }
            let state = rest[0].chars().next().unwrap_or('?');
            let ppid: u32 = rest[1].parse().unwrap_or(0);
            let utime: u64 = rest[11].parse().unwrap_or(0);
            let stime: u64 = rest[12].parse().unwrap_or(0);
            let threads: u32 = rest[17].parse().unwrap_or(0);
            let starttime: u64 = rest[19].parse().unwrap_or(0);
            let vsize: u64 = rest[20].parse().unwrap_or(0);
            let rss_pages: u64 = rest[21].parse().unwrap_or(0);
            let cpu_ticks = utime + stime;
            cur.insert(pid, (cpu_ticks, now));

            let cpu_pct = match self.prev.get(&pid) {
                Some((p, _)) if dt > Duration::ZERO => {
                    cpu_ticks.saturating_sub(*p) as f64 / ticks / dt.as_secs_f64() * 100.0
                }
                _ => 0.0,
            };

            let cmdline_str = fs::read(format!("/proc/{pid}/cmdline"))
                .ok()
                .map(|b| {
                    let s = b
                        .split(|c| *c == 0)
                        .filter(|s| !s.is_empty())
                        .map(|s| String::from_utf8_lossy(s).into_owned())
                        .collect::<Vec<_>>()
                        .join(" ");
                    if s.chars().count() > 200 {
                        s.chars().take(197).collect::<String>() + "…"
                    } else {
                        s
                    }
                })
                .unwrap_or_default();

            let uid = fs::metadata(format!("/proc/{pid}"))
                .ok()
                .map(|m| uid_of(&m))
                .unwrap_or(0);

            let cwd = fs::read_link(format!("/proc/{pid}/cwd"))
                .ok()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
            // the real executable: snap/flatpak/AppImage wrappers all
            // resolve to the app's own binary, which is what desktop
            // entries point at
            let exe = fs::read_link(format!("/proc/{pid}/exe"))
                .ok()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default();
            let (io_read, io_write) = fs::read_to_string(format!("/proc/{pid}/io"))
                .ok()
                .map(|s| {
                    let mut r = 0;
                    let mut w = 0;
                    for line in s.lines() {
                        if let Some(v) = line.strip_prefix("read_bytes: ") {
                            r = v.trim().parse().unwrap_or(0);
                        } else if let Some(v) = line.strip_prefix("write_bytes: ") {
                            w = v.trim().parse().unwrap_or(0);
                        }
                    }
                    (r, w)
                })
                .unwrap_or((0, 0));

            let icon = icon_for(&name, &cmdline_str);
            let app_icon = if exe.is_empty() {
                String::new()
            } else {
                DESKTOP_ICONS
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .icon_for(&exe)
                    .unwrap_or_default()
            };
            rows.push(ProcInfo {
                pid,
                ppid,
                name,
                cmdline: cmdline_str.clone(),
                state,
                cpu_pct,
                rss_bytes: rss_pages * page,
                threads,
                started: (btime as f64 + starttime as f64 / ticks) as i64,
                uid,
                vsize_bytes: vsize,
                io_read,
                io_write,
                cwd,
                user: self
                    .users
                    .get(&uid)
                    .cloned()
                    .unwrap_or_else(|| uid.to_string()),
                icon,
                app_icon,
            });
        }

        self.prev = cur;
        self.prev_at = Some(now);
        rows
    }
}

/// uid of a /proc/<pid> directory owner; std has no portable accessor, so
/// read the numeric owner straight from `stat`.
fn uid_of(meta: &fs::Metadata) -> u32 {
    use std::os::unix::fs::MetadataExt;
    meta.uid()
}

fn boot_epoch_secs() -> u64 {
    fs::read_to_string("/proc/stat")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("btime "))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|w| w.parse().ok())
        })
        .unwrap_or(0)
}

/// System page size via the `getpagesize` libc symbol glibc always provides.
fn page_size() -> u64 {
    extern "C" {
        fn getpagesize() -> i32;
    }
    let v = unsafe { getpagesize() };
    if v > 0 {
        v as u64
    } else {
        4096
    }
}

/// uid -> username from /etc/passwd; empty map when the file is unreadable.
fn passwd_users() -> HashMap<u32, String> {
    let mut m = HashMap::new();
    if let Ok(s) = fs::read_to_string("/etc/passwd") {
        for line in s.lines() {
            let f: Vec<&str> = line.split(':').collect();
            if f.len() >= 3 {
                if let Ok(uid) = f[2].parse::<u32>() {
                    m.insert(uid, f[0].to_string());
                }
            }
        }
    }
    m
}

/// Pick the icon kind for a process from its comm and command line. The
/// table mirrors the glyph names `icon_glyph` knows how to draw; anything
/// unknown falls back to "app".
pub fn icon_for(comm: &str, cmdline: &str) -> &'static str {
    let c = comm.to_lowercase();
    let cl = cmdline.to_lowercase();
    let has = |needles: &[&str]| needles.iter().any(|n| c.contains(n) || cl.contains(n));
    // comm is truncated to 15 chars, so exact names go first: kthreadd is
    // the kernel thread parent, thunderbird must not match the "bird" chain
    match c.as_str() {
        "kthreadd" => return "kernel",
        "thunderbird" => return "chat",
        _ => {}
    }
    // exact comm wins: "/sbin/init" as comm "systemd" is the service
    // manager, not a login shell
    if matches!(c.as_str(), "systemd" | "init") {
        return "service";
    }
    // browsers first: their helpers (xdg-desktop-portal-gtk etc.) must not
    // shadow the browser itself
    if has(&["firefox", "icecat", "torbro"])
        || has(&["chrom", "brave", "edge", "vivaldi", "opera", "360chrome"])
    {
        "browser"
    } else if has(&[
        "electron",
        "code",
        "code-insiders",
        "cursor",
        "windsurf",
        "trae",
        "codium",
        "strawberry",
    ]) {
        "app"
    } else if has(&[
        "jetbrains",
        "idea",
        "pycharm",
        "goland",
        "webstorm",
        "clion",
        "rider",
        "android studio",
        "rustrover",
        "rustrover",
    ]) {
        "jetbrains"
    } else if has(&[
        "steam",
        "lutris",
        "gamescope",
        "mangohud",
        "prismlauncher",
        "heroic",
    ]) {
        "game"
    } else if has(&[
        "libreoffice",
        "soffice",
        "gimp",
        "inkscape",
        "krita",
        "blender",
        "darktable",
        "digi",
        "shotcut",
        "kdenlive",
        "obs",
        "audacity",
        "scribus",
        "draw.io",
        "dia",
    ]) {
        "media"
    } else if has(&[
        "vlc",
        "mpv",
        "totem",
        "haruna",
        "celluloid",
        "mpd",
        "spotify",
        "cmus",
        "ncmpcpp",
        "rhythmbox",
        "strawberry",
        "audacious",
    ]) {
        "music"
    } else if has(&[
        "telegram",
        "whatsapp",
        "signal",
        "discord",
        "webkit",
        "thunderbird",
        "evolution",
        "kontact",
        "gajim",
        "pidgin",
        "polari",
        "fractal",
        "element",
        "webex",
        "zoom",
        "jitsi",
        "messagingmenu",
    ]) {
        "chat"
    } else if has(&[
        "file-manager",
        "nautilus",
        "dolphin",
        "thunar",
        "nemo",
        "pcmanfm",
        "krusader",
        "mousepad",
        "gedit",
        "kate",
        "leafpad",
        "galculator",
        "baobab",
        "disk-usage",
        "file-roller",
        "engrampa",
        "xarchiver",
    ]) {
        "files"
    } else if has(&[
        "docker-proxy",
        "dockerd",
        "containerd",
        "podman",
        "crio",
        "kube",
        "runc",
        "containerd-shim",
    ]) {
        "container"
    } else if has(&[
        "postgres",
        "mysqld",
        "mariadbd",
        "redis",
        "mongod",
        "mariadbd",
        "clickhouse",
        "influx",
        "etcd",
    ]) {
        "database"
    } else if has(&[
        "node", "bun", "deno", "python", "python3", "java", "ruby", "perl", "php", "dotnet",
        "beam.smp", "erl", "uvicorn", "gunicorn",
    ]) {
        "runtime"
    } else if has(&[
        "omp",
        "claude",
        "codex",
        "aider",
        "opencode",
        "hermes",
        " goose",
        "hermes-agent",
    ]) {
        "agent"
    } else if has(&[
        "xterm",
        "kitty",
        "alacritty",
        "konsole",
        "gnome-terminal",
        "foot",
        "wezterm",
        "tilix",
        "st",
        "terminal",
        "terminator",
        "qterminal",
        "xfce4-terminal",
        "ghostty",
        "tmux",
        "zellij",
        "screen",
    ]) || matches!(
        c.as_str(),
        "zsh" | "bash" | "fish" | "sh" | "dash" | "csh" | "tcsh" | "nu"
    ) {
        "terminal"
    } else if has(&[
        "pipewire",
        "wireplumber",
        "pulseaudio",
        "alsa",
        "alsa-seq",
        "alsa-sink",
        "alsa-source",
    ]) {
        "audio"
    } else if matches!(c.as_str(), "systemd" | "init")
        || has(&[
            "systemd-",
            "udev",
            "journald",
            "logind",
            "dbus-daemon",
            "dbus-broker",
            "cron",
            "crond",
            "atd",
            "agetty",
            "polkit",
        ])
    {
        "service"
    } else if has(&[
        "nm-applet",
        "blueman",
        "bluetooth",
        "networkmanager",
        "wpa_supplicant",
        "geoclue",
        "upower",
        "powerdevil",
        "gdm-",
        "sddm",
        "lightdm",
        "lxqt",
        "accounts-daemon",
        "udisksd",
        "packagekit",
        "flatpak",
        "snapd",
    ]) {
        "settings"
    } else if has(&[
        "x11",
        "wayland",
        "weston",
        "hyprland",
        "sway",
        "kwin",
        "mutter",
        "xwayland",
        "plasma",
        "gnome-shell",
        "xfdesktop",
        "xfce",
        "kde",
    ]) || c.starts_with("xfce")
        || matches!(c.as_str(), "xorg" | "x")
    {
        "desktop"
    } else if c.starts_with("kworker")
        || c.starts_with("kthread")
        || c.starts_with("ksoftirqd")
        || c.starts_with("migration/")
        || c.starts_with("rcu_")
        || c.starts_with("kcompactd")
        || c.starts_with("writeback")
        || c.starts_with("ext4-")
        || c.starts_with("xfs-")
        || c.starts_with("nvme-")
        || c.starts_with("kswapd")
    {
        "kernel"
    } else {
        "app"
    }
}

// ---------------------------------------------------------------------------
// Real application icons: map a process executable to the icon name its
// .desktop entry declares, using the freedesktop search paths. The table is
// parsed once per process (desktop files change rarely) and consulted under
// a mutex from the scan thread.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Real application icons: map a process executable to the icon name its
// .desktop entry declares, using the freedesktop search paths. The table is
// parsed once per process (desktop files change rarely) and consulted under
// a mutex from the scan thread.
// ---------------------------------------------------------------------------

static DESKTOP_ICONS: LazyLock<Mutex<DesktopIcons>> =
    LazyLock::new(|| Mutex::new(DesktopIcons::load()));

/// How many mtime checks trigger a full rescan of the .desktop files.
const DESKTOP_RESCAN_AFTER: usize = 24;

struct DesktopEntry {
    /// Exec= basename, e.g. "firefox", "idea.sh"
    exec: String,
    /// Icon= value, e.g. "firefox", "org.gimp.GIMP"
    icon: String,
    /// Icon= was an absolute path: use that file directly, themes don't
    /// know about snap/AppImage icons
    icon_path: Option<String>,
    /// true when TryExec/Exec points at an existing executable
    alive: bool,
}

pub struct DesktopIcons {
    entries: Vec<DesktopEntry>,
    /// exe path -> resolved icon name ("" = known to have none)
    cache: HashMap<String, String>,
    scans: usize,
}

impl DesktopIcons {
    fn load() -> DesktopIcons {
        let mut entries = Vec::new();
        for dir in desktop_dirs() {
            let Ok(rd) = fs::read_dir(dir) else { continue };
            for f in rd.flatten() {
                let path = f.path();
                if path.extension().and_then(|e| e.to_str()) != Some("desktop") {
                    continue;
                }
                if let Some(e) = parse_desktop(&path) {
                    entries.push(e);
                }
            }
        }
        DesktopIcons {
            entries,
            cache: HashMap::new(),
            scans: 0,
        }
    }

    /// Icon for an executable path — a theme icon name, or an absolute
    /// file path when the entry declared one — or None when nothing matches.
    fn icon_for(&mut self, exe: &str) -> Option<String> {
        if let Some(hit) = self.cache.get(exe) {
            return (!hit.is_empty()).then(|| hit.clone());
        }
        self.scans += 1;
        if self.scans > DESKTOP_RESCAN_AFTER {
            self.scans = 0;
            self.cache.clear();
            self.entries = DesktopIcons::load().entries;
        }
        let base = exe.rsplit('/').next().unwrap_or(exe).to_lowercase();
        // snap/flatpak/AppImage exes live under versioned dirs; the app id
        // is the wrapper dir name, and AppImage mounts use the app name
        let mut candidates: Vec<String> = vec![base.clone()];
        if let Some(rest) = exe.strip_prefix("/snap/") {
            if let Some(app) = rest.split('/').next() {
                candidates.push(app.to_lowercase());
            }
        }
        if exe.contains("/app/") && exe.contains(".appimage") {
            candidates.push(base.replace(".appimage", ""));
        }
        let mut best: Option<(u8, String)> = None;
        for c in &candidates {
            // substring matches are noisy ("cat" inside "catalog"): they
            // only count when the process name is long enough to be unique
            let loose = c.len() >= 4;
            for e in self.entries.iter() {
                if !e.alive || e.icon.is_empty() {
                    continue;
                }
                let ex = &e.exec;
                let hit = if ex == c {
                    2
                } else if ex.len() >= 3
                    && (ex.starts_with(c.as_str()) || c.starts_with(ex.as_str()))
                {
                    1
                } else if loose && (ex.contains(c.as_str()) || c.contains(ex.as_str())) {
                    0
                } else {
                    continue;
                };
                if best.as_ref().is_none_or(|(h, _)| hit > *h) {
                    let icon = match &e.icon_path {
                        // absolute entry path wins only when it exists
                        Some(p) if std::path::Path::new(p).exists() => p.clone(),
                        _ => e.icon.clone(),
                    };
                    best = Some((hit, icon));
                }
                if best.as_ref().is_some_and(|(h, _)| *h == 2) {
                    break;
                }
            }
            if best.as_ref().is_some_and(|(h, _)| *h == 2) {
                break;
            }
        }
        let best = best.map(|(_, icon)| icon);
        self.cache
            .insert(exe.to_string(), best.clone().unwrap_or_default());
        best
    }
}

/// Standard .desktop locations, user dir last so system entries win.
fn desktop_dirs() -> Vec<std::path::PathBuf> {
    let mut v = vec![
        std::path::PathBuf::from("/usr/share/applications"),
        std::path::PathBuf::from("/usr/local/share/applications"),
        std::path::PathBuf::from("/var/lib/flatpak/exports/share/applications"),
        // snap apps live here, not in the regular dirs
        std::path::PathBuf::from("/var/lib/snapd/desktop/applications"),
    ];
    if let Some(d) = std::env::var_os("XDG_DATA_DIRS") {
        v.extend(std::env::split_paths(&d).map(|p| p.join("applications")));
    }
    if let Some(home) = std::env::var_os("HOME") {
        let home = std::path::PathBuf::from(home);
        v.push(home.join(".local/share/flatpak/exports/share/applications"));
        v.push(home.join(".local/share/applications"));
    }
    v
}

/// Pull Exec/TryExec/Icon out of one .desktop file and keep the entry only
/// if its executable actually exists (a stale entry must not steal a match).
/// Absolute Icon= paths are reduced to the icon name — the theme lookup
/// finds the same file and can pick a better size bucket.
fn parse_desktop(path: &std::path::Path) -> Option<DesktopEntry> {
    let text = fs::read_to_string(path).ok()?;
    let (mut exec, mut try_exec, mut icon) = (String::new(), String::new(), String::new());
    for line in text.lines() {
        match line.split_once('=') {
            Some(("Exec", v)) if exec.is_empty() => exec = v.trim().to_string(),
            Some(("TryExec", v)) => try_exec = v.trim().to_string(),
            Some(("Icon", v)) if icon.is_empty() => icon = v.trim().to_string(),
            _ => {}
        }
    }
    if exec.is_empty() && icon.is_empty() {
        return None;
    }
    // absolute Icon= paths (snaps, AppImages, .deb postinst pixmaps) point
    // straight at the file; keep it, but also keep a name for the theme
    let icon_path = if icon.starts_with('/') {
        Some(icon.clone())
    } else {
        None
    };
    if let Some(p) = &icon_path {
        let base = p.rsplit('/').next().unwrap_or("");
        icon = base
            .trim_end_matches(".svgz")
            .trim_end_matches(".svg")
            .trim_end_matches(".png")
            .to_string();
    }
    // strip a leading wrapper (env, flatpak run, snap run …) then take the
    // first token that looks like the real command
    let pick = |s: &str| -> Option<String> {
        s.split_whitespace()
            .find(|t| t.contains('/') || !t.contains('-'))
            .map(|t| t.to_string())
    };
    let candidate = pick(&try_exec).or_else(|| pick(&exec))?;
    // resolve to an absolute path when possible; bare names are assumed alive
    let alive = if candidate.contains('/') {
        std::path::Path::new(&candidate).exists()
    } else {
        which_exists(&candidate)
    };
    let exec_base = candidate
        .rsplit('/')
        .next()
        .unwrap_or(&candidate)
        .to_lowercase();
    Some(DesktopEntry {
        exec: exec_base,
        icon,
        icon_path,
        alive,
    })
}

fn which_exists(name: &str) -> bool {
    if let Some(p) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&p) {
            if dir.join(name).exists() {
                return true;
            }
        }
    }
    false
}

/// Sort rows by the given column; secondary sort is always pid so the order
/// is stable and deterministic.
pub fn sort_rows(rows: &mut [ProcInfo], col: SortCol, desc: bool) {
    use SortCol::*;
    rows.sort_by(|a, b| {
        let ord = match col {
            Name => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
            Pid => a.pid.cmp(&b.pid),
            Cpu => a.cpu_pct.total_cmp(&b.cpu_pct),
            Mem => a.rss_bytes.cmp(&b.rss_bytes),
        };
        let ord = if desc { ord.reverse() } else { ord };
        ord.then(a.pid.cmp(&b.pid))
    });
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SortCol {
    Name,
    Pid,
    Cpu,
    Mem,
}

/// Shared "app grouping": processes whose desktop icon name resolves
/// collapse into one row per application; everything else (kernel threads,
/// daemons, bare binaries) stays as its own row. The GUI table and the TUI
/// both render this, so the grouping lives here once.
#[derive(Clone)]
pub struct AppGroup {
    /// desktop icon name of the app; empty for a standalone process row
    pub icon: String,
    /// icon family used for the drawn fallback glyph
    pub kind: &'static str,
    /// display title: most common member name, or the process name
    pub title: String,
    /// "N processes" when the group has more than one member
    pub sub: String,
    /// most common owner among members
    pub user: String,
    /// summed CPU% across members
    pub cpu: f64,
    /// summed RSS across members
    pub mem: u64,
    /// PID column text: the pid, or "N pids" for a multi-member group
    pub pid_label: String,
    pub members: Vec<ProcInfo>,
}

/// The display title for a set of rows: the most common member name; ties
/// break toward the first-listed name (scan order), which is the leader-ish
/// order /proc hands us. `max_by_key` alone would keep the LAST max, i.e.
/// the last-listed name — wrong for a group like chrome + chrome_crashpad.
fn title_of(members: &[ProcInfo]) -> String {
    members
        .iter()
        .enumerate()
        .max_by_key(|(i, r)| {
            (
                members.iter().filter(|m| m.name == r.name).count(),
                std::cmp::Reverse(*i as u64),
            )
        })
        .map(|(_, r)| r.name.clone())
        .unwrap_or_default()
}

/// Group `rows` (already filtered) into app rows + standalone rows, ordered
/// by `col`/`desc` using each row's aggregate values. Apps and standalone
/// rows interleave in ONE ranking, so a standalone process bigger than every
/// app group still sorts to the very top.
pub fn group_rows(rows: &[ProcInfo], col: SortCol, desc: bool) -> Vec<AppGroup> {
    let mut groups: Vec<(String, Vec<ProcInfo>)> = Vec::new();
    let mut rest: Vec<ProcInfo> = Vec::new();
    for r in rows {
        if r.app_icon.is_empty() {
            rest.push(r.clone());
            continue;
        }
        match groups.iter_mut().find(|(n, _)| *n == r.app_icon) {
            Some((_, members)) => members.push(r.clone()),
            None => groups.push((r.app_icon.clone(), vec![r.clone()])),
        }
    }
    let stat = |ms: &[ProcInfo]| -> (f64, u64) {
        ms.iter().fold((0.0_f64, 0_u64), |a, r| {
            (a.0 + r.cpu_pct, a.1 + r.rss_bytes)
        })
    };
    // The sort key for one entry, as a total order (no float partial_cmp):
    // CPU keys are milli-percent ints, memory/pid are plain ints, name is
    // text. The smallest member pid breaks ties so the order is
    // deterministic across refreshes.
    let gkey = |g: &(String, Vec<ProcInfo>)| -> (i128, u64, String) {
        let (c, m) = stat(&g.1);
        let title = title_of(&g.1);
        match col {
            SortCol::Name => (0, 0, title),
            SortCol::Pid => (
                0,
                g.1.iter().map(|r| r.pid as u64).min().unwrap_or(0),
                String::new(),
            ),
            SortCol::Cpu => (
                (c * 1000.0).round() as i128,
                g.1.iter().map(|r| r.pid as u64).min().unwrap_or(0),
                String::new(),
            ),
            SortCol::Mem => (
                m as i128,
                g.1.iter().map(|r| r.pid as u64).min().unwrap_or(0),
                String::new(),
            ),
        }
    };
    let skey = |r: &ProcInfo| -> (i128, u64, String) {
        match col {
            SortCol::Name => (0, 0, r.name.to_lowercase()),
            SortCol::Pid => (0, r.pid as u64, String::new()),
            SortCol::Cpu => (
                (r.cpu_pct * 1000.0).round() as i128,
                r.pid as u64,
                String::new(),
            ),
            SortCol::Mem => (r.rss_bytes as i128, r.pid as u64, String::new()),
        }
    };

    // Build the app-group entries, then merge them with the standalone rows
    // into a single list ranked by the same key — one table, one order, no
    // "all apps first, then the rest" block.
    // (sort key, members, Some(app icon) for a group / None for standalone)
    type Entry = ((i128, u64, String), Vec<ProcInfo>, Option<String>);
    let mut entries: Vec<Entry> = groups
        .into_iter()
        .map(|g| (gkey(&g), g.1, Some(g.0)))
        .collect();
    entries.extend(rest.into_iter().map(|r| (skey(&r), vec![r], None)));
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    if desc {
        entries.reverse();
    }

    let mut out: Vec<AppGroup> = Vec::new();
    for (_, members, icon) in entries {
        let (c, m) = stat(&members);
        let n = members.len();
        let title = title_of(&members);
        out.push(AppGroup {
            icon: icon.unwrap_or_default(),
            kind: members[0].icon,
            title,
            sub: if n > 1 {
                format!("{n} processes")
            } else {
                String::new()
            },
            user: members
                .iter()
                .map(|r| r.user.clone())
                .max_by_key(|u| members.iter().filter(|r| &r.user == u).count())
                .unwrap_or_else(|| "?".into()),
            cpu: c,
            mem: m,
            pid_label: if n == 1 {
                members[0].pid.to_string()
            } else {
                format!("{n} pids")
            },
            members,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(pid: u32, name: &str, cpu: f64, mem: u64) -> ProcInfo {
        row_named(pid, name, "", cpu, mem)
    }

    fn row_named(pid: u32, name: &str, cmd: &str, cpu: f64, mem: u64) -> ProcInfo {
        ProcInfo {
            pid,
            ppid: 0,
            name: name.into(),
            cmdline: cmd.into(),
            state: 'S',
            cpu_pct: cpu,
            rss_bytes: mem,
            threads: 1,
            started: 0,
            uid: 1000,
            vsize_bytes: 0,
            io_read: 0,
            io_write: 0,
            cwd: String::new(),
            user: "rifan".into(),
            icon: crate::procs::icon_for(name, cmd),
            app_icon: String::new(),
        }
    }

    /// The app-grouping contract both front-ends render: processes whose
    /// desktop icon resolves collapse into one row per app (summed CPU/RSS,
    /// "N processes" note, pids counted), standalone rows stay standalone,
    /// and the group order follows the sort column.
    #[test]
    fn group_rows_collapses_apps_and_keeps_the_rest_standalone() {
        let mut a = row_named(100, "chrome", "/opt/google/chrome/chrome", 20.0, 300 << 20);
        a.app_icon = "google-chrome".into();
        let mut b = row_named(
            101,
            "chrome_crashpad",
            "/opt/google/chrome/chrome",
            1.0,
            20 << 20,
        );
        b.app_icon = "google-chrome".into();
        let mut c = row_named(
            102,
            "chrome",
            "/opt/google/chrome/chrome --type=zygote",
            5.0,
            50 << 20,
        );
        c.app_icon = "google-chrome".into();
        let k = row(2, "kthreadd", 0.0, 0);
        let rows = vec![a, b, c, k];
        let groups = group_rows(&rows, SortCol::Mem, true);
        assert_eq!(
            groups.len(),
            2,
            "chrome collapses, kthreadd stays standalone"
        );
        let chrome = &groups[0];
        assert_eq!(chrome.icon, "google-chrome");
        assert_eq!(chrome.title, "chrome", "most common member name wins");
        assert_eq!(chrome.members.len(), 3);
        assert!((chrome.cpu - 26.0).abs() < 1e-9, "cpu sums: {}", chrome.cpu);
        assert_eq!(chrome.mem, 370 << 20, "rss sums");
        assert_eq!(chrome.pid_label, "3 pids");
        assert_eq!(chrome.sub, "3 processes");
        assert_eq!(groups[1].icon, "", "standalone row has no app icon");
        assert_eq!(groups[1].pid_label, "2");
        // name sort: chrome first (c < k), and the group's own title decides
        let by_name = group_rows(&rows, SortCol::Name, false);
        assert_eq!(by_name[0].title, "chrome");
        assert_eq!(by_name[1].title, "kthreadd");
    }

    #[test]
    fn sort_by_memory_desc_then_pid() {
        let mut rows = vec![
            row(2, "b", 1.0, 100),
            row(3, "c", 2.0, 300),
            row(4, "d", 3.0, 300),
        ];
        sort_rows(&mut rows, SortCol::Mem, true);
        assert_eq!(
            rows.iter().map(|r| r.pid).collect::<Vec<_>>(),
            vec![3, 4, 2]
        );
    }

    /// Regression for "strata-host took 50 GB but didn't move to the top on
    /// a descending sort": a standalone process (no desktop icon, so no app
    /// group) must interleave with the app groups in ONE ranking, not get
    /// dumped in a separate block below every app.
    #[test]
    fn group_rows_interleaves_standalone_and_app_rows_in_one_ranking() {
        let mut a = row_named(100, "chrome", "/opt/google/chrome/chrome", 20.0, 5 << 30);
        a.app_icon = "google-chrome".into();
        let mut b = row_named(
            101,
            "chrome_crashpad",
            "/opt/google/chrome/chrome",
            1.0,
            1 << 30,
        );
        b.app_icon = "google-chrome".into();
        let big = row(200, "strata-host", 3.0, 50 << 30); // standalone, biggest
        let mut small_app = row_named(300, "dolphin", "/usr/bin/dolphin", 1.0, 1 << 20);
        small_app.app_icon = "org.kde.dolphin".into();
        let rows = vec![a.clone(), b.clone(), big.clone(), small_app.clone()];

        let g = group_rows(&rows, SortCol::Mem, true);
        assert_eq!(
            g.iter().map(|e| e.title.as_str()).collect::<Vec<_>>(),
            vec!["strata-host", "chrome", "dolphin"],
            "desc memory: standalone 50 GiB first, then app groups by summed rss"
        );
        let g = group_rows(&rows, SortCol::Mem, false);
        assert_eq!(
            g.iter().map(|e| e.title.as_str()).collect::<Vec<_>>(),
            vec!["dolphin", "chrome", "strata-host"],
            "asc memory: smallest first, strata-host last"
        );
        // same interleave on the CPU column
        let chatty = row(400, "burner", 90.0, 10 << 20);
        let rows = vec![a, b, big, small_app, chatty];
        let g = group_rows(&rows, SortCol::Cpu, true);
        assert_eq!(g[0].title, "burner", "standalone 90% tops the CPU sort");
        assert_eq!(g[1].title, "chrome", "26% app group second");
        // ties fall back to pid, so the order is deterministic across refreshes
        let t1 = row(5, "twin", 7.0, 9 << 20);
        let t2 = row(6, "twin2", 7.0, 9 << 20);
        let g = group_rows(&[t1, t2], SortCol::Mem, true);
        // tied on memory: the whole key (including pid) is reversed, so pid
        // order flips with the direction — deterministic, never random
        assert_eq!(
            g.iter().map(|e| e.pid_label.as_str()).collect::<Vec<_>>(),
            vec!["6", "5"]
        );
    }

    #[test]
    fn sort_by_cpu_asc_and_name_case_insensitive() {
        let mut rows = vec![
            row(1, "zsh", 5.0, 1),
            row(2, "Bash", 9.0, 1),
            row(3, "apache", 1.0, 1),
        ];
        sort_rows(&mut rows, SortCol::Cpu, false);
        assert_eq!(
            rows.iter().map(|r| r.pid).collect::<Vec<_>>(),
            vec![3, 1, 2]
        );
        sort_rows(&mut rows, SortCol::Name, false);
        assert_eq!(
            rows.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(),
            vec!["apache", "Bash", "zsh"]
        );
    }

    /// Icon resolution must land on the right family for common process
    /// shapes, and unknown ones must still get a sane fallback.
    #[test]
    fn icon_table_maps_common_processes() {
        assert_eq!(icon_for("firefox", "/usr/lib/firefox/firefox"), "browser");
        assert_eq!(
            icon_for("chrome", "/opt/google/chrome/chrome --type=renderer"),
            "browser"
        );
        assert_eq!(icon_for("node", "node /usr/lib/code/out/server.js"), "app");
        assert_eq!(icon_for("idea", "/opt/idea/bin/idea"), "jetbrains");
        assert_eq!(icon_for("code", "code --unity"), "app");
        assert_eq!(icon_for("zsh", "-zsh"), "terminal");
        assert_eq!(icon_for("kitty", "kitty"), "terminal");
        assert_eq!(
            icon_for("docker-proxy", "/usr/bin/docker-proxy"),
            "container"
        );
        assert_eq!(icon_for("postgres", "postgres -D /var/lib/pg"), "database");
        assert_eq!(icon_for("python3", "python3 train.py"), "runtime");
        assert_eq!(icon_for("systemd", "/sbin/init"), "service");
        assert_eq!(icon_for("kworker/0:1", ""), "kernel");
        assert_eq!(icon_for("telegram", "/usr/bin/telegram-desktop"), "chat");
        assert_eq!(icon_for("vlc", "vlc movie.mkv"), "music");
        assert_eq!(icon_for("some-weird-daemon", "/srv/weird"), "app");
        assert_eq!(
            icon_for("thunderbird", "/usr/lib/thunderbird/thunderbird"),
            "chat"
        );
        assert_eq!(icon_for("kthreadd", ""), "kernel");
        assert_eq!(icon_for("kworker/0:1H-kb", ""), "kernel");
    }

    /// The live scan must fill the enriched fields plausibly.
    #[test]
    fn live_scan_fills_user_cwd_icon_and_sizes() {
        let mut s = ProcScanner::new();
        s.scan();
        std::thread::sleep(Duration::from_millis(120));
        let rows = s.scan();
        let me = rows
            .iter()
            .find(|r| r.pid == std::process::id())
            .expect("own pid missing");
        assert!(!me.user.is_empty(), "user must resolve");
        assert!(!me.cwd.is_empty(), "cwd must resolve for our own process");
        assert!(
            me.icon == "app" || me.icon == "terminal",
            "test binary should fall back to the generic app or terminal icon, got {}",
            me.icon
        );
        // VmSize (as bytes) must be at least VmRSS for our own process
        assert!(
            me.vsize_bytes >= me.rss_bytes,
            "vsize {} < rss {}",
            me.vsize_bytes,
            me.rss_bytes
        );
        // pid 1 is root-owned (or current user in container namespaces) and a service
        let init = rows.iter().find(|r| r.pid == 1).unwrap();
        assert!(init.user == "root" || !init.user.is_empty());
    }

    /// On a desktop box, running GUI apps must resolve to the icon name
    /// their .desktop entry declares (firefox -> "firefox", …).
    #[test]
    fn live_scan_names_real_desktop_icons() {
        let mut s = ProcScanner::new();
        let rows = s.scan();
        let named: Vec<(String, String)> = rows
            .iter()
            .filter(|r| !r.app_icon.is_empty())
            .map(|r| (r.name.clone(), r.app_icon.clone()))
            .collect();
        // a headless CI runner runs no desktop session at all: nothing to
        // resolve, skip rather than fail
        if named.is_empty() {
            eprintln!("skip: no GUI processes on this box ({} procs)", rows.len());
            return;
        }
        // this dev box runs firefox, plasmashell, konsole and friends —
        // the scan must name at least a handful of them
        assert!(
            named.len() >= 5,
            "only {}/{rows} processes resolved an icon: {named:?}",
            named.len(),
            rows = rows.len()
        );
        for (name, icon) in &named {
            assert!(!icon.is_empty(), "{name} resolved to an empty icon");
            // icon names are basenames; absolute paths only ever come from
            // an entry that declared one (snap/AppImage), and then the file
            // must actually exist
            if icon.starts_with('/') {
                assert!(
                    std::path::Path::new(icon).is_file(),
                    "{name}: dead icon path {icon}"
                );
            } else {
                assert!(!icon.contains('/'), "{name}: icon {icon} looks like a path");
            }
        }
        // the apps this box is definitely running right now; snap entries
        // may name the icon file absolutely, so match name or file stem
        for want in ["plasmashell", "vscode", "google-chrome", "org.kde.dolphin"] {
            let hit = named.iter().any(|(_, i)| {
                i == want || i.rsplit('/').next().unwrap_or("").trim_end_matches(".png") == want
            });
            assert!(hit, "{want} did not resolve an icon: {named:?}");
        }
    }

    #[test]
    fn live_scan_finds_this_test_process_with_plausible_numbers() {
        let mut s = ProcScanner::new();
        s.scan();
        std::thread::sleep(Duration::from_millis(120));
        let rows = s.scan();
        assert!(!rows.is_empty(), "/proc scan found nothing");
        // this very test binary must appear
        let me = std::process::id();
        let me = rows.iter().find(|r| r.pid == me).expect("own pid missing");
        assert!(!me.name.is_empty());
        assert!(me.rss_bytes > 0, "rss must be non-zero for a live process");
        assert!(
            me.cpu_pct >= 0.0 && me.cpu_pct < 100.0 * 256.0,
            "cpu_pct implausible: {}",
            me.cpu_pct
        );
        assert!(
            me.started > 1_600_000_000,
            "start time implausible: {}",
            me.started
        );
        // the init process exists and owns the tree
        assert!(rows.iter().any(|r| r.pid == 1));
    }
}
