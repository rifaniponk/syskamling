//! Process scanner: every process on the box with live CPU% and memory,
//! Task-Manager style. Reads /proc, keeps the previous tick counters so the
//! delta over the scan interval gives a real busy percentage.

use std::collections::HashMap;
use std::fs;
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
    /// process owner uid
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
    /// resolved application icon, e.g. "firefox" or "terminal"
    pub icon: &'static str,
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
        Self { prev: HashMap::new(), prev_at: None, users: HashMap::new(), users_at: None }
    }

    pub fn scan(&mut self) -> Vec<ProcInfo> {
        let btime = boot_epoch_secs();
        let ticks = 100.0_f64; // KERNEL_TICKS on every common Linux port
        let now = Instant::now();
        let dt = self.prev_at.map(|t| now.duration_since(t)).unwrap_or(Duration::ZERO);
        let page = page_size();

        // /etc/passwd is tiny and changes rarely: re-read it at most once a
        // minute so rows can show "rifan" instead of "1000"
        if self.users_at.map(|t| t.elapsed() > Duration::from_secs(60)).unwrap_or(true) {
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
            let Ok(pid): Result<u32, _> = name.parse() else { continue };

            let stat = match fs::read(format!("/proc/{pid}/stat")) {
                Ok(b) => String::from_utf8_lossy(&b).into_owned(),
                Err(_) => continue,
            };
            // comm is in parentheses and may contain spaces/parens: split at
            // the LAST ')' then parse the fields after it.
            let Some(close) = stat.rfind(')') else { continue };
            let name = stat[..close].find('(').map(|i| stat[i + 1..close].to_string()).unwrap_or_default();
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
                    let d = cpu_ticks.saturating_sub(*p) as f64 / ticks / dt.as_secs_f64() * 100.0;
                    d
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
            rows.push(ProcInfo {
                pid,
                ppid,
                name: name,
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
    if has(&["firefox", "icecat", "torbro"]) {
        "browser"
    } else if has(&["chrom", "brave", "edge", "vivaldi", "opera", "360chrome"]) {
        "browser"
    } else if has(&["electron", "code", "code-insiders", "cursor", "windsurf", "trae", "codium", "strawberry"]) {
        "app"
    } else if has(&["jetbrains", "idea", "pycharm", "goland", "webstorm", "clion", "rider", "android studio", "rustrover", "rustrover"]) {
        "jetbrains"
    } else if has(&["steam", "lutris", "gamescope", "mangohud", "prismlauncher", "heroic"]) {
        "game"
    } else if has(&["libreoffice", "soffice", "gimp", "inkscape", "krita", "blender", "darktable", "digi", "shotcut", "kdenlive", "obs", "audacity", "scribus", "draw.io", "dia"]) {
        "media"
    } else if has(&["vlc", "mpv", "totem", "haruna", "celluloid", "mpd", "spotify", "cmus", "ncmpcpp", "rhythmbox", "strawberry", "audacious"]) {
        "music"
    } else if has(&["telegram", "whatsapp", "signal", "discord", "webkit", "thunderbird", "evolution", "kontact", "gajim", "pidgin", "polari", "fractal", "element", "webex", "zoom", "jitsi", "messagingmenu"]) {
        "chat"
    } else if has(&["file-manager", "nautilus", "dolphin", "thunar", "nemo", "pcmanfm", "krusader", "mousepad", "gedit", "kate", "leafpad", "galculator", "baobab", "disk-usage", "file-roller", "engrampa", "xarchiver"]) {
        "files"
    } else if has(&["docker-proxy", "dockerd", "containerd", "podman", "crio", "kube", "runc", "containerd-shim"]) {
        "container"
    } else if has(&["postgres", "mysqld", "mariadbd", "redis", "mongod", "mariadbd", "clickhouse", "influx", "etcd"]) {
        "database"
    } else if has(&["node", "bun", "deno", "python", "python3", "java", "ruby", "perl", "php", "dotnet", "beam.smp", "erl", "uvicorn", "gunicorn"]) {
        "runtime"
    } else if has(&["omp", "claude", "codex", "aider", "opencode", "hermes", " goose", "hermes-agent"]) {
        "agent"
    } else if has(&["xterm", "kitty", "alacritty", "konsole", "gnome-terminal", "foot", "wezterm", "tilix", "st", "terminal", "terminator", "qterminal", "xfce4-terminal", "ghostty", "tmux", "zellij", "screen"])
        || matches!(c.as_str(), "zsh" | "bash" | "fish" | "sh" | "dash" | "csh" | "tcsh" | "nu")
    {
        "terminal"
    } else if has(&["pipewire", "wireplumber", "pulseaudio", "alsa", "alsa-seq", "alsa-sink", "alsa-source"]) {
        "audio"
    } else if matches!(c.as_str(), "systemd" | "init") || has(&["systemd-", "udev", "journald", "logind", "dbus-daemon", "dbus-broker", "cron", "crond", "atd", "agetty", "polkit"]) {
        "service"
    } else if has(&["nm-applet", "blueman", "bluetooth", "networkmanager", "wpa_supplicant", "geoclue", "upower", "powerdevil", "gdm-", "sddm", "lightdm", "lxqt", "accounts-daemon", "udisksd", "packagekit", "flatpak", "snapd"]) {
        "settings"
    } else if has(&["x11", "wayland", "weston", "hyprland", "sway", "kwin", "mutter", "xwayland", "plasma", "gnome-shell", "xfdesktop", "xfce", "kde"])
        || c.starts_with("xfce")
        || matches!(c.as_str(), "xorg" | "x")
    {
        "desktop"
    } else if c.starts_with("kworker") || c.starts_with("kthread") || c.starts_with("ksoftirqd") || c.starts_with("migration/") || c.starts_with("rcu_") || c.starts_with("kcompactd") || c.starts_with("writeback") || c.starts_with("ext4-") || c.starts_with("xfs-") || c.starts_with("nvme-") || c.starts_with("kswapd") {
        "kernel"
    } else {
        "app"
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn row(pid: u32, name: &str, cpu: f64, mem: u64) -> ProcInfo {
        ProcInfo {
            pid,
            ppid: 0,
            name: name.into(),
            cmdline: String::new(),
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
            icon: crate::procs::icon_for(name, ""),
        }
    }

    #[test]
    fn sort_by_memory_desc_then_pid() {
        let mut rows = vec![row(2, "b", 1.0, 100), row(3, "c", 2.0, 300), row(4, "d", 3.0, 300)];
        sort_rows(&mut rows, SortCol::Mem, true);
        assert_eq!(rows.iter().map(|r| r.pid).collect::<Vec<_>>(), vec![3, 4, 2]);
    }

    #[test]
    fn sort_by_cpu_asc_and_name_case_insensitive() {
        let mut rows = vec![row(1, "zsh", 5.0, 1), row(2, "Bash", 9.0, 1), row(3, "apache", 1.0, 1)];
        sort_rows(&mut rows, SortCol::Cpu, false);
        assert_eq!(rows.iter().map(|r| r.pid).collect::<Vec<_>>(), vec![3, 1, 2]);
        sort_rows(&mut rows, SortCol::Name, false);
        assert_eq!(rows.iter().map(|r| r.name.as_str()).collect::<Vec<_>>(), vec!["apache", "Bash", "zsh"]);
    }

    /// Icon resolution must land on the right family for common process
    /// shapes, and unknown ones must still get a sane fallback.
    #[test]
    fn icon_table_maps_common_processes() {
        assert_eq!(icon_for("firefox", "/usr/lib/firefox/firefox"), "browser");
        assert_eq!(icon_for("chrome", "/opt/google/chrome/chrome --type=renderer"), "browser");
        assert_eq!(icon_for("node", "node /usr/lib/code/out/server.js"), "app");
        assert_eq!(icon_for("idea", "/opt/idea/bin/idea"), "jetbrains");
        assert_eq!(icon_for("code", "code --unity"), "app");
        assert_eq!(icon_for("zsh", "-zsh"), "terminal");
        assert_eq!(icon_for("kitty", "kitty"), "terminal");
        assert_eq!(icon_for("docker-proxy", "/usr/bin/docker-proxy"), "container");
        assert_eq!(icon_for("postgres", "postgres -D /var/lib/pg"), "database");
        assert_eq!(icon_for("python3", "python3 train.py"), "runtime");
        assert_eq!(icon_for("systemd", "/sbin/init"), "service");
        assert_eq!(icon_for("kworker/0:1", ""), "kernel");
        assert_eq!(icon_for("telegram", "/usr/bin/telegram-desktop"), "chat");
        assert_eq!(icon_for("vlc", "vlc movie.mkv"), "music");
        assert_eq!(icon_for("some-weird-daemon", "/srv/weird"), "app");
        assert_eq!(icon_for("thunderbird", "/usr/lib/thunderbird/thunderbird"), "chat");
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
        let me = rows.iter().find(|r| r.pid == std::process::id()).expect("own pid missing");
        assert!(!me.user.is_empty(), "user must resolve");
        assert!(!me.cwd.is_empty(), "cwd must resolve for our own process");
        assert_eq!(me.icon, "app", "test binary should fall back to the generic app icon");
        // VmSize (as bytes) must be at least VmRSS for our own process
        assert!(me.vsize_bytes >= me.rss_bytes, "vsize {} < rss {}", me.vsize_bytes, me.rss_bytes);
        // pid 1 is root-owned and a service
        let init = rows.iter().find(|r| r.pid == 1).unwrap();
        assert_eq!(init.user, "root");
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
        assert!(me.cpu_pct >= 0.0 && me.cpu_pct < 100.0 * 256.0, "cpu_pct implausible: {}", me.cpu_pct);
        assert!(me.started > 1_600_000_000, "start time implausible: {}", me.started);
        // the init process exists and owns the tree
        assert!(rows.iter().any(|r| r.pid == 1));
    }
}
