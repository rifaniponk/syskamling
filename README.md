# linux-simple-docker-desktop

A lightweight Docker Desktop-style desktop app for Linux, built in Rust.

Talks directly to the Docker Engine API over the local Unix socket — no
Docker CLI dependency, no VM, no proxy layer, no Electron. A single native
egui/eframe window, ~10–20 MB resident.

## Features

- Desktop GUI (default): Containers / Images / Volumes / Logs tabs, dark
  Docker-blue theme, status-dot pulse, action spinners, log fade-in
- Container actions from the detail panel: start/stop, restart, logs,
  remove (with confirmation dialog)
- Live discovery with auto-refresh (2 s default) — running and stopped containers
- Logs viewer: follow mode, scrollback (PageUp/PageDown, g/G), stderr highlighted
- Filter-as-you-type across containers, images, and volumes
- Terminal UI (`sdd tui`) for keyboard-driven, mouse-free use over SSH

## Install

```sh
cargo build --release
# binary at target/release/sdd
```

## Usage

```sh
sdd            # desktop window
sdd gui        # same thing, explicit
sdd tui        # terminal UI
sdd -i 5       # refresh every 5 s (0 disables polling)
sdd -s /path/to/docker.sock
```

### GUI

- Tabs top-right switch screens; `⟳` forces a refresh.
- Click a container to select it; the detail panel shows state, image,
  command, ports, and action buttons (Start/Stop, Restart, Logs, Remove).
- Double-click a container to jump straight to its logs.
- The filter box at the top of the sidebar filters the current list.

### TUI keys

| Key | Action |
|---|---|
| `1`–`4` / `Tab` `←` `→` | Switch screen |
| `j` `k` / `↑` `↓` | Move selection |
| `Enter` | Open logs for selected container |
| `s` | Start (if stopped) / stop (if running) |
| `t` | Stop container |
| `r` | Restart container |
| `x` | Remove container (`y` confirm, `n`/`Esc` cancel) |
| `f` | Toggle log follow |
| `PgUp` `PgDn`, `g`/`G` | Scroll logs (`G` = bottom + follow) |
| `c` | Clear log page (stream keeps running) |
| `/` | Filter containers |
| `R` | Force refresh |
| `q` / `Esc` | Quit |

## Desktop integration (pin to dash)

The app sets its Wayland app id / X11 WM class to `io.sdd.SimpleDockerDesktop`.
To get a launcher you can pin to the KDE dash:

```sh
# 1. launcher entry
cp io.sdd.SimpleDockerDesktop.desktop ~/.local/share/applications/
# edit Exec= to point at your absolute target/release/sdd path

# 2. icon (the whale mark from assets/icon.svg, rendered to PNG)
for s in 512 256 128 64 48 32; do
  mkdir -p ~/.local/share/icons/hicolor/${s}x${s}/apps
  magick assets/icon.png -resize ${s}x${s} \
    ~/.local/share/icons/hicolor/${s}x${s}/apps/io.sdd.SimpleDockerDesktop.png
done
gtk-update-icon-cache -f ~/.local/share/icons/hicolor
```

Then search "Simple Docker Desktop" in the application launcher and pick
*Pin to Task Manager / Add to Dash*. `StartupWMClass` ensures the pinned
icon matches the running window.

## Why Rust

The official Docker Desktop runs a VM and Electron UI on Linux. This tool is
a single small binary that reads `/var/run/docker.sock` directly: the UI
loop never blocks (all Engine calls run on a tokio runtime and report back
through a channel), log streaming is incremental through the multiplexed
stream decoder, and idle CPU is ~0%.

## Tests

```sh
cargo test    # HTTP chunked + Docker stream framing decoders
./smoke_test.py   # headless TUI drive (pty + pyte); requires a live daemon
```
