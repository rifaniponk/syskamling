# linux-simple-docker-desktop

A lightweight Docker Desktop-style client for Linux, built in Rust.

Talks directly to the Docker Engine API over the local Unix socket — no
Docker CLI dependency, no VM, no proxy layer. Native performance, tiny
memory footprint (~1.8 MB binary).

## Features

- Containers screen styled after Docker Desktop: name, image, state, status, ports
- Live discovery with auto-refresh (2 s default) — running and stopped containers
- Container actions: start/stop, restart, remove (with confirmation)
- Logs viewer: follow mode, scrollback (PageUp/PageDown, g/G), stderr highlighted
- Images and Volumes screens
- Filter-as-you-type on the containers list
- Keyboard-driven, mouse-free (but fast)

## Install

```sh
cargo build --release
# binary at target/release/sdd
```

## Usage

```sh
sdd            # containers screen
sdd --logs     # start directly on the logs screen
sdd -i 5       # refresh every 5 s (0 disables polling)
sdd -s /path/to/docker.sock
```

### Keys

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
