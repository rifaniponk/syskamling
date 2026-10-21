# linux-simple-docker-desktop

A lightweight Docker Desktop-style client for Linux, built in Rust.

Talks directly to the Docker Engine API over the local Unix socket — no
Docker CLI dependency, no VM, no proxy layer. Native performance, tiny
memory footprint.

## Features

- Containers screen styled after Docker Desktop: name, image, status, ports
- Live container discovery with auto-refresh
- Detail panel: state, image, ports, uptime, start/stop/restart/remove
- Logs viewer with follow mode, timestamps, and tail window
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
```

### Keys

| Key | Action |
|---|---|
| `1` / `Tab` | Containers screen |
| `2` | Logs screen |
| `j` `k` / `↑` `↓` | Move selection |
| `Enter` | Open logs for selected container |
| `s` | Start container |
| `t` | Stop container |
| `r` | Restart container |
| `x` | Remove container (confirm with `y`) |
| `f` | Toggle log follow |
| `pgup`/`pgdn`, `g`/`G` | Scroll logs |
| `R` | Force refresh |
| `q` / `Esc` | Quit |

## Why Rust

The official Docker Desktop runs a VM and Electron UI on Linux. This tool is
a single static-ish binary (~3 MB) that reads `/var/run/docker.sock`
directly: container list refreshes in ~10 ms, log streaming is zero-copy
through a tokio task, and idle CPU is ~0%.
