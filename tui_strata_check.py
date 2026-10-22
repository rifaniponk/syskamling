#!/usr/bin/env python3
"""Headless smoke test for the Strata tab: drives the TUI via a pty, asserts
the Strata screen reaches the configured inference server and renders live
statistics (model state, KPI lines, recent-request table). Needs a Strata
server at the endpoint in ~/.config/syskamling/config.json.
Run from the repo root after `cargo build --release`. Usage: ./tui_strata_check.py"""
import os, sys, pty, time, struct, fcntl, termios, select
try:
    import pyte
except ImportError:
    sys.exit("pyte missing: pip install pyte")
ROWS, COLS = 44, 150
if not os.path.exists('./target/release/syskamling'):
    sys.exit('build first: cargo build --release')
if not os.path.exists(os.path.expanduser('~/.config/syskamling/config.json')):
    sys.exit('no ~/.config/syskamling/config.json — configure the strata endpoint+key first')
pid, fd = pty.fork()
if pid == 0:
    os.execvp('./target/release/syskamling', ['./target/release/syskamling', 'tui'])
fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack('HHHH', ROWS, COLS, 0, 0))
term = pyte.Screen(COLS, ROWS); stream = pyte.ByteStream(term)
def pump(dur=1.0):
    end = time.time() + dur
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.1)
        if r:
            try: stream.feed(os.read(fd, 65536))
            except OSError: break
pump(1.5)

# --- Strata tab (key 7) ---
os.write(fd, b'7'); pump(4.0)   # give the poller two refresh cycles
scr = "\n".join(term.display)
assert "Strata" in scr, "Strata screen missing"
assert "OFFLINE" not in scr and "not configured" not in scr, f"server unreachable:\n{scr}"
assert "Idle" in scr or "Reading" in scr or "Generating" in scr, "no model-state pill"
assert "GPU speed" in scr and "VRAM" in scr, "KPI lines missing"
assert "Context" in scr, "context line missing"
assert "Recent requests" in scr, "request table missing"
import re
assert re.search(r"GPU speed\s+[\d.]+ tok/s", scr), "gpu speed value missing"

# 'a' toggles the full request list without crashing
os.write(fd, b'a'); pump(1.0)
scr2 = "\n".join(term.display)
assert "Recent requests" in scr2, "table vanished after toggle"

os.write(fd, b'q'); pump(0.5)
print("strata tab OK")
