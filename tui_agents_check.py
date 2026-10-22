#!/usr/bin/env python3
"""Headless smoke test for the Agents tab: drives the TUI via a pty,
asserts the tab lists agent sessions and that filtering narrows them.
Run from the repo root after `cargo build --release`. Usage: ./tui_agents_check.py"""
import os, sys, pty, time, struct, fcntl, termios, select
try:
    import pyte
except ImportError:
    sys.exit("pyte missing: pip install pyte")
ROWS, COLS = 40, 140
if not os.path.exists('./target/release/syskamling'):
    sys.exit('build first: cargo build --release')
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
pump(2.0)
os.write(fd, b'4'); pump(2.0)
scr = "\n".join(term.display)
assert "Agents" in scr and "AGENT" in scr, "Agents tab header missing"
assert any(k in scr for k in ("omp", "Claude Code", "Codex", "Copilot")), "no agent rows"
os.write(fd, b'/'); pump(0.3)
os.write(fd, b'zzzznomatch'); pump(1.0)
scr2 = "\n".join(term.display)
assert "filter: zzzznomatch" in scr2, "filter title missing"
assert not any(k in scr2 for k in ("Claude Code", "Codex CLI")), "filter did not narrow"
os.write(fd, b'q'); pump(0.5)
print("agents tab OK")
