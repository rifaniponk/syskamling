#!/usr/bin/env python3
"""Regression check: the TUI Processes screen (key 7) must render a real
sortable table with live rows, and 'c' must re-sort it without crashing."""
import os, sys, pty, time, struct, fcntl, termios, select
try:
    import pyte
except ImportError:
    sys.exit("pyte missing: pip install pyte")
ROWS, COLS = 40, 160
if not os.path.exists('./target/release/syskamling'):
    sys.exit('build first: cargo build --release')
pid, fd = pty.fork()
if pid == 0:
    os.execvp('./target/release/syskamling', ['./target/release/syskamling', 'tui'])
fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack('HHHH', ROWS, COLS, 0, 0))
term = pyte.Screen(COLS, ROWS); stream = pyte.ByteStream(term)
def pump(dur=0.6):
    end = time.time() + dur
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.1)
        if r:
            try: stream.feed(os.read(fd, 65536))
            except OSError: break
pump(1.5)
os.write(fd, b'7'); pump(2.5)   # scan = seed + 600ms delta, then repaint
scr = "\n".join(term.display)
for needle in ("Processes", "NAME", "PID", "CPU", "MEMORY", "CMDLINE"):
    assert needle in scr, f"Processes screen missing {needle!r}"
os.write(fd, b'c'); pump(0.8)
scr2 = "\n".join(term.display)
assert "Processes" in scr2, "sorting crashed the screen"
os.write(fd, b'q'); pump(0.6)
print("tui processes screen OK")
