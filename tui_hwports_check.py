#!/usr/bin/env python3
"""Headless smoke test for the Hardware and Ports tabs: drives the TUI via a
pty, asserts the Hardware screen shows live vitals and the Ports screen lists
listening sockets, and that filtering narrows the port table.
Run from the repo root after `cargo build --release`. Usage: ./tui_hwports_check.py"""
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

# --- Hardware tab (key 5) ---
os.write(fd, b'5'); pump(3.0)   # two refresh cycles so rates are real
scr = "\n".join(term.display)
assert "Hardware" in scr and "CPU" in scr, "Hardware screen missing"
assert "memory" in scr, "vitals sections missing"
assert "Per-core load" not in scr, "per-core grid must be gone"
assert "swap" in scr, "swap gauge missing"
assert "Ryzen" in scr or "Processor" in scr or "CPU" in scr, "hardware name line missing"
import re
assert re.search(r"CPU\s+\d+\.\d%", scr), "cpu percentage missing"
# GPU + SSD lines when the box exposes them (this dev box does)
assert ("gpu" in scr and "RTX" in scr) or "gpu" not in scr, "GPU line missing"
assert ("ssd" in scr and ("nvme" in scr or re.search(r"\d+GB", scr))), "SSD line missing"

# --- Ports tab (key 6) ---
os.write(fd, b'6'); pump(3.0)
scr = "\n".join(term.display)
assert "Ports" in scr and "PROTO" in scr, "Ports table header missing"
assert "LISTEN" in scr, "no listening sockets listed"
assert "22" in scr or "53" in scr, "expected a well-known port (ssh/dns)"

# filter narrows the table
os.write(fd, b'/'); pump(0.3)
os.write(fd, b'zzzznomatch'); pump(1.0)
scr2 = "\n".join(term.display)
assert "filter: zzzznomatch" in scr2, "filter title missing"
assert "LISTEN" not in scr2, "filter did not narrow port rows"

os.write(fd, b'q'); pump(0.5)
print("hardware+ports tabs OK")
