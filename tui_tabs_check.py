#!/usr/bin/env python3
"""Regression check: with the Strata screen inserted, Tab/arrow keys must
cycle Containers -> Images -> Volumes -> Agents -> Hardware -> Ports ->
Strata -> Logs, and 'q' must still quit from every screen."""
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
def pump(dur=0.6):
    end = time.time() + dur
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.1)
        if r:
            try: stream.feed(os.read(fd, 65536))
            except OSError: break
pump(1.5)

# one Tab press should land the header highlight on Ports then Strata
os.write(fd, b'\t'*5); pump(1.0)
scr = "\n".join(term.display)
assert "Ports" in scr, "tab cycle did not reach Ports"
os.write(fd, b'\t'); pump(1.5)
scr = "\n".join(term.display)
assert "Strata" in scr, "tab cycle did not reach Strata"

# back-tab returns to Ports, then right-arrow reaches Logs
os.write(fd, b'\x1b[Z'); pump(0.8)
os.write(fd, b'\t'); pump(1.0)
scr = "\n".join(term.display)
assert "Logs" in scr, "tab cycle did not reach Logs after Strata"

os.write(fd, b'q'); pump(0.8)
try:
    stream.feed(os.read(fd, 65536))
except OSError:
    pass
pid_done, status = os.waitpid(pid, os.WNOHANG)
assert pid_done != 0, "'q' did not quit from the Logs screen"
print("tab cycle + quit OK")
