#!/usr/bin/env python3
"""Headless smoke test for syskamling: drives the TUI via a pty, asserts on the screen.

Run from the repo root after `cargo build --release`. Needs a live Docker
daemon. Usage: ./smoke_test.py
"""
import fcntl, os, pty, select, struct, sys, termios, time

import pyte

ROWS, COLS = 40, 140
FAILURES = []


def pump(fd, buf, dur=0.5):
    end = time.time() + dur
    while time.time() < end:
        r, _, _ = select.select([fd], [], [], 0.1)
        if r:
            try:
                d = os.read(fd, 65536)
            except OSError:
                return False
            if not d:
                return False
            buf.extend(d)
    return True


def check(label, ok):
    print(('PASS' if ok else 'FAIL'), '-', label)
    if not ok:
        FAILURES.append(label)


def main():
    if not os.path.exists('./target/release/syskamling'):
        sys.exit('build first: cargo build --release')
    pid, fd = pty.fork()
    if pid == 0:
        os.execvp('./target/release/syskamling', ['./target/release/syskamling', 'tui'])
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack('HHHH', ROWS, COLS, 0, 0))
    term = pyte.Screen(COLS, ROWS)
    stream = pyte.ByteStream(term)
    buf = bytearray()

    def feed():
        stream.feed(bytes(buf))
        buf.clear()

    def find(sub):
        return [l.strip() for l in term.display if sub in l]

    def press(keys, dur=0.4):
        for k in keys:
            os.write(fd, k.encode() if isinstance(k, str) else k)
            pump(fd, buf, dur)
        feed()

    pump(fd, buf, 1.2)
    feed()
    check('containers list rendered', bool(find('NAME')) and bool(find('containers:')))

    # enter -> logs stream loads and follows
    press(['\r'], 1.5)
    check('logs opened', bool(find('Logs:')) and bool(find('follow: on')))
    check('log lines rendered', len([l for l in term.display[4:-2] if l[1:-1].strip()]) > 10)

    press(['\x1b[5~', '\x1b[5~'], 0.5)
    check('PageUp pauses follow', bool(find('follow: off')))
    press(['G'], 0.5)
    check('G resumes follow', bool(find('follow: on')))

    press(['\t'], 0.6)
    check('tab -> images', bool(find('Images')))

    # filter: type, Esc cancels without leaking keys
    press(['/'], 0.3)
    for ch in 'kafka':
        os.write(fd, ch.encode())
        pump(fd, buf, 0.15)
    pump(fd, buf, 0.4)
    feed()
    check('filter narrows list', bool(find('filter: kafka')))
    os.write(fd, b'\x1b')
    pump(fd, buf, 0.5)
    feed()
    check('Esc cancels filter', not find('filter:'))

    # remove confirmation: n cancels, stray keys ignored
    press(['x'], 0.4)
    check('remove prompt', bool(find('press y to confirm')))
    press(['z'], 0.4)
    check('stray key does not cancel', bool(find('press y to confirm')))
    press(['n'], 0.4)
    check('n cancels remove', bool(find('remove cancelled')))

    # Processes screen: key 7, table with live rows, 'c' re-sorts without dying
    press(['7'], 2.5)
    procs_ok = all(k in ''.join(term.display) for k in ('Processes', 'PID', 'CPU', 'MEMORY'))
    press(['c'], 0.6)
    procs_ok = procs_ok and 'Processes' in ''.join(term.display)

    press(['q'], 0.5)
    alive = os.waitpid(pid, os.WNOHANG)[0] == 0
    check('processes screen renders', procs_ok)
    check('q quits cleanly', not alive)
    if alive:
        os.kill(pid, 9)
        os.waitpid(pid, 0)

    sys.exit(1 if FAILURES else 0)


if __name__ == '__main__':
    main()
