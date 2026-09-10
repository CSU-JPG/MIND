"""run_dreamx.py — char-by-char stream a subprocess to stdout+log with timestamps.

Replaces the older run_dreamx.ps1 (PowerShell version). Char-level reads keep
tqdm `\r` progress-bar updates visible — line-buffered pipes lose them.

Invocation (from any drive_*.bat):
    "%PY%" "%~dp0run_dreamx.py" "%LOG%" "%PY%" "src\\drive_X.py" --arg1 v1 ...

argv layout:
    argv[1]   = log file path
    argv[2:]  = command + arguments (executable first)

Output: each non-empty line read from the subprocess (separator = \\n or \\r)
gets prefixed with HH:MM:SS, written to the corresponding stream
(stdout/stderr) and also to the log file. Exit code mirrors the subprocess.
"""

import datetime
import subprocess
import sys
import threading


def _stream(reader, target, log_fh, log_lock):
    buf = []
    while True:
        c = reader.read(1)
        if not c:
            break
        if c == "\r" or c == "\n":
            if buf:
                line = "".join(buf)
                stamped = f"{datetime.datetime.now():%H:%M:%S} {line}"
                try:
                    target.write(stamped + "\n")
                    target.flush()
                except Exception:
                    pass
                with log_lock:
                    log_fh.write(stamped + "\n")
                    log_fh.flush()
                buf.clear()
        else:
            buf.append(c)
    if buf:
        line = "".join(buf)
        stamped = f"{datetime.datetime.now():%H:%M:%S} {line}"
        try:
            target.write(stamped + "\n")
            target.flush()
        except Exception:
            pass
        with log_lock:
            log_fh.write(stamped + "\n")
            log_fh.flush()


def main() -> int:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <log_path> <executable> [args...]", file=sys.stderr)
        return 2

    log_path = sys.argv[1]
    cmd = sys.argv[2:]

    log_fh = open(log_path, "w", encoding="utf-8", buffering=1)
    log_lock = threading.Lock()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        bufsize=0,
    )

    t_out = threading.Thread(target=_stream, args=(proc.stdout, sys.stdout, log_fh, log_lock))
    t_err = threading.Thread(target=_stream, args=(proc.stderr, sys.stderr, log_fh, log_lock))
    t_out.start()
    t_err.start()
    t_out.join()
    t_err.join()

    rc = proc.wait()
    log_fh.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
