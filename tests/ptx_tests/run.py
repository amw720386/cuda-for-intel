#!/usr/bin/env python3
"""Run vosen/ptx_tests against a CUDA library and compare with expected_status.tsv.

    run.py --ptx-tests <ptx_tests binary> --cuda <library> [--elements N] [--jobs J]
           [--filter REGEX] [--out results.tsv] [--bless] [--timeout SEC]

The library is `libsimcuda.dylib`/`.so` (tests/ptx_tests/simcuda, the SPIR-V simulator) or our
`nvcuda.dll` on the GPU. `--elements` sets PTX_TESTS_ELEMENTS (inputs per test; 0 = upstream's
2^32). Tests are split over `--jobs` processes. A process that dies or goes quiet for
`--timeout` seconds marks the running test `crash` or `timeout` and the rest are run again.

`--exact-ftz` sets CFI_EXACT_FTZ=1 (exact `.ftz` lowering). expected_status.tsv rows are
`name, mode, status, reason`; mode is `all`, `default` (without CFI_EXACT_FTZ) or `exact-ftz`.

Statuses: pass, mismatch (wrong result; detail has the first failing input), unsupported (the
module did not load), panic (a Driver API call failed), crash, timeout.
Exit status: 0 when every test has the status expected_status.tsv lists (pass if absent); a
listed failure that passes (its failing inputs were not sampled) is reported but allowed.
"""
import argparse
import os
import queue
import re
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = re.compile(r"^([a-z0-9_]+): (OK|FAIL - (.*))$")
# Lines that belong to no test: our diagnostics and Rust's panic message.
NOISE = ("simcuda:", "thread '", "note: run with", "called `")
BINARY = re.compile(r"0b[01]+ ")


def classify(rest, lines, notes):
    if rest is None:
        return "pass", ""
    if rest.startswith("Compilation failed"):
        why = [n for n in notes if "translate:" in n] or notes
        return "unsupported", (why[-1] if why else " ".join(lines[1:2]) or rest)
    if rest.startswith("Panic"):
        return "panic", rest[len("Panic: "):] + ("; " + notes[-1] if notes else "")
    if rest.startswith("with input"):
        detail = " ".join([rest] + [l.strip() for l in lines[1:]])
        return "mismatch", BINARY.sub("", detail).replace("( ", "(").replace(", )", ")")
    return "mismatch", rest


def run_batch(binary, cuda, names, env, timeout, results, lock, log):
    """Runs `names` in one process, restarting after a crash or a timeout."""
    remaining = list(names)
    while remaining:
        pattern = "^(" + "|".join(remaining) + ")$"
        cmd = [binary, cuda, "-f", pattern]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True, errors="replace")
        lines_q = queue.Queue()

        def pump():
            for line in p.stdout:
                lines_q.put(line.rstrip("\r\n"))
            lines_q.put(None)

        threading.Thread(target=pump, daemon=True).start()
        current = None  # (name, rest, lines)
        notes = []
        done = set()

        def finish(cur):
            name, rest, lines = cur
            status, detail = classify(rest, lines, notes)
            with lock:
                results[name] = (status, detail.replace("\t", " "))
            done.add(name)
            notes.clear()

        last = time.time()
        timed_out = False
        while True:
            try:
                line = lines_q.get(timeout=5)
            except queue.Empty:
                if time.time() - last > timeout:
                    timed_out = True
                    p.kill()
                    break
                continue
            if line is None:
                break
            last = time.time()
            if log:
                log.write(line + "\n")
            m = RESULT.match(line)
            if m and m.group(1) in remaining:
                if current:
                    finish(current)
                current = (m.group(1), m.group(3), [m.group(3) or ""])
            elif current and current[1] is not None and not line.startswith(NOISE):
                current[2].append(line)
            else:
                if current:
                    finish(current)
                    current = None
                if line.startswith("simcuda:"):
                    notes.append(line[len("simcuda: "):])
        if current:
            finish(current)
        p.wait()
        remaining = [n for n in remaining if n not in done]
        if remaining:
            # The first unreported test is the one that was running.
            with lock:
                results[remaining[0]] = ("timeout" if timed_out else "crash", (notes[-1] if notes else f"exit {p.returncode}"))
            remaining = remaining[1:]


def load_expected(path, mode):
    """name -> (status, reason, row mode) for the rows that apply to `mode`."""
    expected = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if f[1] in ("all", mode):
                expected[f[0]] = (f[2], f[3] if len(f) > 3 else "", f[1])
    return expected


def load_rows(path):
    rows = []
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            rows.append((f[0], f[1], f[2], f[3] if len(f) > 3 else ""))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptx-tests", required=True)
    ap.add_argument("--cuda", required=True)
    ap.add_argument("--elements", type=int, default=65536)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--filter", default=None)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--expected", default=os.path.join(HERE, "expected_status.tsv"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--log", default=None)
    ap.add_argument("--exact-ftz", action="store_true", help="run with CFI_EXACT_FTZ=1")
    ap.add_argument("--bless", action="store_true", help="rewrite --expected rows of this mode, keeping reasons")
    a = ap.parse_args()

    names = subprocess.run([a.ptx_tests, "-l"], capture_output=True, text=True, check=True).stdout.split()
    if a.filter:
        names = [n for n in names if re.search(a.filter, n)]
    env = dict(os.environ)
    if a.elements:
        env["PTX_TESTS_ELEMENTS"] = str(a.elements)
    else:
        env.pop("PTX_TESTS_ELEMENTS", None)
    env.setdefault("SIMCUDA_VERBOSE", "1")
    mode = "exact-ftz" if a.exact_ftz else "default"
    env["CFI_EXACT_FTZ"] = "1" if a.exact_ftz else "0"
    results, lock = {}, threading.Lock()
    log = open(a.log, "w", encoding="utf-8") if a.log else None
    jobs = max(1, min(a.jobs, len(names)))
    started = time.time()
    threads = [
        threading.Thread(target=run_batch, args=(a.ptx_tests, a.cuda, names[k::jobs], env, a.timeout, results, lock, log))
        for k in range(jobs)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    elapsed = time.time() - started

    expected = load_expected(a.expected, mode)
    counts = {}
    for n in names:
        counts[results[n][0]] = counts.get(results[n][0], 0) + 1
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            for n in names:
                f.write(f"{n}\t{results[n][0]}\t{results[n][1]}\n")
    # A listed failure that passes is only noted: with sampled inputs a rare failing input (a
    # subnormal pair for a .ftz case) may not be drawn. Everything else must match.
    changed, now_pass = [], []
    for n in names:
        want = expected.get(n, ("pass", ""))[0]
        if results[n][0] == want:
            continue
        if results[n][0] == "pass":
            now_pass.append(n)
        else:
            changed.append((n, want, results[n][0], results[n][1]))
    print(f"{len(names)} tests ({mode}), elements {a.elements or 'all'}, {elapsed:.0f} s: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for n, want, got, detail in changed:
        print(f"  {n}: expected {want}, got {got}: {detail[:300]}")
    if now_pass:
        print(f"  listed but passed with this sample ({len(now_pass)}): " + " ".join(now_pass))
    if a.bless:
        other = "default" if mode == "exact-ftz" else "exact-ftz"
        rows = load_rows(a.expected)
        reasons = {(r[0], r[1]): r[3] for r in rows}
        # Rows of the other mode stay; `all` rows of a test that now passes here become rows of
        # the other mode only.
        kept = [r for r in rows if r[1] == other or (r[1] == "all" and r[0] not in names)]
        for r in rows:
            if r[1] == "all" and r[0] in names and results[r[0]][0] == "pass":
                kept.append((r[0], other, r[2], r[3]))
        merged = {(r[0], r[1]): r for r in kept}
        for n in names:
            status = results[n][0]
            if status == "pass":
                continue
            reason = reasons.get((n, "all")) or reasons.get((n, mode)) or "TODO: " + results[n][1][:200]
            row_mode = "all" if (n, "all") in reasons else mode
            other_row = merged.get((n, other))
            if row_mode == mode and other_row and other_row[2] == status and other_row[3] == reason:
                del merged[(n, other)]
                row_mode = "all"
            merged[(n, row_mode)] = (n, row_mode, status, reason)
        with open(a.expected, "w", encoding="utf-8", newline="\n") as f:
            f.write("# vosen/ptx_tests cases that do not pass: name, mode, status, reason. Absent means pass.\n")
            f.write("# mode: all, default (CFI_EXACT_FTZ unset) or exact-ftz. Regenerate with run.sh --bless.\n")
            for key in sorted(merged):
                f.write("\t".join(merged[key]) + "\n")
        print("blessed", a.expected)
        return 0
    if not changed:
        print("all tests match expected_status.tsv")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
