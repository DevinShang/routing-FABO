#!/usr/bin/env python3

import argparse
import csv
import os
import selectors
import signal
import subprocess
import sys
import time
from collections import Counter, deque
from pathlib import Path


def load_tasks(path, done):
    tasks = []
    with path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            task_id = line.split("\t", 1)[0]
            if task_id in done:
                continue
            tasks.append((task_id, line))
    return tasks


def completed_task_ids(path):
    done = set()
    if not path.exists():
        return done
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") == "ok" and row.get("task_id"):
                done.add(row["task_id"])
    return done


def worker_env():
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return env


def start_worker(worker_id, binary, cwd, log_dir):
    log_path = log_dir / f"worker_{worker_id:03d}.err"
    log = log_path.open("ab", buffering=0)
    proc = subprocess.Popen(
        [str(binary)],
        cwd=str(cwd),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=log,
        text=True,
        bufsize=1,
        env=worker_env(),
        preexec_fn=os.setsid,
    )
    return {"id": worker_id, "proc": proc, "log": log, "inflight": None}


def stop_worker(worker):
    proc = worker["proc"]
    try:
        if proc.stdin and proc.poll() is None:
            proc.stdin.write("__quit__\n")
            proc.stdin.flush()
            proc.stdin.close()
    except BrokenPipeError:
        pass
    try:
        proc.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            pass
    worker["log"].close()


def run_queue(root, binary, task_tsv, out_csv, workers, resume):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    log_dir = out_csv.parent / (out_csv.stem + "_worker_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    done = completed_task_ids(out_csv) if resume else set()
    tasks = load_tasks(task_tsv, done)
    pending = deque(tasks)
    total = len(done) + len(tasks)
    completed = len(done)
    retries = Counter()

    write_header = not resume or not out_csv.exists() or out_csv.stat().st_size == 0
    out = out_csv.open("a" if resume else "w", newline="", buffering=1)
    if write_header:
        header = subprocess.check_output([str(binary), "--header"], cwd=str(root), text=True).strip()
        out.write(header + "\n")

    selector = selectors.DefaultSelector()
    active = []

    def assign(worker):
        if not pending:
            return False
        task = pending.popleft()
        worker["inflight"] = task
        worker["proc"].stdin.write(task[1] + "\n")
        worker["proc"].stdin.flush()
        return True

    for wid in range(workers):
        worker = start_worker(wid, binary, root, log_dir)
        active.append(worker)
        selector.register(worker["proc"].stdout, selectors.EVENT_READ, worker)
        assign(worker)

    started = time.time()
    last_report = started
    try:
        while completed < total:
            events = selector.select(timeout=5.0)
            if not events:
                now = time.time()
                if now - last_report >= 30:
                    rate = (completed - len(done)) / max(1e-9, now - started)
                    print(
                        f"progress {completed}/{total} pending={len(pending)} "
                        f"rate={rate:.1f}/s elapsed={now-started:.1f}s",
                        flush=True,
                    )
                    last_report = now
                for worker in list(active):
                    if worker["proc"].poll() is None or worker["inflight"] is None:
                        continue
                    task = worker["inflight"]
                    if retries[task[0]] < 2:
                        retries[task[0]] += 1
                        pending.appendleft(task)
                    else:
                        print(f"failed task after retries: {task[0]}", file=sys.stderr, flush=True)
                        completed += 1
                    selector.unregister(worker["proc"].stdout)
                    worker["log"].close()
                    active.remove(worker)
                    new_worker = start_worker(worker["id"], binary, root, log_dir)
                    active.append(new_worker)
                    selector.register(new_worker["proc"].stdout, selectors.EVENT_READ, new_worker)
                    assign(new_worker)
                continue

            for key, _ in events:
                worker = key.data
                line = worker["proc"].stdout.readline()
                if not line:
                    task = worker["inflight"]
                    if task and retries[task[0]] < 2:
                        retries[task[0]] += 1
                        pending.appendleft(task)
                    selector.unregister(worker["proc"].stdout)
                    worker["log"].close()
                    active.remove(worker)
                    new_worker = start_worker(worker["id"], binary, root, log_dir)
                    active.append(new_worker)
                    selector.register(new_worker["proc"].stdout, selectors.EVENT_READ, new_worker)
                    assign(new_worker)
                    continue

                out.write(line)
                completed += 1
                worker["inflight"] = None
                assign(worker)
    finally:
        for worker in list(active):
            stop_worker(worker)
        out.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--binary", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    binary = args.binary or (root / "build" / "eval_fabo_sweep_worker")
    if not binary.exists():
        raise SystemExit(f"missing binary: {binary}")
    if not args.tasks.exists():
        raise SystemExit(f"missing task TSV: {args.tasks}")
    run_queue(root, binary, args.tasks, args.out, args.workers, args.resume)


if __name__ == "__main__":
    main()
