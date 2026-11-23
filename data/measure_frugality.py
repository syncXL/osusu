#!/usr/bin/env python
"""Frugality measurement of the "Frugal Tonal ASR for Fongbe" competition: the script the organisers run on the
top 20 of the private leaderboard, provided so that every team can check its compliance before submitting.

    python measure_frugality.py \
        --weights my_system/ --weights lm.bin \
        --command "python infer.py --csv {csv} --root {root} --output {output}" \
        --csv Test.csv [--limit 300] [--report report.json]

Your command reads a CSV (`id, audio_path`, paths relative to `{root}`) and writes `{output}` (`id, transcription`,
or the submission format `ID, Target`).
What the script measures, on the complete system, from audio to tonal transcription:

  size        sum of the files given with --weights: ALL the weights (pre-trained models and language model included,
              after any quantisation), threshold 1,000 MB (MB = 10^6 bytes)
  RTF         elapsed time of the command / total audio duration, on ONE single CPU core, threshold 1
  memory      peak resident memory of the command and all its sub-processes, threshold 2 GB (GB = 10^9 bytes)
  offline     the command runs with the network cut (Linux, root rights: `unshare -n`) and without any GPU

If the CSV has a `transcription` column (for example the `recommended_validation` voices of Train_reliable.csv),
the official score is recomputed with `score.py`. The organisers recompute it on the full test set.

Conditions of the official check: recent x86-64 Linux machine, a single thread (CPU affinity + *_NUM_THREADS variables),
no GPU, network cut; model loading counts in the measured time. The thresholds are those of the competition rules;
exceeding one excludes a team from the prizes, it does not change the leaderboard.
"""
import argparse
import csv
import json
import os
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

MAX_SIZE = 1000 * 10 ** 6       # 1,000 MB
MAX_RTF = 1.0
MAX_MEMORY = 2 * 10 ** 9        # 2 GB
THREADS = {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                            "VECLIB_MAXIMUM_THREADS", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "ONNXRUNTIME_NUM_THREADS")}
OFFLINE = {"CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
NETWORK_NOT_CUT = "network NOT cut"


def weights_size(paths):
    seen, total, detail = set(), 0, {}
    for c in paths:
        p = Path(c)
        files = [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()] if p.is_dir() else []
        if not files:
            sys.exit(f"--weights not found or empty: {c}")
        t = 0
        for f in files:
            key = f.resolve()
            if key not in seen:                    # the same file (link) is only counted once
                seen.add(key); t += f.stat().st_size
        detail[str(c)] = t; total += t
    return total, detail


def wav_duration(path):
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def tree_memory(pid):
    """Resident memory (bytes) of the process and all its descendants, read from /proc (Linux)."""
    total, todo = 0, [pid]
    while todo:
        p = todo.pop()
        try:
            for line in open(f"/proc/{p}/status"):
                if line.startswith("VmRSS:"):
                    total += int(line.split()[1]) * 1024
                    break
            for task in os.listdir(f"/proc/{p}/task"):
                with open(f"/proc/{p}/task/{task}/children") as f:
                    todo.extend(int(x) for x in f.read().split())
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return total


def run(command, env, one_core, no_network):
    prefix, notes = [], []
    if no_network:
        if shutil.which("unshare") and os.geteuid() == 0:
            prefix = ["unshare", "-n"]
        else:
            notes.append(f"{NETWORK_NOT_CUT} (Linux, `unshare` and root rights are needed): check yourself that no network call is made")
    core = None
    if one_core and hasattr(os, "sched_setaffinity"):
        core = min(os.sched_getaffinity(0))
    elif one_core:
        notes.append("single core NOT enforced (Linux needed): only the *_NUM_THREADS variables limit the threads")

    def before():
        if core is not None:
            os.sched_setaffinity(0, {core})

    t0 = time.perf_counter()
    proc = subprocess.Popen(prefix + ["/bin/sh", "-c", command], env=env, preexec_fn=before if os.name == "posix" else None)
    peak = 0
    while proc.poll() is None:
        if os.path.isdir("/proc"):
            peak = max(peak, tree_memory(proc.pid))
        time.sleep(0.05)
    elapsed = time.perf_counter() - t0
    # complement: the largest descendant process seen by the kernel (ru_maxrss in KiB on Linux)
    peak = max(peak, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024)
    return proc.returncode, elapsed, peak, notes, core


def main():
    ap = argparse.ArgumentParser(description="Frugality measurement (size, RTF on one core, peak memory, offline).")
    ap.add_argument("--weights", action="append", required=True, help="weights folder or file; repeat for each one")
    ap.add_argument("--command", required=True, help="inference command with {csv}, {root} and {output}")
    ap.add_argument("--csv", required=True, help="CSV with id, audio_path (and transcription to recompute the score)")
    ap.add_argument("--root", help="folder the audio_path paths are relative to (default: the folder of the CSV)")
    ap.add_argument("--limit", type=int, help="only measure the first N utterances (quick check)")
    ap.add_argument("--report", help="JSON file for the report")
    ap.add_argument("--allow-network", action="store_true", help="do not cut the network (not recommended)")
    a = ap.parse_args()

    source = Path(a.csv).resolve()
    root = Path(a.root).resolve() if a.root else source.parent
    with open(source, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if a.limit:
        rows = rows[:a.limit]
    if not rows or not {"id", "audio_path"} <= set(rows[0]):
        sys.exit("the CSV needs at least the columns id and audio_path")
    audio_total = sum(wav_duration(root / r["audio_path"]) for r in rows)

    size, detail = weights_size(a.weights)
    work = Path(tempfile.mkdtemp(prefix="frugality-"))
    input_csv, output = work / "input.csv", work / "submission.csv"
    with open(input_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "audio_path"]); w.writerows((r["id"], r["audio_path"]) for r in rows)
    command = a.command.format(csv=shlex.quote(str(input_csv)), root=shlex.quote(str(root)), output=shlex.quote(str(output)))
    env = dict(os.environ, **THREADS, **OFFLINE)
    print(f"{len(rows)} utterances, {audio_total / 60:.1f} min of audio; weights: {size / 1e6:.1f} MB; running…", flush=True)
    code, elapsed, peak, notes, core = run(command, env, True, not a.allow_network)

    report = {"utterances": len(rows), "audio_s": round(audio_total, 2), "command": a.command, "return_code": code,
              "size_bytes": size, "size_detail": detail, "time_s": round(elapsed, 2),
              "RTF": round(elapsed / audio_total, 4) if audio_total else None, "peak_memory_bytes": peak,
              "core": core, "threads": THREADS, "notes": notes}
    if code != 0 or not output.exists():
        report["error"] = f"the command failed (code {code}) or did not write {output}"
    else:
        with open(output, encoding="utf-8-sig", newline="") as f:
            sub = {(r["ID"] if "ID" in r else r["id"]): (r.get("Target") if "ID" in r else r.get("transcription")) or ""
                   for r in csv.DictReader(f)}
        ids = [r["id"] for r in rows]
        report["submission"] = {"missing": sum(i not in sub for i in ids), "extra": len(set(sub) - set(ids)),
                                "empty": sum(not sub.get(i, "").strip() for i in ids)}
        if "transcription" in rows[0]:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from score import measure
            s = measure([(r["transcription"], sub.get(r["id"], "")) for r in rows])
            report["score"] = {k: s[k] for k in ("score", "TSER", "TER_conditional")}

    def verdict(value, limit):
        return "not measured" if value is None else ("compliant" if value <= limit else "NOT COMPLIANT")
    report["verdict"] = {"size": verdict(size, MAX_SIZE), "RTF": verdict(report["RTF"], MAX_RTF),
                         "memory": verdict(peak, MAX_MEMORY),
                         "offline": "not checked" if any(NETWORK_NOT_CUT in n for n in notes) else "network cut"}
    print(f"size     {size / 1e6:9.1f} MB   (<= {MAX_SIZE / 1e6:.0f})   {report['verdict']['size']}")
    print(f"RTF      {report['RTF']:9.3f}      (<= {MAX_RTF:g})       {report['verdict']['RTF']}   [{elapsed:.1f} s for {audio_total:.1f} s of audio]")
    print(f"memory   {peak / 1e9:9.3f} GB   (<= {MAX_MEMORY / 1e9:.0f})       {report['verdict']['memory']}")
    print(f"network  {report['verdict']['offline']}")
    if "score" in report:
        print(f"score    {report['score']['score']:.4f}  (TSER {report['score']['TSER']:.4f}, conditional TER {report['score']['TER_conditional']:.4f})")
    for n in notes + ([report["error"]] if "error" in report else []):
        print("WARNING:", n)
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    sys.exit(0 if "error" not in report and all(v in ("compliant", "network cut") for v in report["verdict"].values()) else 1)


if __name__ == "__main__":
    main()
