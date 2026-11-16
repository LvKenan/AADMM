#!/usr/bin/env python3
"""C1 driver.  Usage (from the C1 folder):
    python run_c1.py env                 # write C1_out/env.txt (environment record)
    python run_c1.py list [--group G]    # list tasks and their status (done / todo)
    python run_c1.py smoke               # < 2 min self-test (sanity checks + tiny runs)
    python run_c1.py run [--group G]     # run all unfinished tasks (resumable; G = all|ref|nonref|main|sweep)
    python run_c1.py status              # progress summary
    python run_c1.py checkpoint          # write C1_partial_v12_latest.zip now (also done automatically during 'run')
    python run_c1.py pack                # strict figures/tables + zip everything to C1_results_v12_<date>.zip
Single-threaded BLAS is enforced below BEFORE numpy is imported (reproducible timings)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import io
import json
import time
import glob
import platform
import argparse
import datetime
import subprocess
import traceback
import contextlib
import zipfile
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np            # noqa: E402
import scipy                  # noqa: E402
from c1 import __version__    # noqa: E402
from c1.tasks import build_tasks, smoke_tasks, load_config, ref_time_budget   # noqa: E402
from c1.problems import make_problem                         # noqa: E402
from c1.reference import reference_solve                     # noqa: E402
from c1.metrics import Monitor, FIELDS, LEVELS               # noqa: E402
from c1.methods import METHODS                               # noqa: E402

OUT = os.path.join(HERE, "C1_out")


def package_hash():
    h = hashlib.sha256()
    for p in [os.path.join(HERE, "run_c1.py")] + sorted(glob.glob(os.path.join(HERE, "c1", "*.py"))) + [os.path.join(HERE, "c1", "config.json")]:
        with open(p, "rb") as f:
            h.update(os.path.relpath(p, HERE).encode() + b"\0" + f.read())
    return h.hexdigest()


PACKAGE_HASH = package_hash()


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec="seconds")


def log(msg, out):
    line = f"[{now()}] {msg}"
    print(line, flush=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    with open(os.path.join(out, "logs", "run_log.txt"), "a") as fh:
        fh.write(line + "\n")


def result_path(out, tid):
    return os.path.join(out, "results", tid + ".npz")


def is_done(out, tid):
    p = result_path(out, tid)
    if not os.path.exists(p):
        return False
    try:
        with np.load(p, allow_pickle=False) as z:
            m = json.loads(str(z["meta"]))
        if m.get("version") != __version__ or m.get("package_sha256") != PACKAGE_HASH:
            raise ValueError(f"Incompatible existing result: {p}. Use a fresh C1-1.2 output directory; do not mix versions/code/config.")
        return True
    except (OSError, EOFError, zipfile.BadZipFile) as e:
        raise ValueError(f"Unreadable result {p}; preserve it for diagnosis and use a fresh directory: {e}") from e


def save_atomic(out, tid, **arrays):
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    final = result_path(out, tid)
    tmp = final + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, final)


def run_task(task, cfg, out, run_cfg=None, ref_cfg=None):
    run_cfg = run_cfg or cfg["run"]
    ref_cfg = ref_cfg or cfg["reference"]
    t0 = time.perf_counter()
    prob = make_problem(task["spec"])
    gen_time = time.perf_counter() - t0
    meta = dict(task=task, problem=prob.info(), gen_time=gen_time, version=__version__, package_sha256=PACKAGE_HASH,
                numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version())
    if task["kind"] == "ref":
        budget = float(ref_cfg.get("time_budget_by_problem", {}).get(task["P"], ref_cfg["time_budget"]))
        r = reference_solve(prob, tol=ref_cfg["tol"], maxit=ref_cfg["maxit"], time_budget=budget)
        meta.update(dict(Fstar=r["Fstar"], LB=r["LB"], UB=r["UB"], gap_rel=r["gap_rel"], iters=r["iters"],
                         status=r["status"], time=r["time"], ref_cfg=dict(ref_cfg, time_budget_used=budget)))
        save_atomic(out, task["id"], meta=json.dumps(meta, default=float), lam_star=r["lam_star"],
                    y_star=r["y_star"], By_star=r["By_star"], hist=r["hist"])
        return f"ref {r['status']} iters={r['iters']} gap_rel={r['gap_rel']:.1e} {r['time']:.1f}s"
    prob.cnt.reset()
    mon = Monitor(prob, run_cfg)
    METHODS[task["method"]](prob, mon, **task["params"])
    H = mon.result()
    first = {str(k): (list(v) if v is not None else None) for k, v in mon.first_hit.items()}
    meta.update(dict(stop_reason=mon.stop_reason, iters=mon.k, t_alg=mon.t_alg,
                     t_wall=time.perf_counter() - t0, first_hit=first, fields=FIELDS, run_cfg=run_cfg,
                     method_settings=getattr(mon, "method_meta", {}), run_summary=mon.summary(),
                     label=task.get("label"), theory_scope=task.get("theory_scope"), variant=task.get("variant")))
    tr = mon.traces()
    save_atomic(out, task["id"], meta=json.dumps(meta, default=float), hist=H,
                kkt_all=tr["kkt_all"], pen_all=tr["pen_all"], lpred_all=tr["lpred_all"], restart_iters=tr["restart_iters"])
    return f"{mon.stop_reason} it={mon.k} KKT={mon.last_kkt:.1e} t_alg={mon.t_alg:.2f}s"


def cmd_env(out):
    os.makedirs(out, exist_ok=True)
    buf = io.StringIO()
    p = lambda *a: print(*a, file=buf)
    p("C1 environment record", now())
    p("package version:", __version__)
    p("package sha256:", PACKAGE_HASH)
    p("python:", sys.version.replace("\n", " "))
    p("platform:", platform.platform())
    p("numpy:", np.__version__, " scipy:", scipy.__version__)
    p("thread env:", {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    try:
        import threadpoolctl
        p("threadpoolctl:", json.dumps(threadpoolctl.threadpool_info(), indent=1))
    except Exception:
        p("threadpoolctl: not installed")
    p("os.cpu_count():", os.cpu_count())
    try:
        with open("/proc/cpuinfo") as fh:
            models = sorted({l.split(":", 1)[1].strip() for l in fh if l.startswith("model name")})
        p("cpu model:", models)
    except Exception:
        p("cpu model: n/a")
    try:
        with open("/proc/meminfo") as fh:
            p("mem:", " | ".join(fh.read().splitlines()[:3]))
    except Exception:
        pass
    colab = "google.colab" in sys.modules
    try:
        import importlib.util
        colab = colab or importlib.util.find_spec("google.colab") is not None
    except Exception:
        pass
    p("google.colab importable:", colab)
    p("COLAB runtime:", {k: os.environ.get(k) for k in ("COLAB_RELEASE_TAG", "COLAB_IMAGE_TYPE", "COLAB_BACKEND_VERSION", "COLAB_GPU")})
    try:
        g = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        p("nvidia-smi:", (g.stdout or g.stderr).strip() or "none")
    except Exception:
        p("nvidia-smi: not available (CPU runtime)")
    p("\nnumpy.show_config():")
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            np.show_config()
        except Exception as e:
            print("show_config failed:", e)
    p(f.getvalue())
    txt = buf.getvalue()
    with open(os.path.join(out, "env.txt"), "w") as fh:
        fh.write(txt)
    return txt


def select(tasks, group, only):
    if group == "nonref":
        tasks = [t for t in tasks if t["group"] != "ref"]
    elif group and group != "all":
        tasks = [t for t in tasks if t["group"] in (group, "ref")]
        if group != "ref":
            needed = {t["inst"] for t in tasks if t["group"] != "ref"}
            tasks = [t for t in tasks if t["group"] != "ref" or t["inst"] in needed]
    if only:
        tasks = [t for t in tasks if only in t["id"]]
    return tasks


def cmd_list(out, group, only):
    tasks = select(build_tasks(), group, only)
    for t in tasks:
        print(("done " if is_done(out, t["id"]) else "todo ") + t["id"])
    print(f"{sum(is_done(out, t['id']) for t in tasks)}/{len(tasks)} done")


def cmd_status(out):
    tasks = build_tasks()
    by = {}
    for t in tasks:
        d = by.setdefault(t["group"], [0, 0])
        d[1] += 1
        d[0] += is_done(out, t["id"])
    for g, (a, b) in by.items():
        print(f"{g:6s}: {a}/{b} done")
    fails = glob.glob(os.path.join(out, "logs", "FAILED__*.txt"))
    print(f"failed tasks logged: {len(fails)}")
    for f in fails:
        print("  ", os.path.basename(f))


def cmd_run(out, group, only, cfg=None, tasks=None, run_cfg=None, ref_cfg=None):
    cfg = cfg or load_config()
    tasks = tasks if tasks is not None else select(build_tasks(cfg), group, only)
    os.makedirs(out, exist_ok=True)
    if not os.path.exists(os.path.join(out, "env.txt")):
        cmd_env(out)
    todo = [t for t in tasks if not is_done(out, t["id"])]
    log(f"run: {len(tasks)} tasks selected, {len(tasks) - len(todo)} already done, {len(todo)} to do", out)
    t_start = time.time()
    nfail = 0
    ck = {} if os.path.basename(os.path.abspath(out)) == "C1_out_smoke" else cfg.get("checkpoint", {})
    last_ck, n_since = time.time(), 0
    for j, t in enumerate(todo, 1):
        if is_done(out, t["id"]):
            continue
        t1 = time.time()
        try:
            msg = run_task(t, cfg, out, run_cfg, ref_cfg)
            log(f"[{j}/{len(todo)}] OK   {t['id']}  {msg}  ({time.time() - t1:.1f}s, elapsed {(time.time() - t_start) / 60:.1f} min)", out)
            fl = os.path.join(out, "logs", f"FAILED__{t['id']}.txt")
            if os.path.exists(fl):
                os.remove(fl)
        except KeyboardInterrupt:
            log("interrupted by user; finished tasks are kept, rerun 'run' to resume", out)
            raise
        except Exception:
            nfail += 1
            tb = traceback.format_exc()
            os.makedirs(os.path.join(out, "logs"), exist_ok=True)
            with open(os.path.join(out, "logs", f"FAILED__{t['id']}.txt"), "w") as fh:
                fh.write(tb)
            log(f"[{j}/{len(todo)}] FAIL {t['id']}  (traceback in logs/FAILED__{t['id']}.txt)", out)
        n_since += 1
        if ck and (n_since >= ck.get("every_tasks", 25) or time.time() - last_ck >= 60 * ck.get("every_minutes", 15.0)):
            try:
                cmd_checkpoint(out, quiet=True)
            except Exception as e:      # a checkpoint failure must never stop the run
                log(f"checkpoint failed: {e!r}", out)
            last_ck, n_since = time.time(), 0
    if ck and n_since > 0:
        try:
            cmd_checkpoint(out, quiet=True)
        except Exception as e:
            log(f"checkpoint failed: {e!r}", out)
    done = sum(is_done(out, t["id"]) for t in tasks)
    log(f"run finished: {done}/{len(tasks)} done, {nfail} failed in this session, {(time.time() - t_start) / 60:.1f} min", out)
    return nfail


def cmd_smoke():
    t0 = time.time()
    print("== smoke 1/3: sanity checks (tests/sanity.py) ==", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, "tests", "sanity.py")], cwd=HERE)
    ok_sanity = (r.returncode == 0)
    print("== smoke 2/3: tiny runs of every method on every problem ==", flush=True)
    out = os.path.join(HERE, "C1_out_smoke")
    if os.path.isdir(out):
        import shutil
        shutil.rmtree(out)
    cfg = load_config()
    run_cfg = dict(cfg["run"], maxit=3000, time_budget=20.0)
    ref_cfg = dict(cfg["reference"], time_budget=20.0, time_budget_by_problem={})
    nfail = cmd_run(out, None, None, cfg, smoke_tasks(), run_cfg, ref_cfg)
    print("== smoke 3/3: pack ==", flush=True)
    z = cmd_pack(out, prefix="C1_smoke_v12")
    ok = ok_sanity and nfail == 0 and os.path.exists(z)
    print(f"\nSMOKE {'PASS' if ok else 'FAIL'}  ({time.time() - t0:.0f}s)  sanity={'ok' if ok_sanity else 'FAILED'}  "
          f"failed_tasks={nfail}  zip={os.path.basename(z)}")
    return 0 if ok else 1


def _zip_dir(out, zpath, extra_files=()):
    tmp = f"{zpath}.{os.getpid()}.tmp"
    n = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(out):
            for f in sorted(files):
                if f.endswith(".tmp.npz") or f.endswith(".tmp"):
                    continue
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, os.path.dirname(out)))
                n += 1
        for full in extra_files:
            if os.path.exists(full):
                zf.write(full, os.path.join(os.path.basename(out), os.path.basename(full)))
    os.replace(tmp, zpath)       # atomic: an interrupted checkpoint never leaves a broken zip
    return n


def _manifest(out, tasks):
    done = [t["id"] for t in tasks if is_done(out, t["id"])]
    fails = sorted(os.path.basename(f)[8:-4] for f in glob.glob(os.path.join(out, "logs", "FAILED__*.txt")))
    manifest = dict(created=now(), version=__version__, package_sha256=PACKAGE_HASH, n_tasks=len(tasks), n_done=len(done),
                    done=done, missing=[t["id"] for t in tasks if not is_done(out, t["id"])], failed_logged=fails)
    tmp = os.path.join(out, f"manifest.json.{os.getpid()}.tmp")
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=1)
    os.replace(tmp, os.path.join(out, "manifest.json"))
    return manifest


def cmd_checkpoint(out, quiet=False):
    """Partial results zip next to the output folder (overwritten each time).  If the Colab runtime
    disconnects, download this file or simply rerun 'run' (finished tasks are skipped)."""
    if not os.path.isdir(out):
        return ""
    m = _manifest(out, build_tasks())
    zpath = os.path.join(os.path.dirname(os.path.abspath(out)), "C1_partial_v12_latest.zip")
    _zip_dir(out, zpath)
    msg = f"checkpoint: {m['n_done']}/{m['n_tasks']} tasks -> {zpath}"
    log(msg, out) if quiet else print(msg)
    return zpath


def _run_summary(out, tasks):
    """Counts of stop reasons and reference status, written to <out>/run_summary.json and .md."""
    stops, refs, rows = {}, {}, []
    for t in tasks:
        if not is_done(out, t["id"]):
            continue
        z = np.load(result_path(out, t["id"]), allow_pickle=False)
        m = json.loads(str(z["meta"]))
        if t["kind"] == "ref":
            refs.setdefault(m["status"], []).append(f"{t['inst']} gap_rel={m['gap_rel']:.2e} t={m['time']:.0f}s")
        else:
            key = (t["group"], t["method"], t.get("variant", "default"), m["stop_reason"])
            stops[key] = stops.get(key, 0) + 1
            rows.append(m["t_wall"])
    summ = dict(version=__version__, created=now(), n_tasks=len(tasks), n_done=sum(is_done(out, t["id"]) for t in tasks),
                ref_status={k: len(v) for k, v in refs.items()},
                refs_not_tol=[x for k, v in refs.items() if k != "tol" for x in v],
                stop_reasons=[dict(group=k[0], method=k[1], variant=k[2], stop=k[3], n=v) for k, v in sorted(stops.items())],
                method_wall_minutes=sum(rows) / 60.0)
    with open(os.path.join(out, "run_summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1)
    with open(os.path.join(out, "run_summary.md"), "w") as fh:
        fh.write(f"# C1 run summary ({__version__}, {summ['created']})\n\n")
        fh.write(f"tasks done: {summ['n_done']}/{summ['n_tasks']}\n\nreference status: {summ['ref_status']}\n\n")
        if summ["refs_not_tol"]:
            fh.write("references NOT within tolerance:\n" + "".join(f"- {x}\n" for x in summ["refs_not_tol"]) + "\n")
        fh.write("| group | method | variant | stop | n |\n|---|---|---|---|---|\n")
        for r in summ["stop_reasons"]:
            fh.write(f"| {r['group']} | {r['method']} | {r['variant']} | {r['stop']} | {r['n']} |\n")
    return summ


def cmd_pack(out, prefix="C1_results_v12"):
    if not os.path.isdir(out):
        print("nothing to pack:", out)
        return ""
    cmd_env(out)    # refresh env.txt (records the runtime that did the packing)
    tasks = build_tasks() if prefix.startswith("C1_results") else smoke_tasks()
    manifest = _manifest(out, tasks)
    summ = _run_summary(out, tasks)
    figs = os.path.join(out, "figs")
    plot_log = os.path.join(out, "logs", "plot_log.txt")
    if prefix.startswith("C1_results"):
        import shutil
        if os.path.isdir(figs):
            shutil.rmtree(figs)       # never ship figures from an earlier plotting attempt
        # strict plotting: fails if any reference missed its tolerance or is missing
        r = subprocess.run([sys.executable, os.path.join(HERE, "plot_c1.py"), "--res", out, "--fig", figs],
                           capture_output=True, text=True)
        with open(plot_log, "w") as fh:
            fh.write("== strict plot_c1.py ==\n" + r.stdout + r.stderr)
        manifest["strict_plot_ok"] = r.returncode == 0
        with open(os.path.join(out, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)
        if r.returncode != 0 and os.path.isdir(figs):
            shutil.rmtree(figs)       # no partial figures after a strict failure
        if r.returncode != 0:
            print("STRICT PLOTTING FAILED: see logs/plot_log.txt. No fallback or relaxation of tolerances. "
                  "The results zip will still be written for diagnosis; report this as a failure.")
        else:
            print("strict figures and tables written to", figs)
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y%m%d_%H%M")  # JST
    zpath = os.path.join(HERE, f"{prefix}_{stamp}.zip")
    provenance = os.path.join(out, "provenance")
    os.makedirs(provenance, exist_ok=True)
    import shutil
    for name in ("c1/config.json", "RUN_LIST.json", "CHANGELOG.md"):
        src = os.path.join(HERE, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(provenance, os.path.basename(src)))
    _zip_dir(out, zpath)
    print(f"packed {manifest['n_done']}/{len(tasks)} finished tasks -> {zpath}")
    if manifest["missing"]:
        print(f"WARNING: {len(manifest['missing'])} tasks missing (listed in manifest.json)")
    if summ["refs_not_tol"]:
        print(f"WARNING: {len(summ['refs_not_tol'])} reference solves not within tolerance (run_summary.md)")
    return zpath


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["env", "list", "smoke", "run", "status", "checkpoint", "pack"])
    ap.add_argument("--group", default="all", help="all | ref | nonref | main | sweep (default all)")
    ap.add_argument("--only", default=None, help="substring filter on task ids (debug only)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--shard", default=None, help=argparse.SUPPRESS)   # i/n, internal use only
    a = ap.parse_args()
    if a.cmd == "env":
        print(cmd_env(a.out))
    elif a.cmd == "list":
        cmd_list(a.out, a.group, a.only)
    elif a.cmd == "status":
        cmd_status(a.out)
    elif a.cmd == "smoke":
        sys.exit(cmd_smoke())
    elif a.cmd == "run":
        tasks = None
        if a.shard:
            i, n = map(int, a.shard.split("/"))
            tasks = [t for j, t in enumerate(select(build_tasks(), a.group, a.only)) if j % n == i]
        sys.exit(1 if cmd_run(a.out, a.group, a.only, tasks=tasks) else 0)
    elif a.cmd == "checkpoint":
        cmd_checkpoint(a.out)
    elif a.cmd == "pack":
        z = cmd_pack(a.out)
        with open(os.path.join(a.out, "manifest.json")) as f:
            m = json.load(f)
        sys.exit(0 if z and not m["missing"] and m.get("strict_plot_ok", False) else 2)


if __name__ == "__main__":
    main()
