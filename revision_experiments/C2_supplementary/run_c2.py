#!/usr/bin/env python3
"""C2_supp driver.  Usage (from the C2_supp folder):
    python run_c2.py env                 # write C2_out/env.txt (environment record)
    python run_c2.py list [--group G]    # task status (done / todo)
    python run_c2.py smoke               # < 3 min self-test (sanity checks + tiny runs + pack)
    python run_c2.py run [--group G]     # run unfinished tasks (resumable; G = all|ref|a_struct|a_range|b)
    python run_c2.py status              # progress summary
    python run_c2.py checkpoint          # write C2_partial_latest.zip now (also automatic during 'run')
    python run_c2.py pack                # strict figures/tables + zip -> C2_results_<date>.zip
Single-threaded BLAS is enforced below BEFORE numpy is imported (reproducible timings)."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, io, json, time, glob, platform, argparse, datetime, subprocess, traceback, contextlib, zipfile, hashlib, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np            # noqa: E402
import scipy                  # noqa: E402
from c2 import VERSION as __version__                         # noqa: E402
from c2.tasks import build_tasks, smoke_tasks, load_config     # noqa: E402
from c2.problems import InnerProblem, lasso_reference          # noqa: E402
from c2.refs import c1_ref, tvd_ref                            # noqa: E402
from c2.methods import METHODS_A, METHODS_B, InexactMonitor, run_aadmm_c2, run_dual_fista   # noqa: E402
from c1.problems import make_problem                           # noqa: E402
from c1.metrics import Monitor, FIELDS                         # noqa: E402

OUT = os.path.join(HERE, "C2_out")
CODE_FILES = ["run_c2.py", "plot_c2.py", "toy_fig.py"]


def package_hash():
    h = hashlib.sha256()
    files = [os.path.join(HERE, f) for f in CODE_FILES] + sorted(glob.glob(os.path.join(HERE, "c1", "*.py"))) \
        + sorted(glob.glob(os.path.join(HERE, "c2", "*.py"))) + [os.path.join(HERE, "c1", "config.json"),
                                                               os.path.join(HERE, "c2", "config.json"), os.path.join(HERE, "refs", "c1v12_refs.npz")]
    for p in files:
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
    except (OSError, EOFError, zipfile.BadZipFile, KeyError, ValueError) as e:
        raise ValueError(f"Unreadable result {p}; preserve it for diagnosis and use a fresh directory: {e}") from e
    if m.get("version") != __version__ or m.get("package_sha256") != PACKAGE_HASH:
        raise ValueError(f"Incompatible existing result: {p}. Use a fresh output directory; do not mix versions/code/config.")
    return True


def save_atomic(out, tid, **arrays):
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    final = result_path(out, tid)
    tmp = final + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, final)


def _base_meta(task):
    return dict(task=task, version=__version__, package_sha256=PACKAGE_HASH, numpy=np.__version__, scipy=scipy.__version__,
                python=platform.python_version(), started=now())


def _lasso_ref_from_out(out, iid):
    p = result_path(out, f"ref__{iid}")
    if not os.path.exists(p):
        raise RuntimeError(f"LASSO reference ref__{iid} not finished yet: run the 'ref' group first (the task will be retried)")
    z = np.load(p, allow_pickle=False); m = json.loads(str(z["meta"]))
    if m["status"] != "tol":
        raise RuntimeError(f"LASSO reference ref__{iid} not within tolerance ({m['status']}); no fallback")
    return dict(pstar=m["pstar"], lam_star=z["lam_star"], By_star=z["By_star"], source=f"ref__{iid}")


def run_task(task, cfg, out, smoke=False):
    t0 = time.perf_counter()
    meta = _base_meta(task)
    kind = task["kind"]
    if kind == "ref":
        P = InnerProblem(task["spec"])
        rc = cfg["lasso_reference"]
        r = lasso_reference(P, tol=rc["tol"], maxit=rc["maxit"], time_budget=(60.0 if smoke else rc["time_budget"]))
        meta.update({k: r[k] for k in ("pstar", "LB", "UB", "gap_rel", "iters", "status", "time")}, tau=P.tau, LA=P.LA, ref_cfg=rc)
        save_atomic(out, task["id"], meta=json.dumps(meta, default=float), lam_star=r["lam_star"], By_star=r["By_star"], x_star=r["x_star"])
        return f"ref {r['status']} iters={r['iters']} gap_rel={r['gap_rel']:.1e} {r['time']:.1f}s"
    if kind in ("a_struct", "a_range"):
        prob = make_problem(task["spec"])
        ref = c1_ref(task["inst"], task["spec"], allow_compute=smoke)
        meta.update(problem=prob.info(), ref=dict(Fstar=ref["Fstar"], gap_rel=ref["gap_rel"], source=ref["source"],
                                                   norm_lam_star=float(np.linalg.norm(ref["lam_star"]))))
        beta = task["params"]["beta"]
        run_cfg = dict(cfg["a_run"])
        div_bound = run_cfg.pop("div_factor") * (1.0 + float(np.linalg.norm(ref["lam_star"])))
        if kind == "a_struct":
            K = cfg["a_struct"]["K"] if not smoke else 60
            rc = dict(run_cfg, tol=0.0, maxit=K)
            trA, trF = [], []
            mA = Monitor(prob, rc); run_aadmm_c2(prob, mA, beta=beta, init="compatible", div_bound=div_bound, trace=trA)
            prob2 = make_problem(task["spec"])
            mF = Monitor(prob2, rc); run_dual_fista(prob2, mF, beta=beta, div_bound=div_bound, trace=trF)
            n = min(len(trA), len(trF)); sc = 1.0 + float(np.linalg.norm(ref["lam_star"]))
            dev = np.array([np.linalg.norm(trA[i]["lt"] - trF[i]["eta"]) / sc for i in range(n)])
            # Lemma lem:srec (c) second line and (b): identity residual, size of the extra (beta-1) beta r^k term
            ident, extra, sarg_dev = [], [], []
            for i in range(1, n):
                tk, tk1 = trA[i]["t_k"], trA[i]["t_k1"]; th = (tk - 1.0) / tk1
                pred = (1 + th) * trA[i]["u"] - th * trA[i - 1]["u"]
                ex = (tk / tk1) * (beta - 1.0) * beta * trA[i]["r"]
                ident.append(np.linalg.norm(trA[i]["s_next"] - pred - ex) / (1 + np.linalg.norm(trA[i]["s_next"])))
                extra.append(np.linalg.norm(ex) / sc)
                sarg_dev.append(np.linalg.norm(trA[i]["s_next"] - (trF[i + 1]["sarg"] if i + 1 < len(trF) else np.nan)) / sc)
            sb = lambda k: np.linalg.norm(trA[k]["s"] - (trA[k]["u"] + beta * trA[k]["r"])) / (1 + np.linalg.norm(trA[k]["s"]))
            lemb = np.array([sb(i) for i in range(n)])
            meta.update(stop_A=mA.stop_reason, stop_F=mF.stop_reason, iters_A=mA.k, iters_F=mF.k, fields=FIELDS,
                        max_identity_resid=float(np.max(ident)) if ident else None, max_lemma_b_resid=float(np.max(lemb)),
                        dev_final=float(dev[-1]), dev_max=float(dev.max()))
            save_atomic(out, task["id"], meta=json.dumps(meta, default=float), dev=dev, ident=np.array(ident), extra=np.array(extra),
                        sarg_dev=np.array(sarg_dev), lemma_b=lemb, hist_A=mA.result(), hist_F=mF.result(),
                        lpred_A=np.asarray(mA.lpred_all))
            return f"a_struct dev_final={dev[-1]:.1e} dev_max={dev.max():.1e} ident={meta['max_identity_resid']:.1e}"
        if smoke:
            run_cfg.update(maxit=600, time_budget=60.0)
        mon = Monitor(prob, run_cfg)
        METHODS_A[task["method"]](prob, mon, beta=beta, div_bound=div_bound)
        H = mon.result()
        meta.update(stop_reason=mon.stop_reason, iters=mon.k, t_alg=mon.t_alg, t_wall=time.perf_counter() - t0, fields=FIELDS,
                    run_cfg=run_cfg, div_bound=div_bound, run_summary=mon.summary(), label=task.get("label"),
                    theory_scope=task.get("theory_scope"))
        tr = mon.traces()
        save_atomic(out, task["id"], meta=json.dumps(meta, default=float), hist=H, kkt_all=tr["kkt_all"], lpred_all=tr["lpred_all"])
        return f"{mon.stop_reason} it={mon.k} KKT={mon.last_kkt:.1e} t_alg={mon.t_alg:.2f}s"
    if kind == "b":
        P = InnerProblem(task["spec"])
        ref = _lasso_ref_from_out(out, task["inst"]) if P.kind == "LASSO" else tvd_ref(P, allow_compute=smoke)
        run_cfg = dict(cfg["b_run"])
        if smoke:
            run_cfg.update(maxit=60, time_budget=60.0)
        beta = task["params"]["beta"]
        mon = InexactMonitor(P, ref, run_cfg, beta)
        METHODS_B[task["method"]](P, mon, beta=beta, sched=task["params"]["sched"])
        H = mon.result()
        meta.update(stop_reason=mon.stop_reason, iters=mon.k, t_alg=mon.t_alg, t_wall=time.perf_counter() - t0,
                    fields=InexactMonitor.FIELDS, run_cfg=run_cfg, summary=mon.summary(), pstar=ref["pstar"], ref_source=ref["source"],
                    tau=getattr(P, "tau", None), label=task.get("label"), theory_scope=task.get("theory_scope"))
        tr = mon.traces()
        save_atomic(out, task["id"], meta=json.dumps(meta, default=float), hist=H, **tr)
        return f"{mon.stop_reason} it={mon.k} dgap={H[-1, 3]:.1e} inner={mon.inner} t_alg={mon.t_alg:.2f}s"
    raise ValueError(kind)


def cmd_env(out):
    os.makedirs(out, exist_ok=True)
    buf = io.StringIO()
    p = lambda *a: print(*a, file=buf)
    p("C2_supp environment record", now())
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
            p("cpu model:", sorted({l.split(":", 1)[1].strip() for l in fh if l.startswith("model name")}))
    except Exception:
        p("cpu model: n/a")
    try:
        with open("/proc/meminfo") as fh:
            p("mem:", " | ".join(fh.read().splitlines()[:3]))
    except Exception:
        pass
    try:
        import importlib.util
        colab = "google.colab" in sys.modules or importlib.util.find_spec("google.colab") is not None
    except Exception:
        colab = False
    p("google.colab importable:", colab)
    p("COLAB runtime:", {k: os.environ.get(k) for k in ("COLAB_RELEASE_TAG", "COLAB_IMAGE_TYPE", "COLAB_BACKEND_VERSION", "COLAB_GPU")})
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            np.show_config()
        except Exception as e:
            print("show_config failed:", e)
    p("\nnumpy.show_config():\n" + f.getvalue())
    txt = buf.getvalue()
    with open(os.path.join(out, "env.txt"), "w") as fh:
        fh.write(txt)
    return txt


def select(tasks, group, only):
    if group and group != "all":
        tasks = [t for t in tasks if t["group"] == group]
    if only:
        tasks = [t for t in tasks if only in t["id"]]
    return tasks


def cmd_list(out, group, only):
    tasks = select(build_tasks(), group, only)
    for t in tasks:
        print(("done " if is_done(out, t["id"]) else "todo ") + t["id"])
    print(f"{sum(is_done(out, t['id']) for t in tasks)}/{len(tasks)} done")


def cmd_status(out):
    by = {}
    for t in build_tasks():
        d = by.setdefault(t["group"], [0, 0]); d[1] += 1; d[0] += is_done(out, t["id"])
    for g, (a, b) in by.items():
        print(f"{g:9s}: {a}/{b} done")
    fails = glob.glob(os.path.join(out, "logs", "FAILED__*.txt"))
    print(f"failed tasks logged: {len(fails)}")
    for f in fails:
        print("  ", os.path.basename(f))


def cmd_run(out, group, only, cfg=None, tasks=None, smoke=False):
    cfg = cfg or load_config()
    tasks = tasks if tasks is not None else select(build_tasks(cfg), group, only)
    # references first, so that dependent b-tasks find them in the same session
    tasks = sorted(tasks, key=lambda t: 0 if t["kind"] == "ref" else 1)
    os.makedirs(out, exist_ok=True)
    if not os.path.exists(os.path.join(out, "env.txt")):
        cmd_env(out)
    todo = [t for t in tasks if not is_done(out, t["id"])]
    log(f"run: {len(tasks)} tasks selected, {len(tasks) - len(todo)} already done, {len(todo)} to do", out)
    t_start = time.time(); nfail = 0
    ck = {} if smoke else cfg.get("checkpoint", {})
    last_ck, n_since = time.time(), 0
    for j, t in enumerate(todo, 1):
        if is_done(out, t["id"]):
            continue
        t1 = time.time()
        try:
            msg = run_task(t, cfg, out, smoke=smoke)
            log(f"[{j}/{len(todo)}] OK   {t['id']}  {msg}  ({time.time() - t1:.1f}s, elapsed {(time.time() - t_start) / 60:.1f} min)", out)
            fl = os.path.join(out, "logs", f"FAILED__{t['id']}.txt")
            if os.path.exists(fl):
                os.remove(fl)
        except KeyboardInterrupt:
            log("interrupted by user; finished tasks are kept, rerun 'run' to resume", out)
            raise
        except Exception:
            nfail += 1
            os.makedirs(os.path.join(out, "logs"), exist_ok=True)
            with open(os.path.join(out, "logs", f"FAILED__{t['id']}.txt"), "w") as fh:
                fh.write(traceback.format_exc())
            log(f"[{j}/{len(todo)}] FAIL {t['id']}  (traceback in logs/FAILED__{t['id']}.txt)", out)
        n_since += 1
        if ck and (n_since >= ck.get("every_tasks", 25) or time.time() - last_ck >= 60 * ck.get("every_minutes", 15.0)):
            try:
                cmd_checkpoint(out, quiet=True)
            except Exception as e:
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
    print("== smoke 1/3: sanity checks (tests/sanity_c2.py) ==", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, "tests", "sanity_c2.py")], cwd=HERE)
    ok_sanity = r.returncode == 0
    print("== smoke 2/3: tiny runs of every task kind ==", flush=True)
    out = os.path.join(HERE, "C2_out_smoke")
    if os.path.isdir(out):
        shutil.rmtree(out)
    nfail = cmd_run(out, None, None, load_config(), smoke_tasks(), smoke=True)
    print("== smoke 3/3: pack (smoke) ==", flush=True)
    z = cmd_pack(out, smoke=True)
    ok = ok_sanity and nfail == 0 and bool(z) and os.path.exists(z)
    print(f"\nSMOKE {'PASS' if ok else 'FAIL'}  ({time.time() - t0:.0f}s)  sanity={'ok' if ok_sanity else 'FAILED'}  "
          f"failed_tasks={nfail}  zip={os.path.basename(z) if z else None}")
    return 0 if ok else 1


def _zip_dir(out, zpath):
    tmp = f"{zpath}.{os.getpid()}.tmp"
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(out):
            for f in sorted(files):
                if f.endswith(".tmp.npz") or f.endswith(".tmp"):
                    continue
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, os.path.dirname(out)))
    os.replace(tmp, zpath)


def _manifest(out, tasks):
    done = [t["id"] for t in tasks if is_done(out, t["id"])]
    fails = sorted(os.path.basename(f)[8:-4] for f in glob.glob(os.path.join(out, "logs", "FAILED__*.txt")))
    m = dict(created=now(), version=__version__, package_sha256=PACKAGE_HASH, n_tasks=len(tasks), n_done=len(done),
             missing=[t["id"] for t in tasks if t["id"] not in set(done)], failed_logged=fails)
    tmp = os.path.join(out, f"manifest.json.{os.getpid()}.tmp")
    with open(tmp, "w") as fh:
        json.dump(m, fh, indent=1)
    os.replace(tmp, os.path.join(out, "manifest.json"))
    return m


def cmd_checkpoint(out, quiet=False):
    if not os.path.isdir(out):
        return ""
    m = _manifest(out, build_tasks())
    zpath = os.path.join(os.path.dirname(os.path.abspath(out)), "C2_partial_latest.zip")
    _zip_dir(out, zpath)
    msg = f"checkpoint: {m['n_done']}/{m['n_tasks']} tasks -> {zpath}"
    log(msg, out) if quiet else print(msg)
    return zpath


def _run_summary(out, tasks):
    stops, refs, wall = {}, {}, 0.0
    for t in tasks:
        if not is_done(out, t["id"]):
            continue
        m = json.loads(str(np.load(result_path(out, t["id"]), allow_pickle=False)["meta"]))
        if t["kind"] == "ref":
            refs.setdefault(m["status"], []).append(f"{t['inst']} gap_rel={m['gap_rel']:.2e} t={m['time']:.0f}s")
        elif t["kind"] == "a_struct":
            key = (t["group"], "aadmm+fista", f"beta{t['params']['beta']:g}", f"{m['stop_A']}/{m['stop_F']}")
            stops[key] = stops.get(key, 0) + 1
        else:
            v = f"beta{t['params']['beta']:g}" + (("_" + json.dumps(t["params"]["sched"], sort_keys=True)) if "sched" in t["params"] else "")
            key = (t["group"], t["method"], v, m["stop_reason"])
            stops[key] = stops.get(key, 0) + 1
            wall += m.get("t_wall", 0.0)
    summ = dict(version=__version__, created=now(), n_tasks=len(tasks), n_done=sum(is_done(out, t["id"]) for t in tasks),
                ref_status={k: len(v) for k, v in refs.items()}, refs_not_tol=[x for k, v in refs.items() if k != "tol" for x in v],
                stop_reasons=[dict(group=k[0], method=k[1], variant=k[2], stop=k[3], n=v) for k, v in sorted(stops.items())],
                method_wall_minutes=wall / 60.0)
    with open(os.path.join(out, "run_summary.json"), "w") as fh:
        json.dump(summ, fh, indent=1)
    with open(os.path.join(out, "run_summary.md"), "w") as fh:
        fh.write(f"# C2_supp run summary ({__version__}, {summ['created']})\n\ntasks done: {summ['n_done']}/{summ['n_tasks']}\n\n"
                 f"LASSO reference status: {summ['ref_status']}\n\n")
        if summ["refs_not_tol"]:
            fh.write("references NOT within tolerance:\n" + "".join(f"- {x}\n" for x in summ["refs_not_tol"]) + "\n")
        fh.write("| group | method | variant | stop | n |\n|---|---|---|---|---|\n")
        for r in summ["stop_reasons"]:
            fh.write(f"| {r['group']} | {r['method']} | {r['variant']} | {r['stop']} | {r['n']} |\n")
    return summ


def cmd_pack(out, smoke=False):
    if not os.path.isdir(out):
        print("nothing to pack:", out)
        return ""
    cmd_env(out)
    tasks = smoke_tasks() if smoke else build_tasks()
    manifest = _manifest(out, tasks)
    summ = _run_summary(out, tasks)
    figs = os.path.join(out, "figs")
    if os.path.isdir(figs):
        shutil.rmtree(figs)       # never ship figures from an earlier plotting attempt
    args = [sys.executable, os.path.join(HERE, "plot_c2.py"), "--res", out, "--fig", figs] + (["--smoke"] if smoke else [])
    r = subprocess.run(args, capture_output=True, text=True)
    r2 = subprocess.run([sys.executable, os.path.join(HERE, "toy_fig.py"), "--fig", figs], capture_output=True, text=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    with open(os.path.join(out, "logs", "plot_log.txt"), "w") as fh:
        fh.write("== strict plot_c2.py ==\n" + r.stdout + r.stderr + "\n== toy_fig.py ==\n" + r2.stdout + r2.stderr)
    manifest["strict_plot_ok"] = (r.returncode == 0 and r2.returncode == 0)
    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    if not manifest["strict_plot_ok"]:
        if os.path.isdir(figs):
            shutil.rmtree(figs)
        print("STRICT PLOTTING FAILED: see logs/plot_log.txt. No fallback. The zip is still written for diagnosis; report it as a failure.")
    else:
        print("strict figures and tables written to", figs)
    prov = os.path.join(out, "provenance"); os.makedirs(prov, exist_ok=True)
    for name in ("c2/config.json", "c1/config.json", "RUN_LIST.json", "CHANGELOG.md"):
        src = os.path.join(HERE, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(prov, name.replace("/", "_")))
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y%m%d_%H%M")
    zpath = os.path.join(HERE, f"{'C2_smoke' if smoke else 'C2_results'}_{stamp}.zip")
    _zip_dir(out, zpath)
    print(f"packed {manifest['n_done']}/{len(tasks)} finished tasks -> {zpath}")
    if manifest["missing"]:
        print(f"WARNING: {len(manifest['missing'])} tasks missing (listed in manifest.json)")
    if summ["refs_not_tol"]:
        print(f"WARNING: {len(summ['refs_not_tol'])} LASSO references not within tolerance (run_summary.md)")
    return zpath


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["env", "list", "smoke", "run", "status", "checkpoint", "pack"])
    ap.add_argument("--group", default="all", help="all | ref | a_struct | a_range | b")
    ap.add_argument("--only", default=None, help="substring filter on task ids (debug only)")
    ap.add_argument("--out", default=OUT)
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
        sys.exit(1 if cmd_run(a.out, a.group, a.only) else 0)
    elif a.cmd == "checkpoint":
        cmd_checkpoint(a.out)
    elif a.cmd == "pack":
        z = cmd_pack(a.out)
        m = json.load(open(os.path.join(a.out, "manifest.json")))
        sys.exit(0 if z and not m["missing"] and m.get("strict_plot_ok", False) else 2)


if __name__ == "__main__":
    main()
