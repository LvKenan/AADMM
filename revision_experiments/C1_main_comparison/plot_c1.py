#!/usr/bin/env python3
"""Make figures and tables from C1 results ('run_c1.py pack' calls this in strict mode).
    python plot_c1.py --res <path to C1_out or unzipped C1_results_*/C1_out> --fig <output dir>
Outputs: <fig>/curves_<config>.png, <fig>/aadmm_bounds_<config>.png, <fig>/sweep_<P>.png,
         <fig>/table_main.csv, <fig>/table_main.md, <fig>/table_main.tex, <fig>/table_sweep.csv, <fig>/refs.csv,
         <fig>/stops.csv, <fig>/theory_scope.csv
Every (method, variant) pair is a separate curve/row (C1-1.2: admm beta=10/100 and aadmm beta=1.5 in main).
Relative objective error = |F - F*| / (1 + |F*|), Fref is a floating-point reference upper estimate within the requested gap tolerance, not exact F*.
Feasibility in the memo's normalization: ||A0 x + B0 y - b0|| / (1 + ||b0||) (original, unscaled data)."""
import os
import re
import sys
import json
import glob
import argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from c1.metrics import FIELDS   # noqa: E402
from c1.tasks import task_label, theory_scope   # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COL = {"aadmm": "C3", "admm": "C0", "xu_fixed": "C2", "xu_acc": "C1", "iadmm": "C4", "heu_admm": "C5",
       "fast_admm_r": "C7", "aadmm__beta1.5": "C6", "admm__beta10": "C9", "admm__beta100": "C8"}
ORDER = ["aadmm", "aadmm__beta1.5", "admm", "admm__beta10", "admm__beta100", "xu_fixed", "xu_acc", "iadmm",
         "heu_admm", "fast_admm_r"]


def vkey(run):
    t = run["meta"]["task"]
    v = t.get("variant", "default")
    return t["method"] if v == "default" else f"{t['method']}__{v}"


def vlabel(run):
    t = run["meta"]["task"]
    return run["meta"].get("label") or task_label(t["method"], t["params"])
LBL = {"aadmm": "AADMM ($\\beta$=1)", "admm": "ADMM / proximal ADMM", "xu_fixed": "Xu fixed", "xu_acc": "Xu accelerated",
       "iadmm": "Tang-Toh IADMM", "heu_admm": "Tang-Toh Heu-ADMM", "fast_admm_r": "Goldstein fast ADMM restart"}
I = {f: j for j, f in enumerate(FIELDS)}


def load(res, allow_bad_refs=False):
    refs, runs = {}, []
    for f in glob.glob(os.path.join(res, "results", "*.npz")):
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        if meta["task"]["kind"] == "ref":
            tol = meta.get("ref_cfg", {}).get("tol", 1e-12)
            values_finite = all(np.isfinite(meta.get(k, np.inf)) for k in ("LB", "UB", "Fstar", "gap_rel"))
            good_gap = values_finite and abs(meta["UB"] - meta["LB"]) / (1 + abs(meta["LB"])) <= tol
            if meta.get("status") != "tol" or not good_gap or meta.get("gap_rel", np.inf) > tol:
                msg = f"Reference not within requested tolerance {meta['task']['inst']}; inspect LB/UB before plotting objective gaps"
                if not allow_bad_refs:
                    raise ValueError(msg)
                print("WARNING:", msg, "(kept because --allow-bad-refs; objective-gap numbers for this instance are NOT certified)")
            refs[meta["task"]["inst"]] = dict(meta=meta, lam=z["lam_star"], By=z["By_star"])
        else:
            tr = {k: z[k] for k in ("kkt_all", "pen_all", "lpred_all", "restart_iters") if k in z.files}
            runs.append(dict(meta=meta, H=z["hist"], tr=tr))
    missing = {r["meta"]["task"]["inst"] for r in runs} - refs.keys()
    if missing:
        raise ValueError(f"Missing reference solves: {sorted(missing)}")
    return refs, runs


def config_of(inst):
    return re.sub(r"_s\d+$", "", inst)


def rel(v, Fs):
    return np.abs(v - Fs) / (1.0 + abs(Fs))


def first_reach(run, col, lev):
    if col == "KKT" and str(lev) in run["meta"].get("first_hit", {}):
        hit = run["meta"]["first_hit"][str(lev)]
        return tuple(hit) if hit is not None else (np.nan, np.nan)
    H = run["H"]
    idx = np.where(H[:, I[col]] <= lev)[0]
    return (H[idx[0], I["k"]], H[idx[0], I["t_alg"]]) if len(idx) else (np.nan, np.nan)


def fmt(t, e=False):
    mu, sd, n = t
    if not np.isfinite(mu):
        return "--"
    return (f"{mu:.2e}" if e else f"{mu:.4g}") + (f" ± {sd:.2g}" if n > 1 else "")


def med_n(v):
    v = list(v)
    w = [x for x in v if x is not None and np.isfinite(x)]
    if not w:
        return "--" + ("" if not v else f" (0/{len(v)})")
    return f"{np.median(w):.4g}" + ("" if len(w) == len(v) else f" ({len(w)}/{len(v)})")


def dual_gap_reach(run, refs, lev):
    """First iteration with (Fref - d(lam^k))/(1+|Fref|) <= lev.  Exact for AADMM (per-iteration d(lam~^k))."""
    inst = run["meta"]["task"]["inst"]
    if inst not in refs:
        return np.nan
    Fs = refs[inst]["meta"]["Fstar"]
    lp = run.get("tr", {}).get("lpred_all")
    if lp is not None and len(lp):
        idx = np.where((Fs - lp) / (1 + abs(Fs)) <= lev)[0]
        return float(idx[0] + 1) if len(idx) else np.nan
    H = run["H"]
    idx = np.where((Fs - H[:, I["dval"]]) / (1 + abs(Fs)) <= lev)[0]
    return H[idx[0], I["k"]] if len(idx) else np.nan


def ms(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return (np.nan, np.nan, 0) if len(v) == 0 else (v.mean(), v.std(), len(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=os.path.join(HERE, "C1_out"))
    ap.add_argument("--fig", default=os.path.join(HERE, "C1_figs"))
    ap.add_argument("--allow-bad-refs", action="store_true",
                    help="warn instead of failing when a reference solve missed its gap tolerance")
    a = ap.parse_args()
    os.makedirs(a.fig, exist_ok=True)
    refs, runs = load(a.res, a.allow_bad_refs)
    print(f"loaded {len(refs)} reference solves, {len(runs)} runs")
    with open(os.path.join(a.fig, "refs.csv"), "w") as fh:
        fh.write("instance,Fstar,LB,UB,gap_rel,status,iters,time_s\n")
        for k, r in sorted(refs.items()):
            m = r["meta"]
            fh.write(f"{k},{m['Fstar']:.15g},{m['LB']:.15g},{m['UB']:.15g},{m['gap_rel']:.3e},{m['status']},{m['iters']},{m['time']:.2f}\n")
    with open(os.path.join(a.fig, "stops.csv"), "w") as fh:
        fh.write("task,stop_reason,iters,t_alg,t_wall\n")
        for r in sorted(runs, key=lambda r: r["meta"]["task"]["id"]):
            m = r["meta"]
            fh.write(f"{m['task']['id']},{m['stop_reason']},{m['iters']},{m['t_alg']:.3f},{m['t_wall']:.3f}\n")

    main_runs = [r for r in runs if r["meta"]["task"]["group"] == "main"]
    configs = sorted({config_of(r["meta"]["task"]["inst"]) for r in main_runs})
    rows = []
    bound_rows = []
    for cfgname in configs:
        rs = [r for r in main_runs if config_of(r["meta"]["task"]["inst"]) == cfgname]
        methods = [m for m in ORDER if any(vkey(r) == m for r in rs)]
        labels = {vkey(r): vlabel(r) for r in rs}
        fig, ax = plt.subplots(2, 3, figsize=(16, 9))
        for r in rs:
            m = vkey(r); H = r["H"]; inst = r["meta"]["task"]["inst"]
            if inst not in refs:
                continue
            Fs = refs[inst]["meta"]["Fstar"]
            nb = r["meta"]["problem"].get("normb_orig", np.nan)
            k = H[:, I["k"]]; t = H[:, I["t_alg"]]
            kw = dict(color=COL.get(m, "k"), lw=0.8, alpha=0.6)
            ax[0, 0].loglog(k, H[:, I["KKT"]], **kw)
            ax[0, 1].loglog(t, H[:, I["KKT"]], **kw)
            ax[0, 2].loglog(k, rel(H[:, I["F"]], Fs), **kw)
            ax[0, 2].loglog(k, rel(H[:, I["Fbar"]], Fs), ls=":", **kw)
            ax[1, 0].loglog(k, H[:, I["rp_abs"]] / (1 + nb), **kw)
            ax[1, 0].loglog(k, H[:, I["rp_abs_bar"]] / (1 + nb), ls=":", **kw)
            ax[1, 1].loglog(k, np.maximum(H[:, I["Rd_x"]], H[:, I["Rd_y"]]) + 1e-300, **kw)
            ax[1, 2].loglog(k, np.maximum((Fs - H[:, I["dval"]]) / (1 + abs(Fs)), 1e-17), **kw)
        for m in methods:
            ax[0, 0].plot([], [], color=COL.get(m, "k"), label=labels[m])
        ax[0, 0].legend(fontsize=8)
        titles = ["KKT = max(Rp, Rd) vs iteration", "KKT vs algorithm time [s]",
                  "|F-Fref|/(1+|Fref|): last (solid), averages (dotted)",
                  "||Ax+By-b||/(1+||b||): last (solid), ergodic (dotted)", "Rd (Tang-Toh normalized KKT residual)",
                  "(Fref - d(lam))/(1+|Fref|) (reference gap)"]
        for axx, tt in zip(ax.ravel(), titles):
            axx.set_title(tt, fontsize=9); axx.grid(True, which="both", alpha=0.3)
        fig.suptitle(cfgname); fig.tight_layout()
        fig.savefig(os.path.join(a.fig, f"curves_{cfgname}.png"), dpi=130); plt.close(fig)

        # AADMM: dual gap vs the theorem bound C/(k+1)^2, lam^1 = z^1 = 0, beta of the run
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for r in rs:
            if r["meta"]["task"]["method"] != "aadmm":
                continue
            inst = r["meta"]["task"]["inst"]; H = r["H"]
            if inst not in refs:
                continue
            Fs = refs[inst]["meta"]["Fstar"]; lam_s = refs[inst]["lam"]; By_s = refs[inst]["By"]
            beta = float(r["meta"]["task"]["params"].get("beta", 1.0))
            C = (1 / beta) * np.sum(lam_s ** 2) + (1 / (2 - beta)) * np.sum((lam_s - 2 * By_s) ** 2)
            col = COL.get(vkey(r), "C3")
            lp = r["tr"].get("lpred_all")
            if lp is not None and len(lp):          # exact per-iteration dual value d(lam~^k) (C1-1.2)
                kk = np.arange(1, len(lp) + 1)
                v = Fs - lp
                ax[0].loglog(kk, np.maximum(v, 1e-17), color=col, lw=0.8)
                ax[0].loglog(kk, C / (kk + 1) ** 2, color=col, lw=0.6, ls="--")
                bound_rows.append((r["meta"]["task"]["id"], beta, C, float(np.max((kk + 1) ** 2 * v / C))))
            else:
                k = H[:, I["k"]]
                ax[0].loglog(k, np.maximum(Fs - H[:, I["dval"]], 1e-17), color=col, lw=0.8)
                ax[0].loglog(k, C / (k + 1) ** 2, color=col, lw=0.6, ls="--")
            k = H[:, I["k"]]
            ax[1].loglog(k, H[:, I["rp_abs_bar"]], color="C3", lw=0.8, ls=":")
            ax[1].loglog(k, H[:, I["rp_abs"]], color="C3", lw=0.8)
            ax[1].loglog(k, H[:, I["pp_rp_abs"]], color="C2", lw=0.8)
        ax[0].set_title("AADMM: dual gap Fref-d(lam~^k) (solid), estimated C/(k+1)^2 (dashed); red beta=1, pink beta=1.5", fontsize=8)
        ax[1].set_title("AADMM ||r||: predictor (solid), t-weighted ergodic (dotted), Prop. E point (green)", fontsize=9)
        for axx in ax:
            axx.grid(True, which="both", alpha=0.3)
        fig.suptitle(cfgname); fig.tight_layout()
        fig.savefig(os.path.join(a.fig, f"aadmm_bounds_{cfgname}.png"), dpi=130); plt.close(fig)

        for m in methods:
            rm = [r for r in rs if vkey(r) == m]
            row = dict(config=cfgname, method=m, label=labels[m], n=len(rm),
                       scope=rm[0]["meta"].get("theory_scope") or theory_scope(rm[0]["meta"]["task"]["method"], rm[0]["meta"]["task"]["params"]))
            for lev in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
                its, ts = zip(*[first_reach(r, "KKT", lev) for r in rm])
                row[f"it_{lev:g}"] = ms(its); row[f"t_{lev:g}"] = ms(ts)
                row[f"med_{lev:g}"] = med_n(its); row[f"t_{lev:g}_raw"] = ts
            dg = [dual_gap_reach(r, refs, 1e-8) for r in rm]
            row["dgap_1e-8"] = med_n(dg)
            rs_ = [r["meta"].get("run_summary", {}) for r in rm]
            base_m = rm[0]["meta"]["task"]["method"]
            row["Sk_k2"] = med_n([x.get("S_k_over_k2", np.nan) for x in rs_]) if base_m in ("iadmm", "heu_admm") else "n/a"
            row["n_restart"] = med_n([x.get("n_restart", np.nan) for x in rs_]) if base_m == "fast_admm_r" else "n/a"
            row["pen_last"] = med_n([x.get("penalty_last", np.nan) for x in rs_])
            fin = [r["H"][-1] for r in rm]
            row["final_KKT"] = ms([f[I["KKT"]] for f in fin])
            row["final_relF"] = ms([rel(f[I["F"]], refs[r["meta"]["task"]["inst"]]["meta"]["Fstar"])
                                    for f, r in zip(fin, rm) if r["meta"]["task"]["inst"] in refs])
            row["final_rp"] = ms([f[I["rp_abs"]] / (1 + r["meta"]["problem"].get("normb_orig", np.nan)) for f, r in zip(fin, rm)])
            row["fact"] = ms([f[I["fact"]] for f in fin]); row["matvec"] = ms([f[I["matvec"]] for f in fin])
            row["stops"] = ",".join(sorted({r["meta"]["stop_reason"] for r in rm}))
            rows.append(row)

    with open(os.path.join(a.fig, "table_main.md"), "w") as fh:
        fh.write("| config | method | n | KKT<=1e-4 | KKT<=1e-6 | KKT<=1e-8 | t_alg to 1e-8 [s] | dual gap<=1e-8 | final KKT | S_k/k^2 | restarts | last penalty |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['config']} | {r['label']} | {r['n']} | {r['med_0.0001']} | {r['med_1e-06']} | {r['med_1e-08']} | "
                     f"{med_n(r['t_1e-08_raw']) if 't_1e-08_raw' in r else '--'} | {r['dgap_1e-8']} | {fmt(r['final_KKT'], True)} | "
                     f"{r['Sk_k2']} | {r['n_restart']} | {r['pen_last']} |\n")
        fh.write("\nIteration columns: median over seeds, (reached/n) when not all seeds reached the level. KKT first hits are exact "
                 "(per-iteration). Dual gap: exact per-iteration for AADMM (d(lam~^k)), logged-iterate resolution otherwise.\n")
    with open(os.path.join(a.fig, "theory_scope.csv"), "w") as fh:
        fh.write("method_variant,label,theory_scope\n")
        seen = set()
        for r in rows:
            if r["method"] not in seen:
                seen.add(r["method"])
                fh.write(f"{r['method']},\"{r['label']}\",\"{r['scope']}\"\n")
    with open(os.path.join(a.fig, "aadmm_theorem_ratio.csv"), "w") as fh:
        fh.write("task,beta,C,max_k (k+1)^2 v_k / C\n")
        for t_ in bound_rows:
            fh.write(f"{t_[0]},{t_[1]:g},{t_[2]:.6g},{t_[3]:.4g}\n")

    with open(os.path.join(a.fig, "table_main.csv"), "w") as fh:
        fh.write("config,method_variant,label,n_seeds,iters_to_1e-5,reached_1e-5,time_to_1e-5,iters_to_1e-7,reached_1e-7,time_to_1e-7,final_KKT,final_relobj,final_feas,factorizations,matvecs,stop_reasons\n")
        for r in rows:
            fh.write(f"{r['config']},{r['method']},\"{r['label']}\",{r['n']},{fmt(r['it_1e-05'])},{r['it_1e-05'][2]},{fmt(r['t_1e-05'])},"
                     f"{fmt(r['it_1e-07'])},{r['it_1e-07'][2]},{fmt(r['t_1e-07'])},{fmt(r['final_KKT'], True)},"
                     f"{fmt(r['final_relF'], True)},{fmt(r['final_rp'], True)},{fmt(r['fact'])},{fmt(r['matvec'])},\"{r['stops']}\"\n")
    with open(os.path.join(a.fig, "table_main.tex"), "w") as fh:
        fh.write("% config & method & it(1e-5) & time(1e-5) & it(1e-7) & time(1e-7) & final rel. obj. & final feas.\\\\\n")
        for r in rows:
            fh.write(f"{r['config'].replace('_', '-')} & {r['label'].replace('beta', '$\\beta$')} & {fmt(r['it_1e-05'])} & {fmt(r['t_1e-05'])} & "
                     f"{fmt(r['it_1e-07'])} & {fmt(r['t_1e-07'])} & {fmt(r['final_relF'], True)} & {fmt(r['final_rp'], True)}\\\\\n")

    # penalty sweep
    sw = [r for r in runs if r["meta"]["task"]["group"] == "sweep"]
    default = {(config_of(r["meta"]["task"]["inst"]), r["meta"]["task"]["method"], r["meta"]["task"]["inst"]): r
               for r in main_runs if r["meta"]["task"].get("variant", "default") == "default"}
    srows = []
    for P in sorted({r["meta"]["task"]["P"] for r in sw}):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for m, key, base in (("aadmm", "beta", 1.0), ("admm", "beta", 1.0), ("iadmm", "beta1_mult", 1.0), ("heu_admm", "beta1_mult", 1.0)):
            pts, used = {}, {}
            for r in sw:
                t = r["meta"]["task"]
                if t["P"] == P and t["method"] == m and key in t["params"]:
                    pts.setdefault(t["params"][key], []).append(first_reach(r, "KKT", 1e-5)[0])
                    used.setdefault(t["params"][key], set()).add(r["meta"].get("method_settings", {}).get("beta1_used"))
            insts = {r["meta"]["task"]["inst"] for r in sw if r["meta"]["task"]["P"] == P}
            for ii in insts:
                d = default.get((config_of(ii), m, ii))
                if d is not None:
                    pts.setdefault(base, []).append(first_reach(d, "KKT", 1e-5)[0])
            if not pts:
                continue
            xs = sorted(pts)
            med = [np.nanmedian(pts[x]) if np.any(np.isfinite(pts[x])) else np.nan for x in xs]
            ax.semilogx(xs, med, "o-", color=COL[m], label=f"{LBL[m]} ({key})")
            for x in xs:
                srows.append((P, m, key, x, np.nanmedian(pts[x]) if np.any(np.isfinite(pts[x])) else np.nan,
                              int(np.sum(np.isfinite(pts[x]))), len(pts[x]),
                              "|".join("%.3g" % u for u in sorted(v for v in used.get(x, ()) if v is not None))))
        ax.set_xlabel("parameter (beta for AADMM/ADMM; requested multiplier of default beta_1 for Tang-Toh)")
        ax.set_ylabel("iterations to KKT <= 1e-5 (median; missing = not reached)")
        ax.set_yscale("log"); ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8); ax.set_title(f"penalty robustness, {P}")
        fig.tight_layout(); fig.savefig(os.path.join(a.fig, f"sweep_{P}.png"), dpi=130); plt.close(fig)
    with open(os.path.join(a.fig, "table_sweep.csv"), "w") as fh:
        fh.write("problem,method,param,value,median_iters_to_1e-5,n_reached,n_runs,beta1_used\n")
        for s in srows:
            fh.write(",".join(str(x) for x in s) + "\n")
    lin = [r for r in sw if r["meta"]["task"]["params"].get("tt_q") == "lin"]
    if lin:
        with open(os.path.join(a.fig, "table_sweep_P3_ttq_lin.csv"), "w") as fh:
            fh.write("task,iters_to_1e-5,time_to_1e-5,stop,iters\n")
            for r in lin:
                i5, t5 = first_reach(r, "KKT", 1e-5)
                fh.write(f"{r['meta']['task']['id']},{i5},{t5},{r['meta']['stop_reason']},{r['meta']['iters']}\n")
    print("figures and tables written to", a.fig)


if __name__ == "__main__":
    main()
