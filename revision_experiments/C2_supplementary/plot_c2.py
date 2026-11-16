#!/usr/bin/env python3
"""Strict figures and tables for C2_supp.  python plot_c2.py --res C2_out --fig C2_out/figs [--smoke]
Non-smoke: every task of RUN_LIST must be present, otherwise exit 1 (no partial figures are produced)."""
import os, sys, json, argparse, glob
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from c2.tasks import build_tasks, smoke_tasks, sched_tag
from c1.metrics import FIELDS as F1
from c2.methods import InexactMonitor
F2 = InexactMonitor.FIELDS
iF = {f: i for i, f in enumerate(F1)}; iB = {f: i for i, f in enumerate(F2)}
COL = {"aadmm": "#CC3311", "dual_fista": "#0077BB", "dual_pg": "#009988", "admm": "#0077BB"}
def mlabel(m, b):
    return {"aadmm": f"AADMM-{b:g}", "admm": f"ADMM-{b:g}", "dual_fista": "dual FISTA", "dual_pg": "dual prox-grad"}[m]


NAME = {"aadmm": "AADMM-$\\beta$", "dual_fista": "dual FISTA", "dual_pg": "dual prox-grad", "admm": "ADMM"}


def load(res, tid):
    p = os.path.join(res, "results", tid + ".npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    d = {k: z[k] for k in z.files if k != "meta"}; d["meta"] = json.loads(str(z["meta"]))
    return d


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.titlesize": 7, "legend.fontsize": 5.5, "pdf.fonttype": 42, "savefig.dpi": 300,
                         "axes.grid": True, "grid.alpha": 0.3, "lines.linewidth": 0.9})
    return plt


W = 372 / 72.27


def save(fig, figdir, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(figdir, f"{name}.{ext}"), bbox_inches="tight")


def strip_seed(iid):
    return iid.rsplit("_s", 1)[0]


def fmt(v, f="{:.1e}"):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f.format(v)


def refs_C(inst, beta):
    """Theorem constant C at lam^1 = z^1 = 0: ||lam*||^2/beta + ||lam* - 2By*||^2/(2-beta)."""
    from c2.refs import _load
    z, meta = _load()
    if inst not in meta or beta >= 2:
        return np.nan
    l, By = z[f"{inst}__lam_star"], z[f"{inst}__By_star"]
    return float(l @ l) / beta + float((l - 2 * By) @ (l - 2 * By)) / (2 - beta)


# ------------------------------------------------------------------------------------------------- (a) range
def part_a_range(res, figdir, tasks, plt, lines):
    T = [t for t in tasks if t["group"] == "a_range"]
    if not T:
        return
    rows = {}
    for t in T:
        d = load(res, t["id"]); m = d["meta"]; H = d["hist"]
        Fs = m["ref"]["Fstar"]; sc = 1 + abs(Fs)
        gap = (Fs - H[:, iF["dval"]]) / sc if len(H) else np.array([np.nan])
        b = t["params"]["beta"]; Cc = refs_C(t["inst"], b) if t["method"] == "aadmm" else np.nan
        ratio = np.nanmax((H[:, 0] + 1) ** 2 * np.maximum(gap, 0) * sc / Cc) if np.isfinite(Cc) and len(H) else np.nan
        key = (strip_seed(t["inst"]), t["method"], b)
        rows.setdefault(key, []).append(dict(stop=m["stop_reason"], it=m["iters"], kkt=float(d["kkt_all"][-1]) if len(d["kkt_all"]) else np.nan,
                                             gap=float(gap[-1]), gapmin=float(np.nanmin(np.abs(gap))), t=m["t_alg"], ratio=ratio,
                                             H=H, gapk=gap, seed=t["spec"]["seed"]))
    insts = sorted({k[0] for k in rows}, key=lambda s: [t["inst"] for t in T].index(next(t["inst"] for t in T if strip_seed(t["inst"]) == s)))
    betas = sorted({k[2] for k in rows}); meths = [m for m in ("aadmm", "dual_fista", "dual_pg") if any(k[1] == m for k in rows)]
    # table
    md = ["| instance | method | beta | tol / maxit / diverged | median iters (tol runs) | median final KKT | median final rel. dual gap | max theorem ratio (AADMM) |",
          "|---|---|---|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{llr" + "c" * len(betas) + "}", r"\toprule", "instance & method & & " + " & ".join(f"$\\beta={b:g}$" for b in betas) + r"\\", r"\midrule"]
    for inst in insts:
        for mth in meths:
            cells = []
            for b in betas:
                R = rows.get((inst, mth, b), [])
                if not R:
                    cells.append("--"); continue
                n = lambda s: sum(r["stop"] == s for r in R)
                itol = [r["it"] for r in R if r["stop"] == "tol"]
                md.append(f"| {inst} | {mth} | {b:g} | {n('tol')}/{n('maxit') + n('time_budget')}/{n('diverged')} | "
                          f"{fmt(float(np.median(itol)) if itol else None, '{:.0f}')} | {fmt(float(np.median([r['kkt'] for r in R])))} | "
                          f"{fmt(float(np.median([r['gap'] for r in R])))} | {fmt(float(np.nanmax([r['ratio'] for r in R])) if mth == 'aadmm' and b < 2 else None, '{:.2f}')} |")
                if n("diverged"):
                    cells.append(r"div." + (f"\\,({n('diverged')}/{len(R)})" if n("diverged") < len(R) else ""))
                elif n("tol") == len(R):
                    cells.append(f"{np.median(itol):.0f}")
                elif n("tol"):
                    cells.append(f"{np.median(itol):.0f}$^{{{n('tol')}/{len(R)}}}$")
                else:
                    cells.append(f"n.c.\\,({np.median([r['kkt'] for r in R]):.0e})")
            tex.append(f"{inst.replace('_', '-')} & {mth.replace('_', ' ')} & & " + " & ".join(cells) + r"\\")
        tex.append(r"\midrule")
    tex[-1] = r"\bottomrule"; tex.append(r"\end{tabular}")
    open(os.path.join(figdir, "table_a_range.md"), "w").write(
        "# (a) beta range: AADMM-beta vs dual FISTA vs dual proximal gradient (stop at KKT <= tol of c2/config a_run)\n\n"
        "rel. dual gap = (F* - d(lam))/(1+|F*|) at the last logged iterate; theorem ratio = max_k (k+1)^2 gap_k / C (Theorem, lam^1=z^1=0).\n\n"
        + "\n".join(md) + "\n")
    open(os.path.join(figdir, "table_a_range.tex"), "w").write(
        "% cells: median iterations to KKT<=1e-8 (superscript: fraction of seeds reaching it); div.: diverged (|lam| > div bound or non-finite);\n"
        "% n.c.: not converged within maxit (median final KKT in brackets)\n" + "\n".join(tex) + "\n")
    lines.append(f"a_range: {len(T)} runs, table_a_range.md/.tex")
    # outcome figure: final dual gap vs beta
    nc = min(4, len(insts)); nr = int(np.ceil(len(insts) / nc))
    fig, axs = plt.subplots(nr, nc, figsize=(W, 0.62 * W * nr / 2 + 0.3), squeeze=False)
    for ax, inst in zip(axs.flat, insts):
        for mth in meths:
            bs, med, div = [], [], []
            for b in betas:
                R = rows.get((inst, mth, b), [])
                if not R:
                    continue
                if any(r["stop"] == "diverged" for r in R):
                    div.append(b)
                else:
                    bs.append(b); med.append(max(float(np.median([r["kkt"] for r in R])), 1e-16))
            ax.semilogy(bs, med, "-o", ms=2.5, color=COL[mth], label=NAME[mth])
            if div:
                ax.semilogy(div, [1e2] * len(div), "x", ms=4, color=COL[mth])
        ax.axvline(1.001, color="0.5", ls=":", lw=0.7); ax.axvline(2.0, color="0.3", ls="--", lw=0.7)
        ax.set_title(inst.replace("_", " ")); ax.set_xlabel(r"$\beta$"); ax.set_ylim(1e-16, 1e4)
    for ax in list(axs.flat)[len(insts):]:
        ax.axis("off")
    axs[0, 0].set_ylabel("final KKT (median)"); axs[0, 0].legend(loc="lower right")
    fig.suptitle(r"final KKT residual vs $\beta$ ($\times$ at $10^2$: diverged; dotted: $1/L_\Psi$, dashed: $\beta=2$)", fontsize=7)
    fig.tight_layout(); save(fig, figdir, "fig_a_range_outcome"); plt.close(fig)
    # trajectories: two largest instances (if present), seed 0
    pick = [i for i in insts if i.startswith("P2_n700") or i.startswith("P4_N128")] or insts[:2]
    tb = [b for b in (1.0, 1.5, 1.9, 2.5) if b in betas] or betas[:4]
    fig, axs = plt.subplots(len(pick), len(tb), figsize=(W, 0.3 * W * len(pick) + 0.3), squeeze=False, sharey=True)
    for i, inst in enumerate(pick):
        for j, b in enumerate(tb):
            ax = axs[i, j]
            for mth in meths:
                R = [r for r in rows.get((inst, mth, b), []) if r["seed"] == min(rr["seed"] for rr in rows[(inst, mth, b)])]
                if R:
                    H = R[0]["H"]; g = np.abs(R[0]["gapk"])
                    ax.loglog(H[:, 0], np.maximum(g, 1e-17), color=COL[mth], label=NAME[mth], ls="-" if mth == "aadmm" else "--")
                    if R[0]["stop"] == "diverged":
                        ax.plot(H[-1, 0], max(g[-1], 1e-17), "x", color=COL[mth], ms=4)
            ax.set_title(f"{inst.replace('_', ' ')}, $\\beta={b:g}$"); ax.set_xlabel("$k$")
        axs[i, 0].set_ylabel("|rel. dual gap|")
    axs[0, 0].legend(loc="lower left"); fig.tight_layout(); save(fig, figdir, "fig_a_range_traj"); plt.close(fig)
    lines.append("a_range figures: fig_a_range_outcome, fig_a_range_traj")


def part_a_struct(res, figdir, tasks, plt, lines):
    T = [t for t in tasks if t["group"] == "a_struct"]
    if not T:
        return
    md = ["| instance | beta | max ||lam~_AADMM - eta_FISTA|| (rel) | final dev | max Lemma (b) resid | max Lemma (c) resid | max |extra term| |", "|---|---|---|---|---|---|---|"]
    data = {}
    for t in T:
        d = load(res, t["id"]); m = d["meta"]; b = t["params"]["beta"]
        data.setdefault(strip_seed(t["inst"]), {}).setdefault(b, []).append(d)
        md.append(f"| {t['inst']} | {b:g} | {m['dev_max']:.1e} | {m['dev_final']:.1e} | {m['max_lemma_b_resid']:.1e} | "
                  f"{fmt(m['max_identity_resid'])} | {fmt(float(np.max(d['extra'])) if len(d['extra']) else None)} |")
    open(os.path.join(figdir, "table_a_struct.md"), "w").write(
        "# (a) Remark rem:dualfista (i)-(iii): AADMM-beta (compatible init) vs FISTA-beta, deviation of predictor from FISTA iterate\n\n"
        "dev scaled by 1+||lam*||. Expected: ~1e-15 for beta=1 on affine grad Psi (P3, P4); > 0 for P1 (nonaffine, (ii)) and beta != 1 ((iii)).\n\n"
        + "\n".join(md) + "\n")
    insts = list(data)
    fig, axs = plt.subplots(1, len(insts), figsize=(W, 0.3 * W + 0.3), squeeze=False, sharey=True)
    for ax, inst in zip(axs[0], insts):
        for b, cl in zip(sorted(data[inst]), ("#0077BB", "#CC3311", "#009988", "k")):
            d = data[inst][b][0]
            ax.semilogy(np.arange(1, len(d["dev"]) + 1), np.maximum(d["dev"], 1e-17), color=cl, label=f"$\\beta={b:g}$")
            if len(d["extra"]):
                ax.semilogy(np.arange(2, len(d["extra"]) + 2), np.maximum(d["extra"], 1e-17), color=cl, ls=":", lw=0.7)
        ax.set_title(inst.rsplit("_s", 1)[0].replace("_", " ")); ax.set_xlabel("$k$")
    axs[0, 0].set_ylabel(r"$\|\tilde\lambda^k-\eta^k\|/(1+\|\lambda^*\|)$"); axs[0, 0].legend(loc="center right")
    fig.suptitle(r"AADMM-$\beta$ predictor vs FISTA-$\beta$ iterate (solid); extra term $\frac{t_k}{t_{k+1}}(\beta-1)\beta r^k$ (dotted)", fontsize=7)
    fig.tight_layout(); save(fig, figdir, "fig_a_struct"); plt.close(fig)
    lines.append(f"a_struct: {len(T)} runs, table_a_struct.md, fig_a_struct")


# ------------------------------------------------------------------------------------------------- (b) inexact
def part_b(res, figdir, tasks, plt, lines):
    T = [t for t in tasks if t["group"] == "b"]
    if not T:
        return
    rows = {}
    for t in T:
        d = load(res, t["id"]); m = d["meta"]; H = d["hist"]; s = m["summary"]
        key = (strip_seed(t["inst"]), t["method"], t["params"]["beta"], sched_tag(t["params"]["sched"]))
        rows.setdefault(key, []).append(dict(stop=m["stop_reason"], it=m["iters"], dgap=float(d["dgap_all"][-1]),
                                             Lgmin=float(np.nanmin(H[:, iB["Lgap"]])), ratio=float(np.nanmax(H[:, iB["ratio"]])) if np.isfinite(H[:, iB["ratio"]]).any() else np.nan,
                                             oe=float(H[-1, iB["objerr"]]), rp=float(H[-1, iB["rp"]]), inner=s["inner_total"], t=m["t_alg"],
                                             unmet=s["n_unmet"], floor=s["n_floor_clipped"], H=H, dg=d["dgap_all"], seed=t["spec"]["seed"]))
    md = ["| instance | method | beta | inner schedule | stops | median outer its | median final dual gap | min Lgap (<0: L above p*) | max theorem ratio | median final obj. err | median final rp | median inner its | median t_alg [s] | unmet / floor-clipped |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    keys = list(dict.fromkeys((strip_seed(t["inst"]), t["method"], t["params"]["beta"], sched_tag(t["params"]["sched"])) for t in T))
    tex = [r"\begin{tabular}{lllrrrrr}", r"\toprule", r"instance & method & inner accuracy & outer its & dual gap & obj.\ err. & inner its & time [s]\\", r"\midrule"]
    for key in keys:
        R = rows[key]; med = lambda f: float(np.median([r[f] for r in R]))
        stops = ",".join(f"{s}:{sum(r['stop'] == s for r in R)}" for s in sorted({r['stop'] for r in R}))
        md.append(f"| {key[0]} | {key[1]} | {key[2]:g} | {key[3]} | {stops} | {med('it'):.0f} | {med('dgap'):.1e} | {min(r['Lgmin'] for r in R):.1e} | "
                  f"{fmt(float(np.nanmax([r['ratio'] for r in R])) if key[1] == 'aadmm' else None, '{:.2g}')} | {med('oe'):.1e} | {med('rp'):.1e} | {med('inner'):.0f} | {med('t'):.2f} | "
                  f"{sum(r['unmet'] for r in R)} / {sum(r['floor'] for r in R)} |")
        tex.append(f"{key[0].replace('_', '-')} & {mlabel(key[1], key[2])} & {key[3].replace('_', ' ')} & {med('it'):.0f} & {med('dgap'):.1e} & {med('oe'):.1e} & {med('inner'):.0f} & {med('t'):.2f}\\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(figdir, "table_b_inexact.md"), "w").write(
        "# (b) inexact x-subproblems (Remark rem:inexact, comment C5): outer behaviour vs inner accuracy\n\n"
        "tol_cX_pY: inner certificate <= max(X/k^Y, 1e-12)(1+|p*|) by warm-started FISTA+restart; count_nN: N warm-started ISTA (LASSO) / Chambolle (TVD) steps (paper: 10 / 20).\n"
        "dual gap = (p* - d(Pi lam~))/(1+|p*|) (valid); Lgap = (p* - L(x~,y~,lam~))/(1+|p*|); theorem ratio = (k+1)^2 (p* - d)/C.\n\n" + "\n".join(md) + "\n")
    open(os.path.join(figdir, "table_b_inexact.tex"), "w").write("% medians over seeds; dual gap relative (p*-d)/(1+|p*|)\n" + "\n".join(tex) + "\n")
    insts = list(dict.fromkeys(k[0] for k in keys))
    combos = list(dict.fromkeys((k[1], k[2]) for k in keys))
    for xaxis in ("k", "time"):
        fig, axs = plt.subplots(len(insts), len(combos), figsize=(W, 0.27 * W * len(insts) + 0.4), squeeze=False, sharey=True)
        for i, inst in enumerate(insts):
            for j, (mth, b) in enumerate(combos):
                ax = axs[i, j]; sch = [k[3] for k in keys if k[0] == inst and k[1] == mth and k[2] == b]
                cmap = plt.get_cmap("viridis", max(len(sch), 2))
                for c, sname in enumerate(sch):
                    R = rows[(inst, mth, b, sname)]; r = min(R, key=lambda r: r["seed"])
                    if xaxis == "k":
                        ax.loglog(np.arange(1, len(r["dg"]) + 1), np.maximum(np.abs(r["dg"]), 1e-17), color=cmap(c), label=sname)
                    else:
                        ax.loglog(np.maximum(r["H"][:, iB["t_alg"]], 1e-4), np.maximum(np.abs(r["H"][:, iB["dgap"]]), 1e-17), color=cmap(c), label=sname)
                ax.set_title(f"{inst.replace('_', ' ')}, {mlabel(mth, b)}", fontsize=6)
                ax.set_xlabel("$k$" if xaxis == "k" else "algorithm time [s]")
            axs[i, 0].set_ylabel("rel. dual gap")
        for j in range(len(combos)):
            axs[0, j].legend(fontsize=4.5, loc="lower left")
        fig.tight_layout(); save(fig, figdir, f"fig_b_dgap_{xaxis}"); plt.close(fig)
    # theorem ratio, AADMM
    fig, axs = plt.subplots(1, len(insts), figsize=(W, 0.3 * W + 0.3), squeeze=False, sharey=True)
    for ax, inst in zip(axs[0], insts):
        sch = [k for k in keys if k[0] == inst and k[1] == "aadmm" and k[2] == max(kk[2] for kk in keys if kk[1] == "aadmm")]
        cmap = plt.get_cmap("viridis", max(len(sch), 2))
        for c, key in enumerate(sch):
            r = min(rows[key], key=lambda r: r["seed"]); H = r["H"]
            ax.loglog(H[:, 0], np.maximum(np.abs(H[:, iB["ratio"]]), 1e-17), color=cmap(c), label=key[3])
        ax.axhline(1.0, color="k", ls="--", lw=0.7); ax.set_title(inst.replace("_", " ") + (f", $\\beta={sch[0][2]:g}$" if sch else "")); ax.set_xlabel("$k$")
    axs[0, 0].set_ylabel(r"$(k+1)^2(p^*-d)/C$"); axs[0, 0].legend(fontsize=4.5)
    fig.tight_layout(); save(fig, figdir, "fig_b_ratio"); plt.close(fig)
    lines.append(f"b: {len(T)} runs, table_b_inexact.md/.tex, fig_b_dgap_k, fig_b_dgap_time, fig_b_ratio")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--res", required=True); ap.add_argument("--fig", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--partial", action="store_true", help="preview only: plot the finished subset of RUN_LIST (never used by pack)")
    a = ap.parse_args()
    tasks = smoke_tasks() if a.smoke else build_tasks()
    missing = [t["id"] for t in tasks if not os.path.exists(os.path.join(a.res, "results", t["id"] + ".npz"))]
    if missing and not (a.smoke or a.partial):
        print(f"STRICT: {len(missing)} of {len(tasks)} tasks missing (first: {missing[:5]}); no figures produced")
        sys.exit(1)
    tasks = [t for t in tasks if t["id"] not in missing]
    os.makedirs(a.fig, exist_ok=True)
    plt = mpl(); lines = []
    part_a_struct(a.res, a.fig, tasks, plt, lines)
    part_a_range(a.res, a.fig, tasks, plt, lines)
    part_b(a.res, a.fig, tasks, plt, lines)
    print("\n".join(lines))
    print("PLOTS OK")


if __name__ == "__main__":
    main()
