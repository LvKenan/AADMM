#!/usr/bin/env python3
"""R9: analysis of Carbon's C2_supp results (C2_results_20261115_2350.zip, package C2-supp-1.0).
Reads C2_out/results/*.npz only (no rerun). Writes paper-ready figures (figs_c2/), LaTeX tables (tex/),
and r9_numbers.json with every number quoted in R9_carbon_c2_review_zh.md and the proposed texts."""
import os, sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.environ.get("C2_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "C2_out")), "results")
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs_c2"); TEX = os.path.join(HERE, "tex")
os.makedirs(FIG, exist_ok=True); os.makedirs(TEX, exist_ok=True)
TW = 372 / 72.27  # \textwidth of sn-jnl in inches (approx.)
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                     "font.size": 7, "axes.titlesize": 7, "legend.fontsize": 6, "pdf.fonttype": 42, "savefig.dpi": 300,
                     "axes.grid": True, "grid.alpha": 0.3})
COL = {"aadmm": "#CC3311", "dual_fista": "#0077BB", "dual_pg": "#009988", "admm": "#555555"}
LAB = {"aadmm": r"AADMM-$\beta$", "dual_fista": r"dual FISTA, step $\beta$", "dual_pg": r"dual prox-grad, step $\beta$"}


def load(path):
    z = np.load(path, allow_pickle=False)
    return json.loads(str(z["meta"])), z


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


N = {}
# ------------------------------------------------------------------ (a) beta range
BETAS = [0.5, 1.0, 1.25, 1.5, 1.75, 1.9, 2.5, 4.0]
A = {}
for p in glob.glob(os.path.join(RES, "a_range__*.npz")):
    m, z = load(p); t = m["task"]
    cls = t["inst"].rsplit("_s", 1)[0]
    F = m["fields"]; H = z["hist"]; Fs = m["ref"]["Fstar"]
    dcol = F.index("Lpred") if t["method"] == "aadmm" else F.index("dval")
    gap = (Fs - H[:, dcol]) / (1 + abs(Fs))
    kkt = z["kkt_all"]
    A[(cls, t["method"], t["params"]["beta"], t["spec"]["seed"])] = dict(
        stop=m["stop_reason"], it=m["iters"], kkt=float(kkt[-1]) if len(kkt) else np.nan,
        k=H[:, F.index("k")], gap=gap, t=m["t_alg"],
        ratio=None)
CLASSES = ["P1_m100_p500", "P2_n350_p1000", "P2_n700_p2000", "P3_p500_mu0.1", "P3_p500_mu1", "P4_N64", "P4_N128"]
CLN = {"P1_m100_p500": "P1, $m{=}100$", "P2_n350_p1000": "P2, $n{=}350$", "P2_n700_p2000": "P2, $n{=}700$",
       "P3_p500_mu0.1": r"P3, $\mu{=}0.1$", "P3_p500_mu1": r"P3, $\mu{=}1$", "P4_N64": "P4, $N{=}64$", "P4_N128": "P4, $N{=}128$"}
summ = {}
for c in CLASSES:
    for meth in ("aadmm", "dual_fista", "dual_pg"):
        for b in BETAS:
            R = [A[(c, meth, b, s)] for s in range(3)]
            stops = [r["stop"] for r in R]
            e = dict(n_tol=stops.count("tol"), n_maxit=stops.count("maxit"), n_div=stops.count("diverged"),
                     it_tol=float(np.median([r["it"] for r in R if r["stop"] == "tol"])) if "tol" in stops else None,
                     kkt=float(np.median([r["kkt"] for r in R])), gap=float(np.median([r["gap"][-1] for r in R])),
                     it_div=[r["it"] for r in R if r["stop"] == "diverged"])
            summ[f"{c}|{meth}|{b:g}"] = e
N["a_range"] = summ

# outcome counts within (1,2) and beyond 2
def count(meth, cond):
    out = {"tol": 0, "maxit": 0, "diverged": 0}
    for (c, mm, b, s), r in A.items():
        if mm == meth and cond(b):
            out[r["stop"]] += 1
    return out
N["counts"] = {m: {"0<b<=1": count(m, lambda b: b <= 1), "1<b<2": count(m, lambda b: 1 < b < 2), "b>2": count(m, lambda b: b > 2)}
               for m in ("aadmm", "dual_fista", "dual_pg")}

# Theorem ratio for AADMM a_range runs with beta<2 : C with lam^1=z^1=0 requires lam*, By* (not stored) -> use Carbon's table
ratios = []
for line in open(os.path.join(os.path.dirname(RES), "figs", "table_a_range.md")):
    parts = [x.strip() for x in line.split("|")]
    if len(parts) > 8 and parts[2] == "aadmm" and parts[8] not in ("--", ""):
        try:
            ratios.append((parts[1], float(parts[3]), float(parts[8])))
        except ValueError:
            pass
N["aadmm_theorem_ratio_max"] = max(r[2] for r in ratios)
N["aadmm_theorem_ratio_argmax"] = max(ratios, key=lambda r: r[2])

# ---------------- Fig B1: final KKT vs beta (5 classes)
SHOW = ["P1_m100_p500", "P2_n700_p2000", "P3_p500_mu0.1", "P3_p500_mu1", "P4_N128"]
fig, axs = plt.subplots(1, 5, figsize=(TW, 0.30 * TW), sharey=True)
for ax, c in zip(axs, SHOW):
    for meth in ("aadmm", "dual_fista", "dual_pg"):
        xs, ys, xd = [], [], []
        for b in BETAS:
            e = summ[f"{c}|{meth}|{b:g}"]
            if e["n_div"] >= 2:
                xd.append(b)
            else:
                xs.append(b); ys.append(max(e["kkt"], 1e-12))
        ax.semilogy(xs, ys, "-o", color=COL[meth], ms=2.5, lw=0.9, label=LAB[meth])
        ax.semilogy(xd, [3e2] * len(xd), "x", color=COL[meth], ms=4, mew=1.0)
    ax.axvline(1.0, color="k", ls=":", lw=0.6); ax.axvline(2.0, color="k", ls="--", lw=0.6)
    ax.set_title(CLN[c]); ax.set_xlabel(r"$\beta$"); ax.set_ylim(1e-10, 1e3); ax.set_xticks([1, 2, 4])
axs[0].set_ylabel(r"final KKT residual (median)")
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.10))
fig.tight_layout(w_pad=0.3)
save(fig, "fig_r9_dualfista_beta")

# ---------------- Fig B2: relative dual gap vs k (P2 n700 s0, P4 N128 s0)
TB = [1.0, 1.5, 1.9, 2.5]
fig, axs = plt.subplots(2, 4, figsize=(TW, 0.52 * TW), sharey=True)
for i, c in enumerate(["P2_n700_p2000", "P4_N128"]):
    for j, b in enumerate(TB):
        ax = axs[i, j]
        for meth, ls in (("aadmm", "-"), ("dual_fista", "--"), ("dual_pg", "-.")):
            r = A[(c, meth, b, 0)]
            g = np.abs(r["gap"]); k = r["k"]
            ax.loglog(k, np.maximum(g, 1e-14), ls, color=COL[meth], lw=0.9, label=LAB[meth])
            if r["stop"] == "diverged":
                ax.plot(k[-1], max(g[-1], 1e-14), "x", color=COL[meth], ms=4)
        ax.set_title(f"{CLN[c]}, $\\beta={b:g}$"); ax.set_ylim(1e-13, 1e3)
        if i == 1: ax.set_xlabel("$k$")
    axs[i, 0].set_ylabel("|relative dual gap|")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(h_pad=0.5, w_pad=0.3)
save(fig, "fig_r9_dualfista_traj")

# ---------------- a_struct numbers
S = {}
for p in glob.glob(os.path.join(RES, "a_struct__*.npz")):
    m, z = load(p); t = m["task"]; cls = t["inst"].rsplit("_s", 1)[0]
    S.setdefault(f"{cls}|{t['params']['beta']:g}", []).append(dict(dev_max=m["dev_max"], ident=m["max_identity_resid"], lemb=m["max_lemma_b_resid"]))
N["a_struct"] = {k: dict(dev_max_max=max(r["dev_max"] for r in v), ident_max=max(r["ident"] for r in v), lemb_max=max(r["lemb"] for r in v)) for k, v in S.items()}

# ------------------------------------------------------------------ (b) inexact
B = {}
for p in glob.glob(os.path.join(RES, "b__*.npz")):
    m, z = load(p); t = m["task"]; sc = t["params"]["sched"]
    sname = f"tol_c{sc['c']:.0e}_p{sc['p']}" if sc["mode"] == "tol" else f"count_n{sc['n']}"
    inst = t["inst"].rsplit("_s", 1)[0]
    F = m["fields"]; H = z["hist"]
    B.setdefault((inst, t["method"], t["params"]["beta"], sname), []).append(dict(
        stop=m["stop_reason"], it=m["iters"], dgap=float(z["dgap_all"][-1]), inner=m["summary"]["inner_total"], t=m["t_alg"],
        unmet=m["summary"]["n_unmet"], objerr=float(H[-1, F.index("objerr")]), rp=float(H[-1, F.index("rp")]),
        Lgmin=float(np.nanmin(H[:, F.index("Lgap")])), ratio=float(np.nanmax(H[:, F.index("ratio")])) if t["method"] == "aadmm" else None,
        k=H[:, F.index("k")], tt=H[:, F.index("t_alg")], dg=H[:, F.index("dgap")], seed=t["spec"]["seed"]))
def med(R, key):
    v = [r[key] for r in R if r[key] is not None]
    return float(np.median(v)) if v else None
N["b"] = {"|".join(map(str, k)): dict(stops=sorted(set(r["stop"] for r in v)), n=len(v), it=med(v, "it"), dgap=med(v, "dgap"),
                                       inner=med(v, "inner"), t=med(v, "t"), objerr=med(v, "objerr"), rp=med(v, "rp"),
                                       Lgmin=min(r["Lgmin"] for r in v), ratio_max=(max(r["ratio"] for r in v) if v[0]["ratio"] is not None else None),
                                       unmet=sum(r["unmet"] for r in v))
          for k, v in B.items()}
# AADMM-1 vs ADMM-1 identical-count comparison (structural coincidence)
pairs = []
for (inst, meth, b, s), v in B.items():
    if meth == "admm" and s.startswith("count"):
        w = B[(inst, "aadmm", 1.0, s)]
        pairs.append((inst, s, med(w, "dgap"), med(v, "dgap"), med(w, "objerr"), med(v, "objerr"), med(w, "rp"), med(v, "rp")))
N["b_pairs_aadmm1_vs_admm1"] = pairs

SCH = [("tol_c1e-11_p0", r"$\varepsilon_k=10^{-11}$"), ("tol_c1e-04_p0", r"$\varepsilon_k=10^{-4}$"), ("tol_c1e-07_p0", r"$\varepsilon_k=10^{-7}$"),
       ("tol_c1e-04_p1", r"$10^{-4}/k$"), ("tol_c1e-04_p2", r"$10^{-4}/k^{2}$"), ("tol_c1e-04_p3", r"$10^{-4}/k^{3}$"),
       ("count_n1", r"1 inner step"), ("count_PAPER", r"$n_{\rm paper}$ steps"), ("count_5PAPER", r"$5n_{\rm paper}$ steps")]
NP = {"LASSO_m1000_n10000": 10, "TVD_N128": 20, "LASSO_m500_n5000": 10, "TVD_N64": 20, "LASSO_m200_n2000": 10}
def sname_fix(inst, s):
    if s == "count_PAPER": return f"count_n{NP[inst]}"
    if s == "count_5PAPER": return f"count_n{5*NP[inst]}"
    return s
cmap = plt.get_cmap("tab10")
INST2 = [("LASSO_m1000_n10000", r"LASSO"), ("TVD_N128", r"TV")]
METH2 = [("aadmm", 1.0, "AADMM-1"), ("aadmm", 1.5, "AADMM-1.5"), ("admm", 1.0, "ADMM-1")]
for xaxis in ("k", "time"):
    fig, axs = plt.subplots(2, 3, figsize=(TW, 0.62 * TW), sharey=True)
    for i, (inst, iname) in enumerate(INST2):
        for j, (meth, b, mname) in enumerate(METH2):
            ax = axs[i, j]
            for q, (s, slab) in enumerate(SCH):
                key = (inst, meth, b, sname_fix(inst, s))
                if key not in B:
                    continue
                r = sorted(B[key], key=lambda r: r["seed"])[0]
                x = r["k"] if xaxis == "k" else np.maximum(r["tt"], 1e-3)
                ax.loglog(x, np.maximum(np.abs(r["dg"]), 1e-13), color=cmap(q), ls=("-" if q < 5 else "--"), lw=0.9, label=slab)
            ax.set_title(f"{iname}, {mname}"); ax.set_ylim(1e-12, 3)
            if i == 1: ax.set_xlabel("outer iteration $k$" if xaxis == "k" else "algorithm time [s]")
        axs[i, 0].set_ylabel("relative dual gap")
    h, l = [], []
    for ax in axs.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l: h.append(hh); l.append(ll)
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(h_pad=0.6, w_pad=0.3)
    save(fig, f"fig_r9_inexact_{xaxis}")

# ---------------- LaTeX tables
def cell(e):
    if e["n_div"] == 3:
        return r"div.\ (" + f"{int(np.median(e['it_div']))}" + ")"
    if e["n_div"] > 0:
        return f"div.$^{{{e['n_div']}}}$"
    if e["n_tol"] == 3:
        return f"{e['it_tol']:.0f}"
    if e["n_tol"] > 0:
        return f"{e['it_tol']:.0f}$^{{{e['n_tol']}}}$"
    v = f"{e['kkt']:.0e}".replace("e-0", "e-").replace("e+00", "").replace("e+0", "e")
    return r"\textit{" + v + "}"
TB2 = [0.5, 1.0, 1.5, 1.75, 1.9, 2.5, 4.0]
rows = []
for c in ["P1_m100_p500", "P2_n700_p2000", "P3_p500_mu0.1", "P3_p500_mu1", "P4_N128"]:
    for meth, mn in (("aadmm", r"AADMM-$\beta$"), ("dual_fista", "dual FISTA"), ("dual_pg", "dual PG")):
        rows.append(f"{CLN[c] if meth=='aadmm' else ''} & {mn} & " + " & ".join(cell(summ[f'{c}|{meth}|{b:g}']) for b in TB2) + r" \\")
    rows.append(r"\addlinespace[2pt]")
tex = r"""\begin{table}[htbp]
\centering
\caption{AADMM-$\beta$ versus FISTA and the proximal-gradient method with constant step $\beta$ applied to the dual problem $\min(\Phi+\Psi)$, on the normalized problems of Sec.~\ref{sec:c1} ($L_\Psi\le1/1.001$, so the standard step-size ranges are $\beta\le1/L_\Psi$ for FISTA and $\beta<2/L_\Psi$ for the proximal-gradient method). Three random instances per configuration; runs stop at $\max(R_p,R_d)\le10^{-8}$ or after $5000$ iterations. Upright numbers: median iterations to $10^{-8}$ (superscript: number of instances that reached it, if fewer than three). Italic numbers: median KKT residual after $5000$ iterations. ``div.'': divergence guard $\|\lambda^k\|>10^{8}(1+\|\lambda^{*}\|)$ triggered or a non-finite value (median iteration in parentheses). Theorem~\ref{Theor-AADMM} covers AADMM-$\beta$ for $\beta<2$ only.}\label{tab:dualfista}
\scriptsize\setlength{\tabcolsep}{3pt}
\begin{tabular}{llrrrrrrr}
\toprule
Problem & Method & $\beta{=}0.5$ & $1$ & $1.5$ & $1.75$ & $1.9$ & $2.5$ & $4$ \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(TEX, "table_r9_dualfista.tex"), "w").write(tex)

def e2(v):
    return "--" if v is None else f"{v:.1e}".replace("e-0", "e-").replace("e+0", "e")
rows = []
for inst, iname in [("LASSO_m1000_n10000", r"LASSO, $1000\times10000$ (1 instance)"), ("TVD_N128", r"TV, $128\times128$ (3 instances)")]:
    rows.append(r"\multicolumn{10}{l}{" + iname + r"}\\")
    for s, slab in SCH:
        sn = sname_fix(inst, s)
        cells = []
        for meth, b, _ in METH2:
            v = N["b"].get(f"{inst}|{meth}|{b}|{sn}")
            cells += ["", "", ""] if v is None else [e2(v["dgap"]), f"{v['inner']:.0f}", f"{v['t']:.1f}"]
        rows.append(f"\\quad {slab} & " + " & ".join(cells) + r" \\")
    rows.append(r"\addlinespace[2pt]")
tex = r"""\begin{table}[htbp]
\centering
\caption{Inexact $x$-subproblems on the LASSO and TV splittings of Secs.~\ref{section_LR}--\ref{section_ID} ($B=I$, $g=\frac12\|\cdot\|^2$, closed-form $y$-subproblem). Inner solver: warm-started ISTA (LASSO) or Chambolle's method (TV) for a fixed number of steps ($n_{\rm paper}=10$ and $20$, the values of the original experiments), or warm-started FISTA with restart until the inner certificate is at most $\varepsilon_k(1+|p^{*}|)$. Runs stop at a relative dual gap $\le10^{-10}$ or after $500$ outer iterations. Columns: final relative dual gap $(p^{*}-d(\Pi\tilde\lambda^{k}))/(1+|p^{*}|)$ with $\Pi$ a feasibility projection of the multiplier, total inner iterations, algorithm time in seconds; medians over instances. AADMM-1 and ADMM-1 solve the same $x$-subproblems in these splittings (Sec.~\ref{sec:largescale}), and no convergence guarantee is claimed for any inexact run (Remark~\ref{rem:inexact}). Blank: not run. With $\varepsilon_k=10^{-11}$ on TV the inner solver did not always meet the tolerance within $5000$ inner steps.}\label{tab:inexact}
\scriptsize\setlength{\tabcolsep}{2.4pt}
\begin{tabular}{lrrrrrrrrr}
\toprule
 & \multicolumn{3}{c}{AADMM, $\beta{=}1$} & \multicolumn{3}{c}{AADMM, $\beta{=}1.5$} & \multicolumn{3}{c}{ADMM, $\beta{=}1$} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}
Inner rule & gap & inner & s & gap & inner & s & gap & inner & s \\
\midrule
""" + "\n".join(rows[:-1]) + r"""
\bottomrule
\end{tabular}
\end{table}
"""
open(os.path.join(TEX, "table_r9_inexact.tex"), "w").write(tex)

# ------------------------------------------------------------------ (c) toy: regenerate on the box (deterministic)
sys.path.insert(0, os.path.join(HERE, ".."))   # toy_fig.py of this folder
import toy_fig as T
O = (-95.0, 86.0, 46.0); K = 5
adm = T.admm_text(O, K); aad = T.aadmm_text(O, K); pred, corr = T.corrected_points(O, K); ponly = T.aadmm_text(O, K, correct=False)
err = lambda s: [float(np.linalg.norm(np.array(p) - np.array((1.0, 1.0, 2.0)))) for p in s]
toy = dict(errors=dict(admm=err(adm), aadmm=err(aad), pred_only=err(ponly)), B=corr[0], A=aad[1], P2=ponly[2], F2=aad[2])
var = {}
for fk in ("lin", "quad"):
    for b in (0.5, 1.0, 1.5, 1.9):
        var[f"{fk}|{b:g}"] = dict(aadmm=err(T.aadmm_beta(O, 40, b, fk)), pred_only=err(T.aadmm_beta(O, 40, b, fk, correct=False)), admm=err(T.admm_beta(O, 40, b, fk)))
toy["variants"] = var
N["toy"] = toy
from mpl_toolkits.mplot3d import Axes3D  # noqa
def fig3d(name, series, labels):
    fig = plt.figure(figsize=(0.48 * TW, 0.48 * TW * 0.9)); ax = fig.add_subplot(111, projection="3d")
    for lab, pts, col, mk in series:
        P = np.array(pts, dtype=float); ax.plot(P[:, 0], P[:, 1], P[:, 2], color=col, lw=0.9, marker=mk, ms=3, label=lab)
    ax.scatter([1], [1], [2], color="gold", edgecolor="k", s=30, zorder=5, label=r"$(x^*,y^*,\lambda^*)$")
    for txt, p in labels:
        ax.text(p[0], p[1], p[2], txt, fontsize=7)
    ax.set_xlabel("$x$", labelpad=-6); ax.set_ylabel("$y$", labelpad=-6); ax.set_zlabel(r"$\lambda$", labelpad=-6)
    ax.tick_params(labelsize=5, pad=-2); ax.set_box_aspect(None, zoom=0.85); ax.legend(fontsize=5, loc="upper left", frameon=False)
    fig.subplots_adjust(left=0, right=1, bottom=0.02, top=0.98)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"))
    plt.close(fig)
fig3d("fig_r9_E1_left", [("ADMM", adm, "#0077BB", "o"), ("AADMM-1 (predictor)", aad, "#CC3311", "s")],
      [(r"$\mathbf{O}$", O)])
fig3d("fig_r9_E1_right", [("prediction only", ponly, "#0077BB", "o"),
                          ("prediction-correction", [O, pred[1], corr[0]] + pred[2:], "#CC3311", "s")],
      [(r"$\mathbf{O}$", O), ("$A$", pred[1]), ("$B$", corr[0]), (r"$P_2=F_2$", (8, 14, 6))])

json.dump(N, open(os.path.join(HERE, "r9_numbers.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("counts:", json.dumps(N["counts"]))
print("aadmm theorem ratio max:", N["aadmm_theorem_ratio_argmax"])
print("toy errors:", {k: ["%.3g" % e for e in v] for k, v in toy["errors"].items()}, "B=", toy["B"])
for p in pairs: print("pair", p)
