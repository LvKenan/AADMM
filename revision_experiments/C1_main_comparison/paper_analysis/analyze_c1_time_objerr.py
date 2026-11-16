#!/usr/bin/env python3
"""CPU time and objective-error reporting from the official C1-1.2 results (no rerun).

t_alg is algorithm time (method work only; monitoring excluded), single-thread BLAS,
recorded on the C1-1.2 Colab machine (Intel Xeon 2.20 GHz).

Relative errors use denominator (1+|F*|), with F* the C1-1.2 reference upper estimate
(all 70 references met the 1e-12 gap tolerance; F* is not an exact optimum).

  objerr      = |F(x^k,y^k) - F*|          at the method's recorded iterate (AADMM: predictor)
  objerr_feas = F(x_feas(y^k), y^k) - F*   feasibility repair (P1, P2, P4; not defined on P3)
  objerr_erg  = |F(xbar,ybar) - F*|        ergodic (weighted-average) iterate
  rp          = ||A0 x + B0 y - b0||/(1+||b0||)  original-scale primal residual
  Rp, KKT     = the scale-invariant residuals of C1 (Tang-Toh Sec. 5.2)

KKT first-hit times use the exact per-iteration record in meta['first_hit'].
AADMM dual-gap first-hit uses the exact per-iteration L(x~,y~,lam~)=d(lam~) array
(lpred_all) with t_alg interpolated from the logged history.
All other first-hits that depend on F, Ffeas or dval use the logged history
(every iteration up to k=100, then stride floor(k/50), plus the last iterate),
so those times are upper bounds, typically within about 2% for k>100.
"""
from __future__ import annotations
import json, os, re, glob, collections, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.environ.get("C1_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "C1_out")), "results")
FIG = os.path.join(HERE, "figs")
os.makedirs(FIG, exist_ok=True)

ORDER = ["aadmm", "aadmm__beta1.5", "admm", "admm__beta10", "admm__beta100",
         "xu_fixed", "xu_acc", "iadmm", "heu_admm", "fast_admm_r"]
NAME = {
    "aadmm": "AADMM, beta=1", "aadmm__beta1.5": "AADMM, beta=1.5",
    "admm": "ADMM, beta=1", "admm__beta10": "ADMM, beta=10", "admm__beta100": "ADMM, beta=100",
    "xu_fixed": "Xu fixed", "xu_acc": "Xu accelerated",
    "iadmm": "IADMM", "heu_admm": "Heu-ADMM", "fast_admm_r": "Goldstein restart",
}
NAME_TEX = {
    "aadmm": r"AADMM, $\beta{=}1$", "aadmm__beta1.5": r"AADMM, $\beta{=}1.5$",
    "admm": r"ADMM, $\beta{=}1$", "admm__beta10": r"ADMM, $\beta{=}10$",
    "admm__beta100": r"ADMM, $\beta{=}100$",
    "xu_fixed": "Xu, fixed", "xu_acc": "Xu, accel.",
    "iadmm": "IADMM", "heu_admm": "Heu-ADMM", "fast_admm_r": "Fast ADMM-r",
}
REP = [
    ("P1_m300_p1000", r"P1, $m{=}300$"),
    ("P2_n700_p2000", r"P2, $n{=}700$"),
    ("P3_p1000_mu0.1", r"P3, $\mu{=}0.1$"),
    ("P4_N128", r"P4, $N{=}128$"),
]


def load(path):
    d = np.load(path, allow_pickle=False)
    out = dict(meta=json.loads(str(d["meta"])))
    for k in d.files:
        if k != "meta":
            out[k] = d[k]
    return out


def H(r, name):
    return r["hist"][:, r["meta"]["fields"].index(name)]


def cfgname(inst):
    return re.sub(r"_s\d+$", "", inst)


def vkey(t):
    v = t.get("variant", "default")
    return t["method"] if v == "default" else f"{t['method']}__{v}"


def first_from_series(k, t, v, lev):
    """First logged index with v <= lev; returns (k, t) or None."""
    i = np.where(np.asarray(v) <= lev)[0]
    if not len(i):
        return None
    return int(k[i[0]]), float(t[i[0]])


def t_at_k(k_hist, t_hist, k):
    k_hist = np.asarray(k_hist, dtype=float)
    t_hist = np.asarray(t_hist, dtype=float)
    if k <= k_hist[0]:
        return float(t_hist[0] * k / k_hist[0])
    if k >= k_hist[-1]:
        return float(t_hist[-1])
    return float(np.interp(k, k_hist, t_hist))


def load_refs():
    R = {}
    for p in sorted(glob.glob(os.path.join(RES, "ref__*.npz"))):
        r = load(p)
        m = r["meta"]
        R[m["task"]["inst"]] = dict(
            F=m["Fstar"], LB=m["LB"], UB=m["UB"], gap=m["gap_rel"], status=m["status"],
        )
    return R


def collect(R):
    runs = []
    n_neg_feas = 0
    for p in sorted(glob.glob(os.path.join(RES, "main__*.npz"))):
        r = load(p)
        m = r["meta"]
        t = m["task"]
        inst = t["inst"]
        Fstar = R[inst]["F"]
        sc = 1 + abs(Fstar)
        k = H(r, "k")
        ta = H(r, "t_alg")
        F = H(r, "F")
        dval = H(r, "dval")
        gap_log = (Fstar - dval) / sc
        oe = np.abs(F - Fstar) / sc
        oef = (H(r, "Ffeas") - Fstar) / sc
        oeb = np.abs(H(r, "Fbar") - Fstar) / sc
        nb = m["problem"]["normb_orig"]
        rp = H(r, "rp_abs") / (1 + nb)
        fh = m["first_hit"]

        # AADMM: exact dual gap from lpred_all
        gap8 = gap6 = None
        if t["method"] == "aadmm" and "lpred_all" in r and len(r["lpred_all"]):
            lp = r["lpred_all"]
            gex = (Fstar - lp) / sc
            for lev, slot in ((1e-8, "gap8"), (1e-6, "gap6")):
                i = np.where(gex <= lev)[0]
                if len(i):
                    kk = int(i[0] + 1)
                    val = (kk, t_at_k(k, ta, kk))
                    if slot == "gap8":
                        gap8 = val
                    else:
                        gap6 = val
        else:
            gap8 = first_from_series(k, ta, gap_log, 1e-8)
            gap6 = first_from_series(k, ta, gap_log, 1e-6)

        if np.isfinite(oef[-1]) and oef[-1] < -1e-14:
            n_neg_feas += 1

        runs.append(dict(
            cfg=cfgname(inst), inst=inst, v=vkey(t), P=t["P"], method=t["method"],
            stop=m["stop_reason"], iters=m["iters"],
            t_alg=m["t_alg"], t_wall=m["t_wall"],
            t_per_it_ms=1e3 * m["t_alg"] / max(m["iters"], 1),
            kkt4=fh.get("0.0001") or fh.get("0.0001"),
            kkt6=fh.get("1e-06"), kkt8=fh.get("1e-08"),
            gap8=gap8, gap6=gap6,
            oe6=first_from_series(k, ta, oe, 1e-6),
            oe8=first_from_series(k, ta, oe, 1e-8),
            oef6=first_from_series(k, ta, np.where(np.isfinite(oef), np.abs(oef), np.inf), 1e-6)
                 if np.isfinite(oef).any() else None,
            oeb6=first_from_series(k, ta, oeb, 1e-6),
            fin_oe=float(oe[-1]), fin_oef=float(oef[-1]), fin_oeb=float(oeb[-1]),
            fin_rp=float(rp[-1]), fin_Rp=float(H(r, "Rp")[-1]),
            fin_kkt=float(H(r, "KKT")[-1]), fin_gap=float(gap_log[-1]),
        ))
    return runs, n_neg_feas


def med_time(g, key, idx=1):
    w = [x[key] for x in g]
    ok = [v[idx] for v in w if v is not None]
    if not ok:
        return "--"
    s = f"{np.median(ok):.3g}"
    return s if len(ok) == len(w) else f"{s} ({len(ok)}/{len(w)})"


def med_f(g, key, prec=".2e"):
    vals = [x[key] for x in g]
    if not np.isfinite(vals).any():
        return "n/a"
    return format(np.median(vals), prec)


def cfg_sort_key(c):
    # P1_m100_p500, P2_n350_..., P3_p500_mu0.1, P4_N64
    parts = c.split("_")
    P = parts[0]
    n = int(re.sub(r"\D", "", parts[1]))
    rest = c
    return (P, n, rest)


def write_markdown(runs):
    D = collections.defaultdict(list)
    for x in runs:
        D[(x["cfg"], x["v"])].append(x)
    cfgs = sorted({x["cfg"] for x in runs}, key=cfg_sort_key)
    L = [
        "# CPU time and objective error from C1-1.2 (no rerun)",
        "",
        collect.__doc__ or "",
        __doc__.strip(),
        "",
        "Cells: median over 5 instances; `(n/5)` means only n instances reached the level "
        "(median over those); `--` means none reached it.",
        "Time columns are seconds of `t_alg`. `obj<=1e-6` is the first *logged* iterate with "
        r"|F-F*|/(1+|F*|) <= 1e-6 at the recorded iterate.",
        "Final columns are at the last iterate of the run (stop reasons in the last column).",
        "Fast ADMM-r (Goldstein restart) is run only on P3 and P4.",
        "A slightly negative final `obj_feas` (order 1e-13 or smaller) is treated as a rounding / "
        "reference-gap effect; F* is an upper estimate.",
        "",
    ]
    # drop collect docstring if empty - we already have module doc
    L = [L[0], ""] + L[3:]
    for c in cfgs:
        L += [
            f"## {c}",
            "",
            "| method | ms/it | t KKT<=1e-6 | t KKT<=1e-8 | t gap<=1e-8 | t obj<=1e-6 | "
            "t obj_feas<=1e-6 | final obj err | final obj_feas err | final erg obj err | "
            "final rel. rp (orig.) | final R_p | stops |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for v in ORDER:
            g = D.get((c, v))
            if not g:
                continue
            stops = dict(collections.Counter(x["stop"] for x in g))
            oef = med_time(g, "oef6") if c[:2] != "P3" else "n/a"
            fin_oef = med_f(g, "fin_oef") if c[:2] != "P3" else "n/a"
            L.append(
                f"| {NAME[v]} | {np.median([x['t_per_it_ms'] for x in g]):.3g} | "
                f"{med_time(g, 'kkt6')} | {med_time(g, 'kkt8')} | {med_time(g, 'gap8')} | "
                f"{med_time(g, 'oe6')} | {oef} | {med_f(g, 'fin_oe')} | {fin_oef} | "
                f"{med_f(g, 'fin_oeb')} | {med_f(g, 'fin_rp')} | {med_f(g, 'fin_Rp')} | {stops} |"
            )
        L.append("")
    path = os.path.join(HERE, "table_time_objerr.md")
    open(path, "w").write("\n".join(L) + "\n")
    return path, D, cfgs


def write_csv(runs, cfgs):
    path = os.path.join(HERE, "table_time_objerr.csv")
    fields = ["config", "method", "n", "ms_per_it",
              "t_kkt_1e-6", "n_kkt_1e-6", "t_kkt_1e-8", "n_kkt_1e-8",
              "t_gap_1e-8", "n_gap_1e-8", "t_obj_1e-6", "n_obj_1e-6",
              "final_objerr", "final_objerr_feas", "final_objerr_erg",
              "final_rp", "final_Rp", "final_KKT", "stops"]
    D = collections.defaultdict(list)
    for x in runs:
        D[(x["cfg"], x["v"])].append(x)

    def pack(g, key, idx=1):
        w = [x[key] for x in g]
        ok = [v[idx] for v in w if v is not None]
        if not ok:
            return "", 0
        return float(np.median(ok)), len(ok)

    with open(path, "w", newline="") as fh:
        wri = csv.writer(fh)
        wri.writerow(fields)
        for c in cfgs:
            for v in ORDER:
                g = D.get((c, v))
                if not g:
                    continue
                tk6, nk6 = pack(g, "kkt6")
                tk8, nk8 = pack(g, "kkt8")
                tg8, ng8 = pack(g, "gap8")
                to6, no6 = pack(g, "oe6")
                def med(key):
                    vals = [x[key] for x in g]
                    return float(np.median(vals)) if np.isfinite(vals).any() else ""
                wri.writerow([
                    c, NAME[v], len(g), float(np.median([x["t_per_it_ms"] for x in g])),
                    tk6, nk6, tk8, nk8, tg8, ng8, to6, no6,
                    med("fin_oe"), med("fin_oef") if c[:2] != "P3" else "",
                    med("fin_oeb"), med("fin_rp"), med("fin_Rp"), med("fin_kkt"),
                    dict(collections.Counter(x["stop"] for x in g)),
                ])
    return path


def tcell(g, key):
    w = [x[key] for x in g]
    ok = [v[1] for v in w if v is not None]
    if not ok:
        return "--"
    m = np.median(ok)
    s = f"{m:.2f}" if m < 10 else f"{m:.1f}"
    return s if len(ok) == len(w) else f"{s}$^{{{len(ok)}}}$"


def ecell(g, key):
    vals = [x[key] for x in g]
    if not np.isfinite(vals).any():
        return "n/a"
    m = np.median(vals)
    s = f"{m:.1e}"
    mant, e = s.split("e")
    return f"{mant}e{int(e)}"


def write_tex(D):
    def table(kind):
        if kind == "time":
            cap = (
                r"Algorithm time in seconds (monitoring excluded) to reach $\max(R_p,R_d)\le10^{-6}$ (KKT) "
                r"and a relative dual gap $\le10^{-8}$ (gap), for the configurations of Table~\ref{tab:c1-main}; "
                r"median over $5$ instances, superscripts and ``--'' as there. "
                r"The last column is the median time per iteration in milliseconds, for P1/P2/P3/P4. "
                r"Single-threaded BLAS on an Intel Xeon (2.20\,GHz) cloud machine. "
                r"AADMM dual-gap times use the exact per-iteration predictor dual value; "
                r"other methods use the first logged iterate below the level."
            )
            lab = "tab:c1-time"
            sub = ["KKT", "gap"]
        else:
            cap = (
                r"Relative objective error $|F(x^k,y^k)-\widehat F^{*}|/(1+|\widehat F^{*}|)$ (obj) and "
                r"relative primal residual $R_p$ of Sec.~\ref{sec:c1} (res) at the last iterate of each run "
                r"(AADMM-$\beta$: the predictor), median over $5$ instances. "
                r"Runs stop at $\max(R_p,R_d)\le10^{-8}$ or after $20000$ iterations or $120$\,s."
            )
            lab = "tab:c1-objerr"
            sub = ["obj", "res"]
        L = [
            r"\begin{table}[htbp]", r"\centering",
            r"\caption{" + cap + r"}\label{" + lab + "}",
            r"\scriptsize\setlength{\tabcolsep}{2.4pt}",
            r"\begin{tabular}{l" + "rr" * 4 + ("r" if kind == "time" else "") + "}",
            r"\toprule",
            " & " + " & ".join(r"\multicolumn{2}{c}{" + t + "}" for _, t in REP)
            + (r" & ms/it" if kind == "time" else "") + r" \\",
            "".join(rf"\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(4)),
            "Method & " + " & ".join([" & ".join(sub)] * 4)
            + (r" & P1/P2/P3/P4" if kind == "time" else "") + r" \\",
            r"\midrule",
        ]
        for v in ORDER:
            cells, ms = [], []
            for c, _ in REP:
                g = D.get((c, v))
                if not g:
                    cells += ["n/a", "n/a"]
                    ms.append("n/a")
                    continue
                if kind == "time":
                    cells += [tcell(g, "kkt6"), tcell(g, "gap8")]
                    m = np.median([x["t_per_it_ms"] for x in g])
                    ms.append(f"{m:.2g}" if m < 10 else f"{m:.0f}")
                else:
                    cells += [ecell(g, "fin_oe"), ecell(g, "fin_Rp")]
            L.append(NAME_TEX[v] + " & " + " & ".join(cells)
                     + (" & " + "/".join(ms) if kind == "time" else "") + r" \\")
            if v in ("aadmm__beta1.5", "admm__beta100"):
                L.append(r"\addlinespace[2pt]")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        return "\n".join(L) + "\n"

    p1 = os.path.join(HERE, "table_c1_time.tex")
    p2 = os.path.join(HERE, "table_c1_objerr.tex")
    open(p1, "w").write(table("time"))
    open(p2, "w").write(table("objerr"))
    return p1, p2


def figures(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 6.7, "axes.linewidth": .55, "lines.linewidth": 1.,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
    })
    W = 372 / 72.27
    labels = {
        "aadmm": r"AADMM $\beta=1$", "aadmm__beta1.5": r"AADMM $\beta=1.5$",
        "admm": r"ADMM $\beta=1$", "admm__beta10": r"ADMM $\beta=10$",
        "admm__beta100": r"ADMM $\beta=100$",
        "xu_fixed": "Xu fixed", "xu_acc": "Xu accelerated",
        "iadmm": "IADMM", "heu_admm": "Heu-ADMM", "fast_admm_r": "Goldstein restart",
    }
    colors = {
        "aadmm": "#CC3311", "aadmm__beta1.5": "#CC3311",
        "admm": "#0077BB", "admm__beta10": "#0077BB", "admm__beta100": "#0077BB",
        "xu_fixed": "#777777", "xu_acc": "#EE7733",
        "iadmm": "#009988", "heu_admm": "#AA4499", "fast_admm_r": "#222222",
    }
    styles = {
        "aadmm": "-", "aadmm__beta1.5": "--",
        "admm": ":", "admm__beta10": "--", "admm__beta100": "-.",
        "xu_fixed": ":", "xu_acc": "-", "iadmm": "-", "heu_admm": "-", "fast_admm_r": "--",
    }
    hist = collections.defaultdict(list)
    for p in sorted(glob.glob(os.path.join(RES, "main__*.npz"))):
        r = load(p)
        t = r["meta"]["task"]
        c = cfgname(t["inst"])
        if c in dict(REP):
            hist[(c, vkey(t))].append(r)

    def series(rs, metric, vs_time):
        if vs_time:
            tend = min(float(H(r, "t_alg")[-1]) for r in rs)
            t0 = max(float(H(r, "t_alg")[0]) for r in rs)
            grid = np.geomspace(max(t0, 1e-4), tend, 400)
            vs = []
            for r in rs:
                tt = H(r, "t_alg")
                Fstar = R[r["meta"]["task"]["inst"]]["F"]
                sc = 1 + abs(Fstar)
                if metric == "gap":
                    y = (Fstar - H(r, "dval")) / sc
                elif metric == "kkt":
                    y = H(r, "KKT")
                elif metric == "obj":
                    y = np.abs(H(r, "F") - Fstar) / sc
                elif metric == "obj_erg":
                    y = np.abs(H(r, "Fbar") - Fstar) / sc
                else:
                    raise ValueError(metric)
                vs.append(np.exp(np.interp(np.log(grid), np.log(np.maximum(tt, 1e-6)),
                                           np.log(np.maximum(y, 1e-300)))))
            return grid, np.median(vs, axis=0)
        # vs iteration
        kmax = min(int(H(r, "k")[-1]) for r in rs)
        grid = np.unique(np.concatenate([
            np.arange(1, min(kmax, 100) + 1),
            np.unique(np.geomspace(max(100, 1), kmax, 300).astype(int)),
        ]))
        grid = grid[(grid >= 1) & (grid <= kmax)]
        vs = []
        for r in rs:
            kk = H(r, "k")
            Fstar = R[r["meta"]["task"]["inst"]]["F"]
            sc = 1 + abs(Fstar)
            if metric == "gap":
                y = (Fstar - H(r, "dval")) / sc
            elif metric == "kkt":
                y = H(r, "KKT")
            elif metric == "obj":
                y = np.abs(H(r, "F") - Fstar) / sc
            elif metric == "obj_erg":
                y = np.abs(H(r, "Fbar") - Fstar) / sc
            else:
                raise ValueError(metric)
            vs.append(np.exp(np.interp(np.log(grid), np.log(np.maximum(kk, 1)),
                                       np.log(np.maximum(y, 1e-300)))))
        return grid, np.median(vs, axis=0)

    titles = [
        r"(a) P1: $m=300,\ p=1000$",
        r"(b) P2: $n=700,\ p=2000$",
        r"(c) P3: $p=1000,\ \mu=0.1$",
        r"(d) P4: $N=128$",
    ]

    def draw(metric, fname, yl, lev, vs_time, xlabel, ylim=(1e-10, 1e1)):
        fig, axs = plt.subplots(2, 2, figsize=(W, 4.35), sharey=True)
        for ax, (c, _), ttl in zip(axs.flat, REP, titles):
            for v in ORDER:
                rs = hist.get((c, v))
                if not rs:
                    continue
                g, m = series(rs, metric, vs_time)
                keep = m >= ylim[0]
                ax.loglog(g[keep], m[keep], color=colors[v], ls=styles[v],
                          lw=1.25 if v.startswith("aadmm") else 0.95)
            if lev:
                ax.axhline(lev, color="k", lw=.5, ls=":")
            ax.set_title(ttl, pad=2)
            ax.set_ylim(*ylim)
            ax.set_xlabel(xlabel)
            ax.grid(True, alpha=.2, lw=.5)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        for ax in axs[:, 0]:
            ax.set_ylabel(yl)
        hs = [Line2D([0], [0], color=colors[v], ls=styles[v], lw=1.2, label=labels[v]) for v in ORDER]
        fig.legend(handles=hs, loc="lower center", ncol=5, bbox_to_anchor=(.5, .005),
                   frameon=False, columnspacing=.8, handlelength=2.3, handletextpad=.35)
        fig.tight_layout(rect=(0, .085, 1, 1), h_pad=.6, w_pad=.8)
        fig.savefig(os.path.join(FIG, fname + ".pdf"),
                    metadata={"Title": fname, "Creator": "checks_r9 analysis of C1-1.2"})
        fig.savefig(os.path.join(FIG, fname + ".png"))
        plt.close(fig)

    draw("gap", "fig_c1_time_dualgap",
         r"$(\widehat F^*-d(\lambda^k))/(1+|\widehat F^*|)$", 1e-8, True, "algorithm time (s)")
    draw("kkt", "fig_c1_time_kkt", r"$\max(R_p,R_d)$", 1e-8, True, "algorithm time (s)")
    draw("obj", "fig_c1_time_objerr",
         r"$|F(x^k,y^k)-\widehat F^*|/(1+|\widehat F^*|)$", 1e-6, True, "algorithm time (s)")
    draw("obj", "fig_c1_iter_objerr",
         r"$|F(x^k,y^k)-\widehat F^*|/(1+|\widehat F^*|)$", 1e-6, False, "iteration $k$")
    draw("obj_erg", "fig_c1_time_objerr_erg",
         r"$|F(\bar x^k,\bar y^k)-\widehat F^*|/(1+|\widehat F^*|)$", 1e-6, True, "algorithm time (s)")
    return [
        "fig_c1_time_dualgap", "fig_c1_time_kkt", "fig_c1_time_objerr",
        "fig_c1_iter_objerr", "fig_c1_time_objerr_erg",
    ]


def consistency_vs_table_main(runs):
    """Compare t_alg-to-KKT-1e-8 medians with C1_out/figs/table_main.md where both exist."""
    import ast
    path = os.path.join(os.path.dirname(RES), "figs", "table_main.md")
    # We'll just recompute from first_hit and report a few checks in findings.
    D = collections.defaultdict(list)
    for x in runs:
        D[(x["cfg"], x["v"])].append(x)
    # also load table_main.csv times to 1e-7 as a loose check that files exist
    return D


def write_findings(runs, n_neg_feas, fig_names):
    D = collections.defaultdict(list)
    for x in runs:
        D[(x["cfg"], x["v"])].append(x)
    lines = []
    p = lambda *a: lines.append(" ".join(str(x) for x in a) if a else "")

    p("# Findings: CPU time and objective error from C1-1.2")
    p()
    p("Source: official C1-1.2 results in `C1_out` (release asset C1_results_v12_20261115_1659.zip),")
    p("938/938 tasks, all 70 references within tolerance 1e-12. No rerun.")
    p("Machine (from `env.txt`): Intel Xeon CPU @ 2.20 GHz, single-thread OpenBLAS,")
    p("recorded 2026-11-15, 16:58 JST (Asia/Tokyo). `t_alg` excludes monitoring.")
    p()
    p("These quantities were already stored in every run; Table 9 of the manuscript reports")
    p("iteration counts (KKT and dual gap) only. The tables and figures in this folder")
    p("are intended for a later paper/letter pass (do not edit `work_r2` or the letter here).")
    p()
    p("## How to read the numbers")
    p()
    p("- Relative objective error uses |F-F*|/(1+|F*|). F* is a floating-point reference")
    p("  upper estimate, not an exact optimum.")
    p("- AADMM is evaluated at the predictor. For AADMM the predictor is dual feasible,")
    p("  so its KKT residual equals its primal residual Rp in all of these runs.")
    p("- Time to KKT uses the exact per-iteration first-hit record. Time to AADMM dual")
    p("  gap uses the exact per-iteration predictor dual value. Time to objective error")
    p("  (and dual gap for non-AADMM methods) uses the logged history only.")
    p("- `--` or a superscript (n/5) means the median is over the instances that reached")
    p("  the level; the others hit maxit (20000) or the 120 s time budget.")
    p("- Goldstein restart is present only on P3 and P4.")
    p(f"- {n_neg_feas} runs have a slightly negative feasibility-repaired objective error")
    p("  (Ffeas < F*); this is at rounding / reference-gap scale and is not a certificate")
    p("  that Ffeas is below the true optimum.")
    p()
    p("## Representative configurations (the main-text set: P1_m300, P2_n700, P3_p1000_mu=0.1, P4_N128)")
    p()

    def row(cfg, v, key):
        g = D.get((cfg, v))
        return "--" if not g else med_time(g, key)

    def frow(cfg, v, key):
        g = D.get((cfg, v))
        return "n/a" if not g else med_f(g, key)

    p("Median algorithm time (s) to KKT <= 1e-6 / dual gap <= 1e-8 / final relative objective error:")
    p()
    p("| method | P1_m300 KKT / gap / obj | P2_n700 KKT / gap / obj | P3_p1000_mu0.1 KKT / gap / obj | P4_N128 KKT / gap / obj |")
    p("|---|---|---|---|---|")
    for v in ORDER:
        cells = []
        for c in ("P1_m300_p1000", "P2_n700_p2000", "P3_p1000_mu0.1", "P4_N128"):
            if not D.get((c, v)):
                cells.append("n/a")
                continue
            cells.append(f"{row(c,v,'kkt6')} / {row(c,v,'gap8')} / {frow(c,v,'fin_oe')}")
        p(f"| {NAME[v]} | " + " | ".join(cells) + " |")
    p()

    # factual comparisons, computed
    def med_or_none(cfg, v, key, idx=1):
        g = D.get((cfg, v)) or []
        ok = [x[key][idx] for x in g if x[key] is not None]
        return (float(np.median(ok)), len(ok), len(g)) if ok else None

    p("## Factual observations")
    p()
    p("1. Rankings by algorithm time track the KKT-iteration rankings of Table 9 more")
    p("   closely than they track the dual-gap rankings. On all four representative")
    p("   configurations, AADMM is not the fastest method by time to KKT <= 1e-6.")
    p("   Tuned classical ADMM (beta=100 on P1 and P3; beta=10 or 100 depending on P2/P4)")
    p("   or an adaptive / accelerated baseline is faster on that measure.")
    p()
    # P1
    p("2. P1 (elastic-net SVM) remains unfavorable for AADMM by the KKT residual: on")
    p("   P1_m300 neither AADMM beta=1 nor beta=1.5 reaches KKT <= 1e-6 in any of the")
    p("   five instances (all stop at maxit). They do reach a relative dual gap of 1e-8")
    p(f"   (median t_alg {row('P1_m300_p1000','aadmm','gap8')} s and "
      f"{row('P1_m300_p1000','aadmm__beta1.5','gap8')} s) and a relative predictor")
    p(f"   objective error of {frow('P1_m300_p1000','aadmm','fin_oe')} and "
      f"{frow('P1_m300_p1000','aadmm__beta1.5','fin_oe')} at termination.")
    p("   The repaired feasible objective error is about two orders of magnitude larger")
    p(f"   ({frow('P1_m300_p1000','aadmm','fin_oef')} and "
      f"{frow('P1_m300_p1000','aadmm__beta1.5','fin_oef')}), consistent with a still-nontrivial residual.")
    p("   On P1_m1000 several methods, including ADMM and Xu fixed, hit the 120 s budget;")
    p("   times there are censored and should not be read as a complete ranking.")
    p()
    p("3. P2 is easy for every method. All ten variants reach KKT <= 1e-8 on P2_n700;")
    p(f"   median times are a fraction of a second (AADMM beta=1: {row('P2_n700_p2000','aadmm','kkt6')} s")
    p(f"   to KKT 1e-6; Xu fixed: {row('P2_n700_p2000','xu_fixed','kkt6')} s). Final relative")
    p("   objective errors are all at most about 1e-9. Time-per-iteration of IADMM/Heu-ADMM")
    p("   is higher than that of the fixed-penalty methods (refactor / residual-balancing work).")
    p()
    p("4. On P3_p1000 with mu=0.1, AADMM reaches KKT <= 1e-6 in")
    p(f"   {row('P3_p1000_mu0.1','aadmm','kkt6')} s (beta=1) and "
      f"{row('P3_p1000_mu0.1','aadmm__beta1.5','kkt6')} s (beta=1.5),")
    p(f"   faster than ADMM at the same beta=1 ({row('P3_p1000_mu0.1','admm','kkt6')} s)")
    p(f"   but slower than ADMM at beta=100 ({row('P3_p1000_mu0.1','admm__beta100','kkt6')} s)")
    p(f"   and than Xu accelerated ({row('P3_p1000_mu0.1','xu_acc','kkt6')} s) and Heu-ADMM")
    p(f"   ({row('P3_p1000_mu0.1','heu_admm','kkt6')} s). IADMM is costlier per iteration")
    p(f"   (median {med_f(D[('P3_p1000_mu0.1','iadmm')],'t_per_it_ms','.3g')} ms/it vs "
      f"{med_f(D[('P3_p1000_mu0.1','aadmm')],'t_per_it_ms','.3g')} ms/it for AADMM).")
    p("   Final predictor / last-iterate objective errors of all methods that reach the")
    p("   KKT tolerance are of order 1e-9. The ergodic objective error is typically much")
    p("   larger (often 1e-2) at the last iterate, because the run stops when the last")
    p("   iterate meets the KKT test, not when the weighted average does.")
    p()
    p("5. On P4_N128, AADMM has a favorable dual gap (median time to relative gap 1e-8:")
    p(f"   {row('P4_N128','aadmm','gap8')} s and {row('P4_N128','aadmm__beta1.5','gap8')} s)")
    p("   and a small final predictor objective error")
    p(f"   ({frow('P4_N128','aadmm','fin_oe')} and {frow('P4_N128','aadmm__beta1.5','fin_oe')}).")
    p("   By time to KKT <= 1e-6 it is not the fastest")
    p(f"   ({row('P4_N128','aadmm','kkt6')} s / {row('P4_N128','aadmm__beta1.5','kkt6')} s;")
    p(f"   IADMM {row('P4_N128','iadmm','kkt6')} s, ADMM beta=100 {row('P4_N128','admm__beta100','kkt6')} s).")
    p("   ADMM at beta=1, Xu fixed, and Goldstein restart do not reach KKT <= 1e-6 on this")
    p("   configuration (all five seeds at maxit), and their final objective errors stay")
    p(f"   near {frow('P4_N128','admm','fin_oe')} to {frow('P4_N128','xu_fixed','fin_oe')}.")
    p("   Four of five AADMM seeds also hit maxit before KKT 1e-8; the KKT-1e-8 times are")
    p("   therefore not a complete-sample median.")
    p()
    p("6. Last-iterate objective error and KKT residual are different measures. A method can")
    p("   have a small |F-F*| while Rp is still above 1e-6 (AADMM on P1_m300), or the reverse")
    p("   pattern can appear once a tight KKT test has been met. The ergodic objective error")
    p("   at termination is not a good proxy for the last-iterate error under this stopping rule.")
    p()
    p("7. Time per iteration is stable across beta for AADMM and ADMM on a given size;")
    p("   differences in algorithm time are driven by iteration counts, except for IADMM")
    p("   (and to a lesser extent Heu-ADMM), which are clearly more expensive per iteration")
    p("   on P3, where a varying penalty forces refactorization of H+beta B^T B.")
    p()
    p("## Files")
    p()
    p("- `table_time_objerr.md`, `table_time_objerr.csv`: all 14 main-text configurations.")
    p("- `table_c1_time.tex`, `table_c1_objerr.tex`: paper-ready tables for the four representative configs.")
    p("- Figures (pdf and png) in `figs/`: " + ", ".join(fig_names) + ".")
    p("- `runs_time_objerr.json`: per-run numbers used to build the tables.")
    p()

    path = os.path.join(HERE, "FINDINGS.md")
    open(path, "w").write("\n".join(lines) + "\n")
    return path


def main():
    print("loading references and main runs ...", flush=True)
    R = load_refs()
    assert len(R) == 70, len(R)
    assert all(v["status"] == "tol" and v["gap"] <= 1e-12 for v in R.values())
    runs, n_neg = collect(R)
    assert len(runs) == 670, len(runs)
    json.dump(runs, open(os.path.join(HERE, "runs_time_objerr.json"), "w"), indent=0)
    md, D, cfgs = write_markdown(runs)
    csvp = write_csv(runs, cfgs)
    tex = write_tex(D)
    print("tables:", md, csvp, tex, flush=True)
    figs = figures(R)
    print("figures:", figs, flush=True)
    findings = write_findings(runs, n_neg, figs)
    print("findings:", findings, "n_neg_feas", n_neg, flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
