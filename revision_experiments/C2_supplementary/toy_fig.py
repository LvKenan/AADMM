#!/usr/bin/env python3
"""(c) Task L2: regenerate Fig. F_E1 (sn-article.tex Sec. 4.1, toy problem (E1): min x + y^2 s.t. x/2 + y = 3/2,
solution (x*, y*, lam*) = (1, 1, 2)) with the CORRECTED updates printed in the revised text, and a non-degenerate
variant with the same solution (f(x) = x^2/2 instead of x).  Deterministic, seconds, no data needed.
  left : ADMM (x^{k+1}, y^{k+1}, lam^{k+1}) vs AADMM-1 predictor (x~^k, y~^k, lam~^k), k <= 5
  right: AADMM-1 prediction only (O -> A -> P2 -> P3 -> P4) vs prediction-correction (O -> A -B-> F2 -> F3 -> F4)
Initial point O = (-95, 86, 46) (the values stated in the revision memo for the original figure; the paper says
'random integers in [-100, 100]').
The ORIGINAL (erroneous) x-update x~ = lam - 2y + 1 is also iterated (orig_text) to document what the old figure showed.
Outputs (in --fig): fig_E1_corrected_{left,right}.pdf/.png, fig_E1_quad_{left,right}.pdf/.png, toy_E1_sequences.json."""
import os, sys, json, argparse
import numpy as np

A, B, b = 0.5, 1.0, 1.5


def admm_text(O, K):
    x, y, l = O; out = [O]
    for _ in range(K):
        x = 2 * l - 2 * y - 1; y = -x / 6 + l / 3 + 0.5; l = l - (0.5 * x + y - 1.5); out.append((x, y, l))
    return out


def aadmm_text(O, K, correct=True, orig=False):
    """The AADMM-1 scheme exactly as printed in the revised Sec. 4.1 (y^k stands for z^k); orig=True uses the
    original erroneous x-update x~ = lam - 2y + 1."""
    x0, y, l = O; t = 1.0; yt_prev, lt_prev = y, l; out = [O]; pts = dict(B=[])
    for _ in range(K):
        xt = (l - 2 * y + 1) if orig else (2 * l - 2 * y - 1)
        lt = l - (0.5 * xt + y - 1.5); yt = 0.5 * lt
        out.append((xt, yt, lt))
        if correct:
            t1 = (1 + np.sqrt(1 + 4 * t * t)) / 2
            yn = (t1 - 1) / (2 * t1) * yt - (t - 1) / (2 * t1) * yt_prev + t / (2 * t1) * (lt - l + y)
            ln = lt + (t - 1) / t1 * (lt - lt_prev) + t / t1 * (lt - l + y - yt) - yn
            pts["B"].append((xt, yn, ln))
            y, l, t, yt_prev, lt_prev = yn, ln, t1, yt, lt
        else:
            y, l = yt, lt
    return out


def corrected_points(O, K):
    """Same as aadmm_text(correct=True) but also returns the corrected points (x^{k+1}, y^{k+1}, lam^{k+1})."""
    x0, y, l = O; t = 1.0; yt_prev, lt_prev = y, l; pred, corr = [O], []
    for _ in range(K):
        xt = 2 * l - 2 * y - 1; lt = l - (0.5 * xt + y - 1.5); yt = 0.5 * lt; pred.append((xt, yt, lt))
        t1 = (1 + np.sqrt(1 + 4 * t * t)) / 2
        yn = (t1 - 1) / (2 * t1) * yt - (t - 1) / (2 * t1) * yt_prev + t / (2 * t1) * (lt - l + y)
        ln = lt + (t - 1) / t1 * (lt - lt_prev) + t / t1 * (lt - l + y - yt) - yn
        corr.append((xt, yn, ln)); y, l, t, yt_prev, lt_prev = yn, ln, t1, yt, lt
    return pred, corr


def aadmm_beta(O, K, beta=1.0, f_kind="lin", correct=True, return_corr=False):
    """AADMM-beta, sn-article.tex Alg. 2 (17a)-(19c), scalar toy, B = 1, g = y^2 (mu = 2 >= B^2: (5) holds, no scaling).
    f_kind 'lin': f(x) = x (the paper's toy);  'quad': f(x) = x^2/2 (same solution (1,1,2), non-degenerate dual).
    Initialization as in the paper: (x~^0, y~^0, lam~^0) = (x^1, y^1, lam^1) = O, z = B y."""
    _, z, lam = O; t = 1.0; zt_prev, lt_prev = z, lam; out = [O]; corr = []
    for _ in range(K):
        if f_kind == "lin":
            xt = (lam * A - 1.0) / (beta * A * A) - (z - b) / A
        else:
            xt = (lam * A - beta * A * (z - b)) / (1.0 + beta * A * A)
        lt = lam - beta * (A * xt + z - b); yt = 0.5 * lt; zt = B * yt
        out.append((xt, yt, lt))
        if correct:
            t1 = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z_new = ((t1 - 1.0) / (2 * t1)) * zt - ((t - 1.0) / (2 * t1)) * zt_prev + (t / (2 * t1)) * (lt - lam + (1.0 - beta) * zt + beta * z)
            lam_new = ((t1 - 1.0) / (2 * t1)) * (2.0 * lt - beta * zt) + ((t - 1.0) / (2 * t1)) * (beta * zt_prev - 2.0 * lt_prev) \
                + (t / (2 * t1)) * ((beta - 2.0) * lam + (4.0 - beta) * lt + (beta * beta - 3.0 * beta) * zt + (2.0 * beta - beta * beta) * z)
            corr.append((xt, z_new, lam_new))
            z, lam, zt_prev, lt_prev, t = z_new, lam_new, zt, lt, t1
        else:
            z, lam = zt, lt
    return (out, corr) if return_corr else out


def admm_beta(O, K, beta=1.0, f_kind="lin"):
    _, y, lam = O; out = [O]
    for _ in range(K):
        if f_kind == "lin":
            x = (lam * A - 1.0) / (beta * A * A) - (y - b) / A
        else:
            x = (lam * A - beta * A * (y - b)) / (1.0 + beta * A * A)
        y = (lam - beta * (A * x - b)) / (2.0 + beta)          # argmin y^2 - lam y + beta/2 (A x + y - b)^2
        lam = lam - beta * (A * x + y - b); out.append((x, y, lam))
    return out


def _fig(fig_dir, name, series, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                         "font.size": 7, "pdf.fonttype": 42, "savefig.dpi": 600})
    fig = plt.figure(figsize=(0.48 * 372 / 72.27, 0.48 * 372 / 72.27 * 0.9))
    ax = fig.add_subplot(111, projection="3d")
    for lab, pts, col, mk in series:
        P = np.array(pts, dtype=float)
        ax.plot(P[:, 0], P[:, 1], P[:, 2], color=col, lw=0.9, marker=mk, ms=3, label=lab)
    ax.scatter([1], [1], [2], color="gold", edgecolor="k", s=30, zorder=5, label=r"$(x^*,y^*,\lambda^*)$")
    ax.set_xlabel("$x$", labelpad=-6); ax.set_ylabel("$y$", labelpad=-6); ax.set_zlabel(r"$\lambda$", labelpad=-6)
    ax.tick_params(labelsize=5, pad=-2); ax.set_title(title, fontsize=7, pad=0)
    ax.set_box_aspect(None, zoom=0.85)
    ax.legend(fontsize=5, loc="upper left", frameon=False)
    fig.subplots_adjust(left=0, right=1, bottom=0.02, top=0.93)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--fig", default="toy_out"); a = ap.parse_args()
    os.makedirs(a.fig, exist_ok=True)
    O = (-95.0, 86.0, 46.0); K = 5
    adm = admm_text(O, K); aad = aadmm_text(O, K)
    pred, corr = corrected_points(O, K)
    ponly = aadmm_text(O, K, correct=False)
    # consistency of the printed scheme with Alg. 2 (beta = 1, f = x)
    alg2, alg2c = aadmm_beta(O, K, 1.0, "lin", return_corr=True)
    ok_text_alg2 = bool(np.allclose(np.array(aad), np.array(alg2), atol=1e-9) and np.allclose(np.array(admm_beta(O, K, 1.0, "lin")), np.array(adm), atol=1e-9))
    orig = aadmm_text(O, 60, orig=True)
    q_pred, q_corr = aadmm_beta(O, 12, 1.0, "quad", return_corr=True)
    q_only = aadmm_beta(O, 12, 1.0, "quad", correct=False)
    q_admm = admm_beta(O, 12, 1.0, "quad")
    err = lambda s: [float(np.linalg.norm(np.array(p) - np.array((1.0, 1.0, 2.0)))) for p in s]
    res = dict(O=O, admm=adm, aadmm_pred=aad, aadmm_pred_only=ponly, corrected_points_B=corr, text_equals_alg2=ok_text_alg2,
               original_erroneous_formula_limit=orig[-1], original_erroneous_errors_k1_5=err(orig[1:6]),
               errors=dict(admm=err(adm), aadmm=err(aad), pred_only=err(ponly)),
               quad_variant=dict(f="x^2/2", solution=(1, 1, 2), admm=q_admm, aadmm_pred=q_pred, pred_only=q_only, B=q_corr,
                                 errors=dict(admm=err(q_admm), aadmm=err(q_pred), pred_only=err(q_only))))
    with open(os.path.join(a.fig, "toy_E1_sequences.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    _fig(a.fig, "fig_E1_corrected_left", [("ADMM", adm, "#0077BB", "o"), ("AADMM", aad, "#CC3311", "s")], r"(E1), corrected updates, $k\leq5$")
    _fig(a.fig, "fig_E1_corrected_right", [("prediction only", ponly, "#0077BB", "o"),
                                           ("prediction-correction", [O, pred[1], corr[0]] + pred[2:], "#CC3311", "s")], "(E1): A, B, $P_k$, $F_k$")
    _fig(a.fig, "fig_E1_quad_left", [("ADMM", q_admm[:6], "#0077BB", "o"), ("AADMM", q_pred[:6], "#CC3311", "s")], r"$f=x^2/2$, $k\leq5$")
    _fig(a.fig, "fig_E1_quad_right", [("prediction only", q_only[:6], "#0077BB", "o"),
                                      ("prediction-correction", [O, q_pred[1], q_corr[0]] + q_pred[2:6], "#CC3311", "s")], r"$f=x^2/2$: A, B, $P_k$, $F_k$")
    print("toy: text scheme == Alg. 2:", ok_text_alg2, "| corrected AADMM errors k=0..5:", ["%.3g" % e for e in err(aad)],
          "| original formula limit:", np.round(orig[-1], 4).tolist())
    if not ok_text_alg2:
        sys.exit(1)


if __name__ == "__main__":
    main()
