"""R6 analysis of C1-1.2 results: theorem bound, slopes, comparison table, beta, S_k, restarts."""
import json, collections, numpy as np
from common import *
R = refs()
runs = []
for p in files("main") + files("sweep"):
    r = load(p); m = r["meta"]; t = m["task"]; ref = R[t["inst"]]
    F = ref["F"]; sc = 1 + abs(F)
    k = H(r, "k").astype(int); dv = H(r, "dval")
    rec = dict(id=t["id"], group=t["group"], P=t["P"], cfg=cfgname(t["inst"]), inst=t["inst"], method=t["method"], v=vkey(t),
               params=t["params"], stop=m["stop_reason"], iters=m["iters"], t_alg=m["t_alg"], t_wall=m["t_wall"],
               kkt_hit={lv: m["first_hit"].get(f"{lv:g}", [None, None]) for lv in (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)},
               final_kkt=float(r["kkt_all"][-1]), rs=m["run_summary"])
    g = (F - dv) / sc
    rec["dg_hit"] = {lv: first_reach(k, g, lv) for lv in (1e-4, 1e-6, 1e-8, 1e-10)}
    rec["dg_final"] = float(g[-1]); rec["dg_min"] = float(np.min(g))
    if t["method"] == "aadmm":
        beta = float(t["params"].get("beta", 1.0))
        C = float(ref["lam"] @ ref["lam"]) / beta + float(np.sum((ref["lam"] - 2 * ref["By"]) ** 2)) / (2 - beta)
        L = r["lpred_all"]; kk = np.arange(1, len(L) + 1); vk = F - L
        unc = ref["gap"] * sc + 1e-13 * sc
        ratio = (kk + 1) ** 2 * vk / C
        rec.update(beta=beta, C=C, max_ratio=float(np.max(ratio)), k_at_max=int(kk[np.argmax(ratio)]),
                   n_viol=int(np.sum(vk - unc > C / (kk + 1) ** 2)), n_weak_dual_viol=int(np.sum(L > ref["UB"] + unc)),
                   ratio_last=float(ratio[-1]))
        vr = vk / sc; sel = (kk >= 100) & (vr > 1e-10)
        rec["slope"] = float(np.polyfit(np.log(kk[sel]), np.log(vr[sel]), 1)[0]) if sel.sum() > 20 else None
        # slope on the window [100, 1000]
        sel2 = (kk >= 100) & (kk <= 1000) & (vr > 1e-10)
        rec["slope_100_1000"] = float(np.polyfit(np.log(kk[sel2]), np.log(vr[sel2]), 1)[0]) if sel2.sum() > 20 else None
    if t["method"] in ("iadmm", "heu_admm"):
        pen = r["pen_all"]; S = np.cumsum(pen); kk = np.arange(1, len(S) + 1)
        rec["Sk_k2"] = {kx: float(S[kx - 1] / kx ** 2) for kx in (100, 1000, 10000) if kx <= len(S)}
        rec["Sk_k2_final"] = float(S[-1] / len(S) ** 2)
        sel = kk >= 100
        rec["Sk_exponent"] = float(np.polyfit(np.log(kk[sel]), np.log(S[sel]), 1)[0]) if sel.sum() > 20 else None
        rec["pen_max_over_k"] = float(np.max(pen / kk)); rec["pen_last"] = float(pen[-1]); rec["pen_max"] = float(pen.max())
        rec["S_check"] = abs(S[-1] - m["run_summary"]["S_k"]) / S[-1]
    if t["method"] == "fast_admm_r":
        rec["n_restart"] = int(len(r["restart_iters"])); rec["restart_frac"] = rec["n_restart"] / m["iters"]
    runs.append(rec)
json.dump(runs, open("runs_r6.json", "w"), default=float)
print("runs", len(runs))
