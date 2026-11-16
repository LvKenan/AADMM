"""C2_supp implementation checks (run inside 'smoke').  Exit code 0 iff all checks pass."""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np
from c1.problems import make_problem
from c1.metrics import Monitor
from c1.methods import run_aadmm
from c2.methods import run_aadmm_c2, run_dual_fista, run_dual_pg, run_aadmm_inexact, run_admm_inexact, InexactMonitor
from c2.problems import InnerProblem, lasso_reference
from c2.refs import c1_ref, tvd_ref, _load

FAIL = []


def check(name, ok, info=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        FAIL.append(name)


CFG = dict(tol=0.0, maxit=60, time_budget=60, log_all_until=100, log_density=50)
# 1. c2 AADMM (zero init) == c1 AADMM, bit for bit
for spec in (dict(P="P3", p=60, mu=1.0, seed=0), dict(P="P1", m=20, p=60, seed=0), dict(P="P4", N=16, seed=0)):
    for beta in (1.0, 1.5):
        t1, t2 = [], []
        p1 = make_problem(spec); run_aadmm(p1, Monitor(p1, CFG), beta=beta, trace=t1)
        p2 = make_problem(spec); run_aadmm_c2(p2, Monitor(p2, CFG), beta=beta, init="zero", trace=t2)
        same = len(t1) == len(t2) and all(np.array_equal(a["lt"], b["lt"]) and np.array_equal(a["yt"], b["yt"]) for a, b in zip(t1, t2))
        check(f"c2 AADMM == c1 AADMM ({spec['P']}, beta={beta:g})", same)
# 2. Remark rem:dualfista (i): beta=1, affine grad Psi (P3, P4), compatible init -> AADMM-1 predictor == FISTA-1 iterate
for spec in (dict(P="P3", p=60, mu=1.0, seed=0), dict(P="P4", N=16, seed=0)):
    for beta, expect_equal in ((1.0, True), (1.5, False)):
        tA, tF = [], []
        pa = make_problem(spec); run_aadmm_c2(pa, Monitor(pa, CFG), beta=beta, init="compatible", trace=tA)
        pf = make_problem(spec); run_dual_fista(pf, Monitor(pf, CFG), beta=beta, trace=tF)
        dev = max(np.linalg.norm(a["lt"] - f["eta"]) / (1 + np.linalg.norm(f["eta"])) for a, f in zip(tA, tF))
        check(f"Remark (i)/(iii) {spec['P']} beta={beta:g}: AADMM vs FISTA max rel. dev", (dev < 1e-10) if expect_equal else (dev > 1e-6), f"{dev:.2e}")
        # Lemma lem:srec (b) and (c), second line
        r1 = max(np.linalg.norm(t["s"] - t["u"] - beta * t["r"]) / (1 + np.linalg.norm(t["s"])) for t in tA)
        r2 = 0.0
        for i in range(1, len(tA)):
            th = (tA[i]["t_k"] - 1) / tA[i]["t_k1"]
            pred = (1 + th) * tA[i]["u"] - th * tA[i - 1]["u"] + tA[i]["t_k"] / tA[i]["t_k1"] * (beta - 1) * beta * tA[i]["r"]
            r2 = max(r2, np.linalg.norm(tA[i]["s_next"] - pred) / (1 + np.linalg.norm(pred)))
        check(f"Lemma srec (b),(c) identities {spec['P']} beta={beta:g}", r1 < 1e-11 and r2 < 1e-11, f"{r1:.1e} {r2:.1e}")
# 3. nonaffine grad Psi (P1), beta = 1: AADMM and FISTA differ (Remark (ii)) -- informational, must be finite
tA, tF = [], []
pa = make_problem(dict(P="P1", m=20, p=60, seed=0)); run_aadmm_c2(pa, Monitor(pa, CFG), beta=1.0, init="compatible", trace=tA)
pf = make_problem(dict(P="P1", m=20, p=60, seed=0)); run_dual_fista(pf, Monitor(pf, CFG), beta=1.0, trace=tF)
dev = max(np.linalg.norm(a["lt"] - f["eta"]) for a, f in zip(tA, tF))
check("Remark (ii) P1 beta=1: deviation finite (informational)", np.isfinite(dev), f"{dev:.2e}")
# 4. dual PG / FISTA with beta <= 1 decrease the dual gap on P3 small
for fn, name in ((run_dual_pg, "dual_pg"), (run_dual_fista, "dual_fista")):
    p = make_problem(dict(P="P3", p=60, mu=1.0, seed=0)); m = Monitor(p, dict(CFG, maxit=400)); fn(p, m, beta=1.0)
    d = m.result()[:, 5]
    check(f"{name} beta=1 dual value increases on P3 small", d[-1] > d[0] and np.isfinite(d[-1]))
# 5. shipped references: sha256, fingerprint, TVD duality identity
z, meta = _load()
check("refs/c1v12_refs.npz sha256 and 21 instances", len(meta) == 21)
r = c1_ref("P4_N64_s0", dict(P="P4", N=64, seed=0))
check("C1-1.2 reference fingerprint P4_N64_s0", r["gap_rel"] <= 1e-12)
P = InnerProblem(dict(kind="TVD", N=64, seed=0)); tr = tvd_ref(P)
rel = abs(tr["pstar"] - tr["d_at_ustar"]) / (1 + abs(tr["pstar"]))
check("TVD: d(u*) = 1/2||f0||^2 - F*_P4 (dual model DM = ROF)", rel < 1e-10, f"{rel:.1e}")
# 6. inner certificates: gap >= true suboptimality >= 0 ; both inner modes decrease h
for spec in (dict(kind="LASSO", m=40, n=120, seed=0), dict(kind="TVD", N=16, seed=0)):
    P = InnerProblem(spec); rng = np.random.default_rng(5)
    q = P.b + 0.1 * rng.standard_normal(P.m); beta = 1.3
    xs, j, g = P.inner_solve(np.zeros(P.nx), q, beta, "tol", tol=1e-12 * (1 + q @ q), maxinner=200000)
    hstar = P.inner_gap(xs, q, beta)[1]
    ok = True
    for cnt in (1, 5, 50):
        x, _, gap = P.inner_solve(np.zeros(P.nx), q, beta, "count", count=cnt)
        h = P.inner_gap(x, q, beta)[1]
        ok &= (gap >= h - hstar - 1e-9 * (1 + abs(hstar))) and (h - hstar >= -1e-9 * (1 + abs(hstar)))
    check(f"inner certificate >= suboptimality, count modes ({spec['kind']})", ok and g <= 1e-12 * (1 + q @ q), f"exact gap {g:.1e} in {j} its")
# 7. LASSO reference certificate and B=I degeneracy (Remark rem:dualfista (i)): exact AADMM-1 is optimal after 1 outer step
P = InnerProblem(dict(kind="LASSO", m=40, n=120, seed=0)); ref = lasso_reference(P, tol=1e-13, time_budget=60)
check("LASSO reference within tol", ref["status"] == "tol" and ref["UB"] >= ref["LB"], f"{ref['gap_rel']:.1e}")
rc = dict(tol_dgap=1e-11, maxit=50, time_budget=60, log_all_until=100, log_density=50, eps_floor_rel=1e-12, maxinner={"LASSO": 200000, "TVD": 5000})
m = InexactMonitor(P, ref, rc, 1.0); run_aadmm_inexact(P, m, beta=1.0, sched=dict(mode="tol", c=1e-13, p=0))
check("LASSO beta=1 exact inner: AADMM-1 hits dgap<=1e-11 within 2 outer its (B=I degeneracy)", m.stop_reason == "tol" and m.k <= 2, f"k={m.k}")
m = InexactMonitor(P, ref, rc, 1.5); run_aadmm_inexact(P, m, beta=1.5, sched=dict(mode="tol", c=1e-13, p=0))
check("LASSO beta=1.5 exact inner: AADMM converges, theorem ratio <= 1", m.stop_reason == "tol" and np.nanmax(m.result()[:, 5]) <= 1.0, f"k={m.k}")
m = InexactMonitor(P, ref, dict(rc, maxit=400), 1.0); run_admm_inexact(P, m, beta=1.0, sched=dict(mode="tol", c=1e-13, p=0))
check("LASSO ADMM exact inner converges", m.result()[-1, 3] < 1e-8, f"dgap={m.result()[-1, 3]:.1e}")
# 8. toy example (L2): corrected AADMM-1 updates of Sec. 4.1 converge to (1,1,2)
sys.path.insert(0, HERE)
import toy_fig
seq = toy_fig.aadmm_text((-95.0, 86.0, 46.0), 10)
check("toy (E1) corrected AADMM-1 reaches (1,1,2)", np.allclose(seq[-1], (1, 1, 2), atol=1e-12))
seq = toy_fig.aadmm_beta((-95.0, 86.0, 46.0), 400, beta=0.5, f_kind="quad")
check("toy variant f=x^2/2 (same optimum), AADMM-0.5 converges", np.allclose(seq[-1], (1, 1, 2), atol=1e-6))
print("SANITY", "PASS" if not FAIL else f"FAIL: {FAIL}")
sys.exit(0 if not FAIL else 1)
