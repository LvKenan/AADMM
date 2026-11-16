"""Sanity checks of the C1 implementation (run: python tests/sanity.py).  Exit code 0 = all pass.

 S1  operator checks: GradOp adjoint, DCT-based exact ADMM y-step for P4, P3 Cholesky y-step.
 S2  AADMM (Alg. 2) converges on small NON-degenerate instances of P1-P4; Theorem 3 bound
     (k+1)^2 v_k <= C (eq. (27), lam^1 = z^1 = 0) holds along the run, beta in {0.5, 1, 1.5};
     L(x~,y~,lam~) = d(lam~) (memo Prop. A).
 S3  Degenerate case of memo Sec. 4.1 (g = 1/2||y||^2, B = I, mu = 1, no rescaling, beta = 1):
     lam^k - z^k = 0 and A x~^k + B y~^k - b = 0 for k >= 2 (exact feasibility); control runs with
     beta = 0.5 and with B^T B != mu I are NOT exactly feasible; ADMM residual decays only geometrically.
 S4  Alg. 1 (AADMM-1, eqs. (13)-(15)) == Alg. 2 with beta = 1 (eqs. (17)-(19)), iterate by iterate.
 S5  Formal B=0 limit: asserts agreement with the revised s=lambda-z recurrence and disagreement
     with He-Yuan (2010), (3.4b), which was incorrectly identified as the limit in the old manuscript.
 S6  Spectral normalization, current-instance nondegeneracy, P3 reference gap and IADMM floor.
 S7  C1-1.2 run list (main variants, theory scopes, P4 reference budget) and the per-iteration
     records added in 1.2 (S_k, penalty trace, restart flags/count, KKT trace, AADMM dual values).
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from c1.problems import make_problem, Problem, DenseOp, Counter, GradOp, soft, P4_TV, norm2_bound
from c1.reference import reference_solve
from c1.metrics import Monitor, FIELDS
from c1.methods import run_aadmm, run_admm, run_iadmm

FAIL = []


def check(name, ok, msg=""):
    print(("PASS " if ok else "FAIL ") + name + ("  " + msg if msg else ""))
    if not ok:
        FAIL.append(name)


def s1():
    rng = np.random.default_rng(1)
    cnt = Counter()
    G = GradOp(7, cnt, c=-0.7)
    u = rng.standard_normal(49); p = rng.standard_normal(98)
    check("S1 GradOp adjoint", abs(G.mv(u) @ p - u @ G.rmv(p)) < 1e-12)
    pr = P4_TV(8, seed=3)
    beta = 0.7
    lam = rng.standard_normal(pr.m); c = rng.standard_normal(pr.m)
    y = pr.make_admm_ysolver(beta)(lam, c)
    # optimality of argmin 1/2||y-f0||^2 - <lam, By> + beta/2||By + c||^2
    grad = (y - pr.f0) - pr.B.rmv(lam) + beta * pr.B.rmv(pr.B.mv(y) + c)
    check("S1 P4 exact y-step (DCT)", np.linalg.norm(grad) < 1e-10, "||grad||=%.1e" % np.linalg.norm(grad))
    pr3 = make_problem(dict(P="P3", p=60, mu=0.5, seed=2))
    lam = rng.standard_normal(pr3.m); c = rng.standard_normal(pr3.m)
    y = pr3.make_admm_ysolver(beta)(lam, c)
    grad = pr3.H @ y + pr3.h - pr3.B.rmv(lam) + beta * pr3.B.rmv(pr3.B.mv(y) + c)
    check("S1 P3 exact y-step (Cholesky)", np.linalg.norm(grad) < 1e-9, "||grad||=%.1e" % np.linalg.norm(grad))
    ev = np.linalg.eigvalsh(pr3.mu * np.eye(pr3.ny) - pr3.B.M.T @ pr3.B.M)
    check("S1 scaling eq.(5): mu I - B^T B >= 0", ev.min() >= -1e-12, "min eig=%.2e" % ev.min())


def s2():
    i = FIELDS.index
    specs = [dict(P="P1", m=60, p=200, seed=0), dict(P="P2", n=100, p=300, s=30, seed=0),
             dict(P="P3", p=120, mu=0.5, seed=0), dict(P="P4", N=24, seed=0)]
    for sp in specs:
        pr = make_problem(sp)
        ref = reference_solve(pr, tol=1e-13, time_budget=60)
        check("S2 %s reference reaches gap tolerance" % pr.inst, ref["status"] == "tol")
        Fs = ref["Fstar"]
        for beta in (0.5, 1.0, 1.5):
            pr.cnt.reset()
            mon = Monitor(pr, dict(tol=1e-9, maxit=6000, time_budget=60, log_all_until=10 ** 9))
            run_aadmm(pr, mon, beta=beta)
            R = mon.result()
            k = R[:, i("k")]
            v = Fs - R[:, i("dval")]                       # v_k = p* - d(lam~^k)
            lam_s, By_s = ref["lam_star"], ref["By_star"]
            C = (1 / beta) * np.sum(lam_s ** 2) + (1 / (2 - beta)) * np.sum((lam_s - 2 * By_s) ** 2)
            ratio = np.max((k + 1) ** 2 * np.maximum(v, 0) / C)
            idgap = np.max(np.abs(R[:, i("Lpred")] - R[:, i("dval")])) / (1 + abs(Fs))
            kkt = R[-1, i("KKT")]
            ok = (kkt < 1e-5) and ratio <= 1.0 + 1e-6 and idgap < 1e-10
            check("S2 %s beta=%.1f" % (pr.inst, beta), ok,
                  "iters=%d KKT=%.1e  max (k+1)^2 v_k/C=%.3f  |L(pred)-d(lam~)|=%.1e  ref gap=%.1e"
                  % (k[-1], kkt, ratio, idgap, ref["gap_rel"]))


class Degenerate(Problem):
    """f = tau||x||_1, A = I, g = 1/2||y||^2 (+ optional extra), B = Bmat, b; mu = 1; NO rescaling."""
    has_feas = True
    has_exact_y = True

    def __init__(self, n, seed, Bscale=1.0, tau=0.1):
        rng = np.random.default_rng(seed)
        self.cnt = Counter()
        self.B = DenseOp(Bscale * np.eye(n), self.cnt)
        self.a = 1.0; self.b = rng.standard_normal(n); self.mu = 1.0; self.tau = tau
        self.nx = self.ny = self.m = n
        self.s = 1.0; self.normB2 = Bscale ** 2; self.normB2_orig = Bscale ** 2
        self.inst = "degenerate"; self.meta = {}; self.name = "DEG"

    def f(self, x): return self.tau * np.abs(x).sum()
    def prox_f(self, v, t): return soft(v, t * self.tau)
    def fconj(self, u): return 0.0
    def proj_dual(self, lam): return np.clip(lam, -self.tau, self.tau)
    def x_feas(self, y, By): return self.b - By
    def g(self, y): return 0.5 * float(y @ y)
    def make_prox_g(self): return lambda v, t: v / (1 + t)
    def make_prox_g_fixed(self, t, monitor=False): return lambda v: v / (1 + t)
    def make_gstar_grad(self, monitor=False): return lambda w: w.copy()
    def gconj_val(self, w, ys): return 0.5 * float(w @ w)

    def make_admm_ysolver(self, beta):
        M = self.B.M
        K = np.eye(self.ny) + beta * M.T @ M
        return lambda lam, c: np.linalg.solve(K, M.T @ (lam - beta * c))


def s3():
    i = FIELDS.index
    pr = Degenerate(50, seed=0)
    tr = []
    mon = Monitor(pr, dict(tol=-1.0, maxit=30, time_budget=30))
    run_aadmm(pr, mon, beta=1.0, trace=tr)
    R = mon.result()
    lz = [np.linalg.norm(d["lam"] - d["z"]) for d in tr]          # lam^{k+1} - z^{k+1}
    rk = R[:, i("rp_abs")]
    ok = max(lz[:-1]) < 1e-13 and np.max(rk[1:]) < 1e-13
    check("S3 degenerate (g=1/2||y||^2, B=I, beta=1): exact feasibility for k>=2", ok,
          "max_k ||lam^{k+1}-z^{k+1}||=%.1e, max_{k>=2} ||r^k||=%.1e, ||r^1||=%.2e" % (max(lz), np.max(rk[1:]), rk[0]))
    mon2 = Monitor(pr, dict(tol=-1.0, maxit=30, time_budget=30))
    run_admm(pr, mon2, beta=1.0)
    r2 = mon2.result()[:, i("rp_abs")]
    print("     ADMM (beta=1) residuals k=1..8: " + " ".join("%.1e" % v for v in r2[:8]))
    check("S3 control: ADMM residual is not exactly zero (decays geometrically)", r2[5] > 1e-6)
    for lbl, prc, bt in (("beta=0.5", Degenerate(50, 0), 0.5), ("B=0.8 I", Degenerate(50, 0, Bscale=0.8), 1.0)):
        m3 = Monitor(prc, dict(tol=-1.0, maxit=30, time_budget=30))
        run_aadmm(prc, m3, beta=bt)
        r3 = m3.result()[:, i("rp_abs")]
        check("S3 control %s: NOT exactly feasible at k=2..5" % lbl, np.min(r3[1:5]) > 1e-10,
              "||r^k|| k=1..6: " + " ".join("%.1e" % v for v in r3[:6]))


def s4():
    pr = make_problem(dict(P="P1", m=40, p=120, seed=5))
    tr = []
    mon = Monitor(pr, dict(tol=-1.0, maxit=60, time_budget=30))
    run_aadmm(pr, mon, beta=1.0, trace=tr)
    # Algorithm 1 (AADMM-1), eqs. (13)-(15), independent transcription
    a, b, B = pr.a, pr.b, pr.B
    gs = pr.make_gstar_grad()
    z = np.zeros(pr.m); lam = np.zeros(pr.m); yt_prev = np.zeros(pr.ny); lt_prev = np.zeros(pr.m); t = 1.0
    err = 0.0
    for k in range(60):
        xt = pr.prox_f((lam - (z - b)) / a, 1.0 / (a * a))            # (13) line 1, beta=1
        lt = lam - (a * xt + z - b)                                   # (13) line 2
        yt = gs(B.rmv(lt))                                            # (13) line 3
        t1 = 0.5 * (1 + np.sqrt(1 + 4 * t * t))                       # (14)
        zn = (t1 - 1) / (2 * t1) * B.mv(yt) - (t - 1) / (2 * t1) * B.mv(yt_prev) + t / (2 * t1) * (lt - lam + z)   # (15) line 2
        ln = lt + (t - 1) / t1 * (lt - lt_prev) + t / t1 * (lt - lam + z - B.mv(yt)) - zn                         # (15) line 3
        d = tr[k]
        err = max(err, np.linalg.norm(xt - d["xt"]), np.linalg.norm(lt - d["lt"]), np.linalg.norm(zn - d["z"]),
                  np.linalg.norm(ln - d["lam"]))
        z, lam, yt_prev, lt_prev, t = zn, ln, yt, lt, t1
    check("S4 Alg. 1 == Alg. 2 at beta=1 (60 iterations)", err < 1e-10, "max diff=%.1e" % err)


def s5():
    """B = 0, one block: min 1/2||x-c||^2 s.t. A x = b (A general).  x-step is a linear solve."""
    rng = np.random.default_rng(0)
    m, n = 15, 30
    A = rng.standard_normal((m, n)); c = rng.standard_normal(n); b = rng.standard_normal(m)
    K = np.eye(n) + A.T @ A
    xs = lambda lam, z: np.linalg.solve(K, c + A.T @ (lam - z + b))    # argmin 1/2||x-c||^2 - <lam,Ax+z-b> + 1/2||Ax+z-b||^2
    xstar = c - A.T @ np.linalg.solve(A @ A.T, A @ c - b)

    def aadmm1(N):
        z = np.zeros(m); lam = np.zeros(m); lt_prev = np.zeros(m); t = 1.0; X = []
        for _ in range(N):
            x = xs(lam, z); lt = lam - (A @ x + z - b)
            t1 = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            zn = t / (2 * t1) * (lt - lam + z)                            # (15) with B y~ = 0
            ln = lt + (t - 1) / t1 * (lt - lt_prev) + t / t1 * (lt - lam + z) - zn
            z, lam, lt_prev, t = zn, ln, lt, t1; X.append(x)
        return np.array(X)

    def he_yuan_aalm(N):
        lam = np.zeros(m); lt_prev = np.zeros(m); t = 1.0; X = []
        for _ in range(N):
            x = xs(lam, np.zeros(m)); lt = lam - (A @ x - b)
            t1 = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            ln = lt + (t - 1) / t1 * (lt - lt_prev) + t / t1 * (lt - lam)
            lam, lt_prev, t = ln, lt, t1; X.append(x)
        return np.array(X)

    def nesterov_alm(N):
        lam = np.zeros(m); lt_prev = np.zeros(m); t = 1.0; X = []
        for _ in range(N):
            x = xs(lam, np.zeros(m)); lt = lam - (A @ x - b)
            t1 = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
            ln = lt + (t - 1) / t1 * (lt - lt_prev)
            lam, lt_prev, t = ln, lt, t1; X.append(x)
        return np.array(X)
    N = 40
    X1, X16, XN = aadmm1(N), he_yuan_aalm(N), nesterov_alm(N)
    d16 = np.max(np.linalg.norm(X1 - X16, axis=1)); dN = np.max(np.linalg.norm(X1 - XN, axis=1))
    check("S5 revised B=0 recurrence agrees with AADMM-1", dN < 1e-10, "max diff=%.2e" % dN)
    check("S5 He-Yuan (3.4b) is a different recurrence", d16 > 1e-3, "max diff=%.2e" % d16)
    print("     errors ||x^k - x*|| at k=5,10,20,40:  AADMM-1 %s | He-Yuan %s | revised limit %s" % (
        " ".join("%.1e" % np.linalg.norm(X1[k - 1] - xstar) for k in (5, 10, 20, 40)),
        " ".join("%.1e" % np.linalg.norm(X16[k - 1] - xstar) for k in (5, 10, 20, 40)),
        " ".join("%.1e" % np.linalg.norm(XN[k - 1] - xstar) for k in (5, 10, 20, 40))))
    return d16, dN


def s6():
    op = DenseOp(np.diag([3., 1.]), Counter())
    check("S6 spectral bound targets the largest eigenvalue", 9 <= norm2_bound(op) < 9.001)
    specs = [dict(P="P1", m=20, p=40, seed=0), dict(P="P2", n=20, p=40, s=4, seed=0),
             dict(P="P3", p=20, mu=.5, seed=0), dict(P="P4", N=8, seed=0)]
    for spec in specs:
        pr = make_problem(spec)
        true_bound = norm2_bound(pr.B)
        check("S6 %s has strict normalized spectral slack" % pr.name, true_bound < pr.mu)
        mon = Monitor(pr, dict(tol=-1., maxit=5, time_budget=30))
        run_aadmm(pr, mon, beta=1.)
        check("S6 %s is not structurally feasible at step 2" % pr.name,
              mon.result()[1, FIELDS.index("rp_abs")] > 1e-10)
    pr = make_problem(dict(P="P3", p=2, mu=1., seed=2))
    pr.H = np.diag([1., 4.]); pr.h = np.array([-2., -8.])
    pr.B = DenseOp(.5 * np.eye(2), pr.cnt); pr.a = -.5
    pr.normB2 = .25; pr.feasible_anchor = np.zeros(2)
    pr.lo = -np.ones(2); pr.hi = np.ones(2)
    optimum = np.ones(2)
    exact_value = pr.g(optimum)
    trial = reference_solve(pr, maxit=1)
    check("S6 P3 repairs an infeasible recovered point into a finite upper bound",
          np.max(trial["y_star"]) > 1. and np.isfinite(trial["UB"])
          and trial["UB"] >= exact_value - 1e-12)
    ref = reference_solve(pr, tol=1e-10, time_budget=10)
    check("S6 P3 reference brackets a known feasible optimum",
          ref["LB"] <= exact_value + 1e-10 and ref["UB"] >= exact_value - 1e-10
          and ref["status"] == "tol" and np.isfinite(ref["UB"]))
    pr = make_problem(dict(P="P1", m=100, p=500, seed=0))
    mon = Monitor(pr, dict(tol=-1., maxit=3, time_budget=30))
    run_iadmm(pr, mon, beta1_mult=.001)
    H = mon.result()
    check("S6 IADMM initial penalty respects its floor",
          mon.method_meta["beta1_requested"] < mon.method_meta["beta_floor"]
          and H[0, FIELDS.index("penalty")] >= mon.method_meta["beta_floor"])
    check("S6 IADMM records cumulative penalty weights",
          np.allclose(H[:, FIELDS.index("weight_sum")], np.cumsum(H[:, FIELDS.index("penalty")])) )


def s7():
    import c1
    from c1.tasks import build_tasks, load_config, ref_time_budget, smoke_tasks
    from c1.methods import run_heu_admm, run_fast_admm_restart
    cfg = load_config()
    check("S7 package/config version C1-1.2", c1.__version__ == "C1-1.2" and cfg["version"] == "C1-1.2")
    T = build_tasks(cfg)
    ids = [t["id"] for t in T]
    n = {g: sum(t["group"] == g for t in T) for g in ("ref", "main", "sweep")}
    check("S7 task counts ref/main/sweep = 70/670/198 (938)", (n["ref"], n["main"], n["sweep"]) == (70, 670, 198)
          and len(ids) == 938, str(n))
    check("S7 task ids unique", len(set(ids)) == len(ids))
    import json as _json
    rl = _json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RUN_LIST.json")))
    check("S7 RUN_LIST.json matches the generated task list", [t["id"] for t in rl["tasks"]] == ids and rl["n_tasks"] == 938)
    need = ["main__P1_m100_p500_s0__admm__beta10", "main__P1_m100_p500_s0__admm__beta100",
            "main__P4_N192_s4__aadmm__beta1.5", "main__P3_p2000_mu1_s4__admm__beta100"]
    check("S7 new main variants present", all(x in ids for x in need))
    runs = [t for t in T if t["kind"] == "run"]
    check("S7 every run task has a label and a theory scope", all(t.get("label") and t.get("theory_scope") for t in runs))
    byid = {t["id"]: t for t in T}
    sc_admm = byid["main__P1_m100_p500_s0__admm__beta100"]["theory_scope"]
    sc_a15 = byid["main__P4_N192_s4__aadmm__beta1.5"]["theory_scope"]
    check("S7 tuned ADMM scope: no O(1/k^2) guarantee", "no O(1/k^2)" in sc_admm)
    check("S7 AADMM beta=1.5 scope: covered by the paper theorem", "O(1/k^2)" in sc_a15 and "(0,2)" in sc_a15)
    check("S7 labels distinguish variants", byid["main__P1_m100_p500_s0__admm__beta10"]["label"]
          != byid["main__P1_m100_p500_s0__admm__default"]["label"])
    check("S7 reference budget P4 = 2400 s, others 300 s",
          ref_time_budget(cfg, "P4") == 2400.0 and ref_time_budget(cfg, "P1") == 300.0)
    check("S7 smoke covers the new variants", any(t.get("params") == {"beta": 100.0} and t.get("method") == "admm"
                                                    for t in smoke_tasks()))
    # per-iteration records
    pr = make_problem(dict(P="P2", n=60, p=150, s=15, seed=0))
    mon = Monitor(pr, dict(tol=-1., maxit=150, time_budget=30, log_all_until=20, log_density=10))
    run_iadmm(pr, mon)
    sm = mon.summary()
    tr = mon.traces()
    check("S7 IADMM: penalty trace has one entry per iteration and the penalty adapts",
          len(tr["pen_all"]) == mon.k == 150 and sm["n_penalty_changes"] > 10)
    check("S7 IADMM: S_k equals the sum of the actually used penalties (= ergodic weight sum)",
          np.isclose(sm["S_k"], float(np.sum(tr["pen_all"])), rtol=1e-12)
          and np.isclose(sm["S_k"], sm["ergodic_weight_sum"], rtol=1e-12)
          and np.isclose(sm["S_k_over_k2"], sm["S_k"] / 150 ** 2))
    H = mon.result()
    check("S7 logged penalty_sum column is the exact running sum",
          np.allclose(H[:, FIELDS.index("penalty_sum")], np.cumsum(tr["pen_all"].astype(np.float64))[H[:, 0].astype(int) - 1], rtol=1e-6))
    kk = tr["kkt_all"]
    ok = True
    for lv, hit in mon.first_hit.items():
        idx = np.where(kk <= lv)[0]
        ok &= (hit is None and len(idx) == 0) or (hit is not None and len(idx) > 0 and hit[0] == idx[0] + 1)
    check("S7 first_hit agrees with the per-iteration KKT trace", ok and len(kk) == 150
          and mon.first_hit[1e-4] is not None)
    pr = make_problem(dict(P="P3", p=40, mu=.5, seed=0))
    mon = Monitor(pr, dict(tol=1e-10, maxit=400, time_budget=30))
    run_fast_admm_restart(pr, mon)
    sm = mon.summary(); tr = mon.traces()
    H = mon.result()
    full = H[:, 0] <= 100
    check("S7 Goldstein: restart count == number of flagged iterations",
          sm["n_restart"] == len(tr["restart_iters"]) > 0 and
          int(np.sum(H[full, FIELDS.index("restart")])) == int(np.sum(tr["restart_iters"] <= min(100, mon.k))))
    pr = make_problem(dict(P="P2", n=20, p=40, s=4, seed=0))
    mon = Monitor(pr, dict(tol=-1., maxit=60, time_budget=30))
    run_aadmm(pr, mon, beta=1.5)
    tr = mon.traces(); sm = mon.summary()
    check("S7 AADMM: per-iteration L(pred) trace and penalty recorded",
          len(tr["lpred_all"]) == 60 and np.all(np.isfinite(tr["lpred_all"])) and sm["penalty_min"] == sm["penalty_max"] == 1.5)


if __name__ == "__main__":
    s1(); s2(); s3(); s4(); s5(); s6(); s7()
    print("\nSANITY: %s" % ("ALL PASS" if not FAIL else "FAILED: " + ", ".join(FAIL)))
    sys.exit(1 if FAIL else 0)
