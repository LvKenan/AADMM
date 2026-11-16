"""Per-iteration monitoring.  Monitoring work is NOT included in the reported algorithm
time 't_alg' (only the method's own work, incl. its own residual computations for the
adaptive-penalty rules of Tang-Toh (2024), is timed).  't_wall' includes monitoring.

Residual definitions (all scale invariant, see README):
  R_p   = ||A x + B y - b|| / max(||A x||, ||B y||, ||b||)                   (Tang-Toh (2024) Sec. 5.2)
  R_d   = max( ||x - prox_f(x + A^T lam)|| / max(||x||, ||A^T lam||),
               ||y - prox_g(y + B^T lam)|| / max(||y||, ||B^T lam||) )      (Tang-Toh (2024) Sec. 5.2,
                                         with the sign of lam flipped to the paper's convention)
  KKT   = max(R_p, R_d)          (stopping quantity, identical for all methods)
Also stored: rp_abs = ||A0 x + B0 y - b0|| in the ORIGINAL (unscaled) constraint, F = f(x)+g(y),
Ffeas = F(x_feas(y), y) (P1, P2, P4), dval = d(Pi(lam)) (Lagrange dual value, a valid lower
bound of F*), and the same quantities for the ergodic (weighted-average) iterate.
"""
import time
import numpy as np

TINY = 1e-300
LEVELS = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)

FIELDS = ["k", "t_alg", "t_wall", "F", "Ffeas", "dval", "rp_abs", "Rp", "Rd_x", "Rd_y", "KKT",
          "Fbar", "Ffeas_bar", "rp_abs_bar", "Rp_bar", "penalty", "matvec", "fact",
          "Lpred", "pp_rp_abs", "pp_KKT", "weight_sum", "restart", "penalty_sum", "n_restart"]


class StopRun(Exception):
    pass


def kkt_parts(prob, x, y, lam, By, prox_g1, ATlam=None, BTlam=None):
    """Returns (Rp, Rd_x, Rd_y, ||r||_scaled).  prox_g1 = prox_g(., 1)."""
    a = prob.a
    Ax = a * x
    r = Ax + By - prob.b
    nr = float(np.linalg.norm(r))
    Rp = nr / max(float(np.linalg.norm(Ax)), float(np.linalg.norm(By)), float(np.linalg.norm(prob.b)), TINY)
    if ATlam is None:
        ATlam = a * lam
    if BTlam is None:
        BTlam = prob.B.rmv(lam)
    px = prob.prox_f(x + ATlam, 1.0)
    Rdx = float(np.linalg.norm(x - px)) / max(float(np.linalg.norm(x)), float(np.linalg.norm(ATlam)), TINY)
    py = prox_g1(y + BTlam)
    Rdy = float(np.linalg.norm(y - py)) / max(float(np.linalg.norm(y)), float(np.linalg.norm(BTlam)), TINY)
    return Rp, Rdx, Rdy, nr


class Monitor:
    def __init__(self, prob, cfg):
        self.prob = prob
        self.cfg = cfg
        self.tol = cfg["tol"]
        self.maxit = cfg["maxit"]
        self.time_budget = cfg["time_budget"]
        # log every iteration up to log_all_until, then with stride max(1, k // log_density)
        # (about log_density points per factor e of iterations); the stopping test runs every iteration
        self.full_every_until = cfg.get("log_all_until", 100)
        self.log_density = cfg.get("log_density", 50)
        # monitoring helpers (own factorizations, not timed, counters restored afterwards)
        mv0, f0 = prob.cnt.matvec, prob.cnt.fact
        self.gstar = prob.make_gstar_grad(monitor=True)
        self.prox_g1 = prob.make_prox_g_fixed(1.0, monitor=True)
        prob.cnt.matvec, prob.cnt.fact = mv0, f0
        self.rows = []
        self.t_alg = 0.0
        self.t0 = time.perf_counter()
        self.W = 0.0
        self.xbar = self.ybar = self.Bybar = None
        self.first_hit = {lv: None for lv in LEVELS}
        # C1-1.2: exact per-iteration records (one entry per iteration, not subsampled)
        self.kkt_all = []          # stopping quantity KKT_k
        self.pen_all = []          # penalty actually used in iteration k
        self.lpred_all = []        # AADMM only: L(x~^k, y~^k, lam~^k) = d(lam~^k)
        self.restart_iters = []    # Goldstein restart: iterations whose trial was rejected
        self.pen_sum = 0.0         # S_k = sum of the penalties actually used
        self.n_pen_changes = 0
        self.stop_reason = None
        self.k = 0
        self.last_kkt = np.inf

    # -- timing helpers used by methods
    def tic(self):
        self._tic = time.perf_counter()

    def toc(self):
        self.t_alg += time.perf_counter() - self._tic

    def _no_count(self):
        return self.prob.cnt.matvec, self.prob.cnt.fact

    def record(self, x, y, lam, By, weight=1.0, penalty=np.nan, aadmm_extra=None, restart=False):
        """Called once per iteration after the step (outside the timed region)."""
        prob = self.prob
        self.k += 1
        k = self.k
        mv0, f0 = self._no_count()
        # ergodic accumulation (weights given by the method)
        if weight is not None and weight > 0:
            self.W += weight
            c = weight / self.W
            if self.xbar is None:
                self.xbar, self.ybar, self.Bybar = x.copy(), y.copy(), By.copy()
            else:
                self.xbar += c * (x - self.xbar)
                self.ybar += c * (y - self.ybar)
                self.Bybar += c * (By - self.Bybar)
        pen = float(penalty)
        if self.pen_all and pen != self.pen_all[-1]:
            self.n_pen_changes += 1
        self.pen_all.append(pen)
        self.pen_sum += pen
        if restart:
            self.restart_iters.append(k)
        if aadmm_extra is not None and "Lpred" in aadmm_extra:
            self.lpred_all.append(float(aadmm_extra["Lpred"]))
        BTlam = prob.B.rmv(lam)
        Rp, Rdx, Rdy, nr = kkt_parts(prob, x, y, lam, By, self.prox_g1, BTlam=BTlam)
        kkt = max(Rp, Rdx, Rdy)
        self.last_kkt = kkt
        self.kkt_all.append(kkt)
        for lv in LEVELS:
            if self.first_hit[lv] is None and kkt <= lv:
                self.first_hit[lv] = (k, self.t_alg)
        stop = None
        if not np.isfinite(kkt) or kkt > 1e12:
            stop = "diverged"
        elif kkt <= self.tol:
            stop = "tol"
        elif k >= self.maxit:
            stop = "maxit"
        elif time.perf_counter() - self.t0 >= self.time_budget:
            stop = "time_budget"
        log_now = (k <= self.full_every_until) or (k % max(1, k // self.log_density) == 0) or (stop is not None)
        if log_now:
            s = prob.s
            F = prob.F(x, y)
            xf = prob.x_feas(y, By)
            Ffeas = prob.F(xf, y) if xf is not None else np.nan
            dval, _ = prob.dual_value(lam, self.gstar)
            Fbar = prob.F(self.xbar, self.ybar)
            xfb = prob.x_feas(self.ybar, self.Bybar)
            Ffeas_bar = prob.F(xfb, self.ybar) if xfb is not None else np.nan
            Axb = prob.a * self.xbar
            rb = Axb + self.Bybar - prob.b
            nrb = float(np.linalg.norm(rb))
            Rp_bar = nrb / max(float(np.linalg.norm(Axb)), float(np.linalg.norm(self.Bybar)),
                               float(np.linalg.norm(prob.b)), TINY)
            Lpred = pp_rp = pp_kkt = np.nan
            if aadmm_extra is not None:
                Lpred = aadmm_extra.get("Lpred", np.nan)
                pp = aadmm_extra.get("pp")
                if pp is not None:
                    xh, lh = pp
                    Rp2, Rdx2, Rdy2, nr2 = kkt_parts(prob, xh, y, lh, By, self.prox_g1)
                    pp_rp, pp_kkt = nr2 / abs(s), max(Rp2, Rdx2, Rdy2)
            self.rows.append([k, self.t_alg, time.perf_counter() - self.t0, F, Ffeas, dval, nr / abs(s),
                              Rp, Rdx, Rdy, kkt, Fbar, Ffeas_bar, nrb / abs(s), Rp_bar, penalty,
                              mv0, f0, Lpred, pp_rp, pp_kkt, self.W, float(restart),
                              self.pen_sum, float(len(self.restart_iters))])
        prob.cnt.matvec, prob.cnt.fact = mv0, f0
        if stop is not None:
            self.stop_reason = stop
            raise StopRun(stop)

    def traces(self):
        """Exact per-iteration arrays saved with every run (C1-1.2)."""
        return dict(kkt_all=np.asarray(self.kkt_all, dtype=np.float64),
                    pen_all=np.asarray(self.pen_all, dtype=np.float64),
                    lpred_all=np.asarray(self.lpred_all, dtype=np.float64),
                    restart_iters=np.asarray(self.restart_iters, dtype=np.int64))

    def summary(self):
        """Scalar run summary (stored in meta['run_summary'])."""
        k = max(self.k, 1)
        pen = np.asarray(self.pen_all, dtype=np.float64)
        fin = pen[np.isfinite(pen)]
        return dict(iters=self.k, S_k=self.pen_sum, S_k_over_k2=self.pen_sum / k ** 2,
                    penalty_first=float(pen[0]) if len(pen) else None,
                    penalty_last=float(pen[-1]) if len(pen) else None,
                    penalty_min=float(fin.min()) if len(fin) else None,
                    penalty_max=float(fin.max()) if len(fin) else None,
                    penalty_last_over_k=float(pen[-1]) / k if len(pen) else None,
                    n_penalty_changes=self.n_pen_changes,
                    n_restart=len(self.restart_iters), restart_fraction=len(self.restart_iters) / k,
                    ergodic_weight_sum=self.W, final_KKT=float(self.last_kkt),
                    first_hit={str(lv): (list(v) if v is not None else None) for lv, v in self.first_hit.items()})

    def result(self):
        arr = np.array(self.rows, dtype=np.float64) if self.rows else np.zeros((0, len(FIELDS)))
        return arr
