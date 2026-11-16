"""C2 methods.  c1/ is an unmodified copy of C1-1.2 (generators, operators, Monitor, AADMM).

(a) dual-scheme comparison on the C1 problems (A = a I, normalized so L_Psi = ||B||^2/mu <= 1/1.001):
  aadmm_c2   AADMM-beta, the equations of c1.methods.run_aadmm (sn-article.tex Alg. 2, (17a)-(19c)),
             plus an optional compatible initialization (Remark rem:dualfista (i)) and a divergence guard.
             With init='zero' it reproduces c1.methods.run_aadmm bit for bit (tests/sanity_c2.py).
  dual_fista FISTA (Beck-Teboulle 2009) with constant step beta on  -d = Phi + Psi:
                 eta^k = prox_{beta Phi}( xi^k - beta grad Psi(xi^k) ),  xi^{k+1} = eta^k + theta_k (eta^k - eta^{k-1}),
             theta_k = (t_k - 1)/t_{k+1} (same t_k as AADMM, eq. (18)); grad Psi(xi) = B y(xi);
             prox_{beta Phi}(s) via the x-subproblem (Lemma lem:srec (a)):  x = argmin f - <s, Ax - b> + beta/2||Ax-b||^2,
             eta = s - beta (A x - b).  Recorded iterate: (x(eta), y(eta), eta); y(eta) is one extra y-solve
             done outside the timed region (Remark rem:dualfista (ii)).  This is the 'fast AMA' of Goldstein et al. (2014).
  dual_pg    same without momentum (proximal gradient on the dual = AMA, Tseng 1991); standard theory: beta < 2/L_Psi.
(b) inexact subproblems on the paper's B = I, g = 1/2||.||^2 splittings (c2/problems.py InnerProblem):
  aadmm_inexact, admm_inexact.
"""
import time
import numpy as np
from c1.metrics import StopRun
from c1.methods import _xstep


class Diverged(Exception):
    pass


def _guard(v, bound):
    nv = float(np.linalg.norm(v))
    if not np.isfinite(nv) or nv > bound:
        raise Diverged(nv)


def _theta_seq():
    t = 1.0
    while True:
        t1 = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        yield t, t1
        t = t1


def run_aadmm_c2(prob, mon, beta=1.0, init="zero", div_bound=np.inf, trace=None):
    """Equations copied verbatim from c1.methods.run_aadmm.  init='compatible': z^1 = B y(lam^1) with lam^1 = 0
    (Remark rem:dualfista (i): tilde lam^0 = lam^1 = xi^1 = 0, tilde y^0 = y(xi^1))."""
    a, b, B = prob.a, prob.b, prob.B
    mon.tic()
    gstar_grad = prob.make_gstar_grad()
    lam = np.zeros(prob.m)
    if init == "zero":
        z = np.zeros(prob.m)
    elif init == "compatible":
        z = B.mv(gstar_grad(B.rmv(lam)))
    else:
        raise ValueError(init)
    zt_prev = z.copy()          # z~^0 = B y~^0 (= z^1)
    lt_prev = lam.copy()        # lam~^0 = lam^1
    t = 1.0
    mon.toc()
    k = 0
    try:
        while True:
            k += 1
            mon.tic()
            s = lam - beta * z                                   # s^k (Lemma lem:srec), traced only
            xt = _xstep(prob, lam, z - b, beta)                  # (17a)
            lt = lam - beta * (a * xt + z - b)                   # (17b)
            yt = gstar_grad(B.rmv(lt))                           # (17c)
            zt = B.mv(yt)
            t1 = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))       # (18)
            z_new = ((t1 - 1.0) / (2 * t1)) * zt - ((t - 1.0) / (2 * t1)) * zt_prev \
                + (t / (2 * t1)) * (lt - lam + (1.0 - beta) * zt + beta * z)          # (19b)
            lam_new = ((t1 - 1.0) / (2 * t1)) * (2.0 * lt - beta * zt) \
                + ((t - 1.0) / (2 * t1)) * (beta * zt_prev - 2.0 * lt_prev) \
                + (t / (2 * t1)) * ((beta - 2.0) * lam + (4.0 - beta) * lt
                                    + (beta * beta - 3.0 * beta) * zt + (2.0 * beta - beta * beta) * z)   # (19c)
            z, lam, zt_prev, lt_prev, t_k, t = z_new, lam_new, zt, lt, t, t1
            mon.toc()
            if trace is not None:
                trace.append(dict(k=k, s=s, lt=lt.copy(), u=lt - beta * zt, r=a * xt + zt - b, t_k=t_k, t_k1=t,
                                  s_next=lam - beta * z, xt=xt.copy(), yt=yt.copy()))
            _guard(lt, div_bound)
            extra = {"Lpred": prob.F(xt, yt) - float(lt @ (a * xt + zt - b))}
            mon.record(xt, yt, lt, zt, weight=t_k, penalty=beta, aadmm_extra=extra)
    except StopRun:
        pass
    except Diverged:
        mon.stop_reason = "diverged"


def _dual_prox_grad(prob, mon, beta, accelerated, div_bound=np.inf, trace=None):
    a, b, B = prob.a, prob.b, prob.B
    mon.tic()
    gstar_grad = prob.make_gstar_grad()
    eta_prev = np.zeros(prob.m)
    xi = np.zeros(prob.m)            # xi^1 = 0  (= AADMM lam^1)
    ts = _theta_seq()
    mon.toc()
    k = 0
    try:
        while True:
            k += 1
            mon.tic()
            gpsi = B.mv(gstar_grad(B.rmv(xi)))          # grad Psi(xi) = B y(xi)
            sarg = xi - beta * gpsi                        # forward point T_beta(xi)
            x = _xstep(prob, sarg, -b, beta)               # prox_{beta Phi}(sarg) via the x-subproblem
            eta = sarg - beta * (a * x - b)
            t_k, t_k1 = next(ts)
            if accelerated:
                xi = eta + ((t_k - 1.0) / t_k1) * (eta - eta_prev)
            else:
                xi = eta
            eta_prev = eta
            mon.toc()
            if trace is not None:
                trace.append(dict(k=k, eta=eta.copy(), sarg=sarg.copy()))
            _guard(eta, div_bound)
            y = gstar_grad(B.rmv(eta)); By = B.mv(y)      # monitoring only (untimed)
            mon.record(x, y, eta, By, weight=1.0, penalty=beta)
    except StopRun:
        pass
    except Diverged:
        mon.stop_reason = "diverged"


def run_dual_fista(prob, mon, beta=1.0, div_bound=np.inf, trace=None):
    _dual_prox_grad(prob, mon, beta, True, div_bound, trace)


def run_dual_pg(prob, mon, beta=1.0, div_bound=np.inf, trace=None):
    _dual_prox_grad(prob, mon, beta, False, div_bound, trace)


METHODS_A = {"aadmm": run_aadmm_c2, "dual_fista": run_dual_fista, "dual_pg": run_dual_pg}


# =====================================================================================================
# (b) inexact x-subproblems
class InexactMonitor:
    """Monitoring for (b).  t_alg includes the inner solver (that is the cost being studied) but not monitoring.
    Logged every iteration up to log_all_until, then with stride max(1, k // log_density).  Per log point:
      dgap   = (p* - d(Pi lam~)) / (1+|p*|)  with Pi the dual-feasible scaling (valid lower bound, always >= 0 up to rounding)
      Lgap   = (p* - L(x~, y~, lam~)) / (1+|p*|)  (the paper's certificate; may be NEGATIVE when x~ is inexact,
               Remark rem:inexact)
      ratio  = (k+1)^2 (p* - d(Pi lam~)) / C   (Theorem Theor-AADMM constant (C) at lam^1 = z^1 = 0)
      objerr = (F(x~, b - A x~) - p*) / (1+|p*|)  objective error of the feasible repair (upper bound)
      rp     = ||A x~ + y~ - b|| / max(||A x~||, ||y~||, ||b||),  Rdx = ||x - prox_f(x + A^T lam)|| / max(||x||, ||A^T lam||)
      dinf   = LASSO: ||A^T lam~||_inf / tau  (> 1: lam~ outside dom Phi);  TVD: 0
      inner  = cumulative inner iterations; cert = inner certificate at this k; eps = requested tolerance."""
    FIELDS = ["k", "t_alg", "t_wall", "dgap", "Lgap", "ratio", "objerr", "rp", "Rdx", "dinf", "inner", "cert", "eps", "met"]

    def __init__(self, prob, ref, cfg, beta):
        self.prob, self.ref, self.cfg = prob, ref, cfg
        self.pstar = ref["pstar"]; self.sc = 1 + abs(self.pstar)
        ls, Bys = ref["lam_star"], ref["By_star"]
        self.C = float(ls @ ls) / beta + float((ls - 2 * Bys) @ (ls - 2 * Bys)) / (2 - beta) if beta < 2 else np.nan
        self.rows = []; self.k = 0; self.t_alg = 0.0; self.t0 = time.perf_counter(); self.inner = 0
        self.stop_reason = None; self.n_unmet = 0; self.n_floor = 0
        self.dgap_all = []; self.cert_all = []; self.eps_all = []

    def tic(self):
        self._tic = time.perf_counter()

    def toc(self):
        self.t_alg += time.perf_counter() - self._tic

    def record(self, x, y, lam, inner_j, cert, eps, Ax):
        P = self.prob
        self.k += 1; k = self.k; self.inner += inner_j
        met = (eps is None) or (cert <= eps * (1 + 1e-9))
        self.n_unmet += (not met)
        lp, dinf = P.dual_feasible(lam)
        dg = (self.pstar - P.d(lp)) / self.sc
        self.dgap_all.append(dg); self.cert_all.append(cert); self.eps_all.append(np.nan if eps is None else eps)
        stop = None
        if not np.isfinite(dg) or abs(dg) > 1e12:
            stop = "diverged"
        elif dg <= self.cfg["tol_dgap"]:
            stop = "tol"
        elif k >= self.cfg["maxit"]:
            stop = "maxit"
        elif time.perf_counter() - self.t0 >= self.cfg["time_budget"]:
            stop = "time_budget"
        if k <= self.cfg["log_all_until"] or k % max(1, k // self.cfg["log_density"]) == 0 or stop is not None:
            r = Ax + y - P.b
            rp = float(np.linalg.norm(r)) / max(float(np.linalg.norm(Ax)), float(np.linalg.norm(y)), float(np.linalg.norm(P.b)), 1e-300)
            ATl = P.AT(lam)
            Rdx = float(np.linalg.norm(x - P.prox_f(x + ATl, 1.0))) / max(float(np.linalg.norm(x)), float(np.linalg.norm(ATl)), 1e-300)
            L = P.F(x, y) - float(lam @ r)
            Lg = (self.pstar - L) / self.sc
            yf = P.b - Ax
            oe = (P.F(x, yf) - self.pstar) / self.sc
            ratio = (k + 1) ** 2 * dg * self.sc / self.C if np.isfinite(self.C) else np.nan
            self.rows.append([k, self.t_alg, time.perf_counter() - self.t0, dg, Lg, ratio, oe, rp, Rdx, dinf,
                              self.inner, cert, np.nan if eps is None else eps, float(met)])
        if stop is not None:
            self.stop_reason = stop
            raise StopRun(stop)

    def result(self):
        return np.array(self.rows, dtype=np.float64) if self.rows else np.zeros((0, len(self.FIELDS)))

    def summary(self):
        return dict(iters=self.k, inner_total=self.inner, n_unmet=self.n_unmet, n_floor_clipped=self.n_floor,
                    t_alg=self.t_alg, C=self.C, final_dgap=self.dgap_all[-1] if self.dgap_all else None)

    def traces(self):
        return dict(dgap_all=np.asarray(self.dgap_all), cert_all=np.asarray(self.cert_all), eps_all=np.asarray(self.eps_all))


def _schedule(sched, k, scale, floor_rel):
    """sched {'mode':'tol','c':c,'p':p}: eps_k = max(c / k^p, floor_rel) * (1+|p*|); returns (eps, clipped)."""
    raw = sched["c"] / k ** sched["p"]
    return max(raw, floor_rel) * scale, raw < floor_rel


def _inner(prob, x, q, beta, sched, k, mon):
    if sched["mode"] == "count":
        x, j, cert = prob.inner_solve(x, q, beta, "count", count=int(sched["n"]))
        return x, j, cert, None
    eps, clipped = _schedule(sched, k, mon.sc, mon.cfg["eps_floor_rel"])
    mon.n_floor += clipped
    x, j, cert = prob.inner_solve(x, q, beta, "tol", tol=eps, maxinner=mon.cfg["maxinner"][prob.kind],
                                  check_every=mon.cfg.get("check_every", 1))
    return x, j, cert, eps


def run_aadmm_inexact(prob, mon, beta=1.0, sched=None):
    """sn-article.tex Alg. 2 (equations as in c1.methods.run_aadmm) with A general, B = I, g = 1/2||.||^2:
    (17a) inexact (warm start x~^{k-1}), (17b) lam~ = lam - beta (A x~ + z - b), (17c) y~ = argmin g - <lam~, y> = lam~."""
    b = prob.b
    mon.tic()
    z = np.zeros(prob.m); lam = np.zeros(prob.m); zt_prev = np.zeros(prob.m); lt_prev = np.zeros(prob.m)
    t = 1.0; xt = np.zeros(prob.nx)
    mon.toc()
    k = 0
    try:
        while True:
            k += 1
            mon.tic()
            q = b - z + lam / beta
            xt, j, cert, eps = _inner(prob, xt, q, beta, sched, k, mon)
            Ax = prob.A(xt)
            lt = lam - beta * (Ax + z - b)
            yt = lt.copy()
            zt = yt
            t1 = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z_new = ((t1 - 1.0) / (2 * t1)) * zt - ((t - 1.0) / (2 * t1)) * zt_prev \
                + (t / (2 * t1)) * (lt - lam + (1.0 - beta) * zt + beta * z)
            lam_new = ((t1 - 1.0) / (2 * t1)) * (2.0 * lt - beta * zt) \
                + ((t - 1.0) / (2 * t1)) * (beta * zt_prev - 2.0 * lt_prev) \
                + (t / (2 * t1)) * ((beta - 2.0) * lam + (4.0 - beta) * lt
                                    + (beta * beta - 3.0 * beta) * zt + (2.0 * beta - beta * beta) * z)
            z, lam, zt_prev, lt_prev, t = z_new, lam_new, zt, lt, t1
            mon.toc()
            mon.record(xt, yt, lt, j, cert, eps, Ax)
    except StopRun:
        pass


def run_admm_inexact(prob, mon, beta=1.0, sched=None):
    """Classical ADMM, sn-article.tex eq. (4) (Sec. 4.2 (solution-2) for beta = 1), inexact x-step (warm start)."""
    b = prob.b
    mon.tic()
    x = np.zeros(prob.nx); y = np.zeros(prob.m); lam = np.zeros(prob.m)
    mon.toc()
    k = 0
    try:
        while True:
            k += 1
            mon.tic()
            q = b - y + lam / beta
            x, j, cert, eps = _inner(prob, x, q, beta, sched, k, mon)
            Ax = prob.A(x)
            y = (lam - beta * (Ax - b)) / (1.0 + beta)
            lam = lam - beta * (Ax + y - b)
            mon.toc()
            mon.record(x, y, lam, j, cert, eps, Ax)
    except StopRun:
        pass


METHODS_B = {"aadmm": run_aadmm_inexact, "admm": run_admm_inexact}
