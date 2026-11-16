"""Algorithms compared in C1.  All run on the SCALED problem (problems.py) with the paper's
sign convention L = f + g - <lam, Ax+By-b>.  A = a*I, so every x-subproblem is a prox of f:
    argmin_x f(x) - <lam, a x> + beta/2 ||a x + c||^2  =  prox_f( (lam/beta - c)/a , 1/(beta a^2) ).

Methods (names used in tasks/results):
  aadmm        AADMM-beta, sn-article.tex Algorithm 2, eqs. (17a)-(17c), (18), (19a)-(19c).
  admm         classical ADMM, sn-article.tex eq. (4), fixed beta.  y-step exact where a closed
               form / one factorization exists (P3, P4); for P1, P2 (elastic-net g with general B,
               no closed form) the y-step is linearized (proximal ADMM, Q = beta(mu I - B^T B)).
  xu_fixed     Xu (2017) Alg. 2 (1.15a)-(1.15c) with fixed parameters (2.26) as in Xu (2017) Sec. 3.3:
               beta=gamma=1/(2||B||^2), P=0, Q = I/2 - B^T B/(2||B||^2) (linearized y-step).
  xu_acc       Xu (2017) Alg. 2 with adaptive parameters (2.27): beta_k=gamma_k=(k+1)gamma, P^k=0,
               Q^k=(k+1)(q0 I - gamma B^T B), q0 = mu/20, gamma = q0/||B||^2 (as in Xu (2017) Sec. 3.3).
  iadmm        Tang-Toh Algorithm 1 (IADMM) with adaptive strategy (35) of the journal version (MPC 16, 2024).
  heu_admm     Tang-Toh (2024) Heu-ADMM: same iteration with the heuristic rule (40) of the journal version.
  fast_admm_r  Goldstein-O'Donoghue-Setzer-Baraniuk (SIIMS 2014) fast ADMM with restart
               (their Alg. 8), only where the exact y-step is available (P3, P4).
Every UNCERTAIN implementation choice is marked 'UNCERTAIN' and listed in README.md.
"""
import numpy as np
from .metrics import StopRun, kkt_parts


def _xstep(prob, lam, c, beta):
    a = prob.a
    return prob.prox_f((lam / beta - c) / a, 1.0 / (beta * a * a))


# ---------------------------------------------------------------------------------------
def run_aadmm(prob, mon, beta=1.0, pp_every=1, trace=None):
    """sn-article.tex Algorithm 2 (AADMM-beta).  Output/recorded iterate: the predictor
    (x~^k, y~^k, lam~^k) (Alg. 2 'Output').  Ergodic iterate: t_k-weighted average of the
    predictors (memo Theorem D: xbar^k = t_k^{-2} sum_j t_j x~^j).  Also records
    L(x~,y~,lam~) and the post-processed point of memo Proposition E (one extra x-solve,
    done outside the timed region)."""
    a, b, B = prob.a, prob.b, prob.B
    mon.tic()
    gstar_grad = prob.make_gstar_grad()          # y-subproblem (17c) = argmin g(y) - <B^T lam~, y>
    # Input of Alg. 2: x~^0 = 0, lam~^0 = 0, y~^0 = 0 (z~^0 = B y~^0 = 0);
    # x^1 = x~^0, z^1 = z~^0, lam^1 = lam~^0, t_1 = 1.
    z = np.zeros(prob.m)
    lam = np.zeros(prob.m)
    zt_prev = np.zeros(prob.m)       # z~^{k-1}
    lt_prev = np.zeros(prob.m)       # lam~^{k-1}
    t = 1.0
    mon.toc()
    k = 0
    try:
        while True:
            k += 1
            mon.tic()
            # (17a): x~^k = argmin f(x) - <lam^k, Ax + z^k - b> + beta/2 ||Ax + z^k - b||^2
            xt = _xstep(prob, lam, z - b, beta)
            # (17b): lam~^k = lam^k - beta (A x~^k + z^k - b)
            lt = lam - beta * (a * xt + z - b)
            # (17c): y~^k = argmin g_c(y) - <lam~^k, By> + 1/2||By||^2 = argmin g(y) - <B^T lam~^k, y>
            yt = gstar_grad(B.rmv(lt))
            zt = B.mv(yt)                                  # z~^k := B y~^k
            # (18): t_{k+1} = (1 + sqrt(1 + 4 t_k^2)) / 2
            t1 = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            # (19a): x^{k+1} = x~^k  (not needed by the recursion: (17a) does not read x^k)
            # (19b): z^{k+1}
            z_new = ((t1 - 1.0) / (2 * t1)) * zt - ((t - 1.0) / (2 * t1)) * zt_prev \
                + (t / (2 * t1)) * (lt - lam + (1.0 - beta) * zt + beta * z)
            # (19c): lam^{k+1}
            lam_new = ((t1 - 1.0) / (2 * t1)) * (2.0 * lt - beta * zt) \
                + ((t - 1.0) / (2 * t1)) * (beta * zt_prev - 2.0 * lt_prev) \
                + (t / (2 * t1)) * ((beta - 2.0) * lam + (4.0 - beta) * lt
                                    + (beta * beta - 3.0 * beta) * zt + (2.0 * beta - beta * beta) * z)
            z, lam, zt_prev, lt_prev, t_k, t = z_new, lam_new, zt, lt, t, t1
            mon.toc()
            if trace is not None:     # used only by tests/sanity.py
                trace.append(dict(xt=xt.copy(), yt=yt.copy(), lt=lt.copy(), z=z.copy(), lam=lam.copy()))
            # ---- monitoring (untimed) ----
            extra = {"Lpred": prob.F(xt, yt) - float(lt @ (a * xt + zt - b))}
            if pp_every and (k % pp_every == 0):
                # memo Prop. E: x^ = argmin f - <lam~, Ax + B y~ - b> + beta/2||Ax + B y~ - b||^2,
                #               lam^ = lam~ - beta (A x^ + B y~ - b)
                xh = _xstep(prob, lt, zt - b, beta)
                lh = lt - beta * (a * xh + zt - b)
                extra["pp"] = (xh, lh)
            mon.record(xt, yt, lt, zt, weight=t_k, penalty=beta, aadmm_extra=extra)
    except StopRun:
        pass


# ---------------------------------------------------------------------------------------
def _proximal_admm(prob, mon, beta_rule, gamma_rule, q_rule, exact, weight_rule,
                   adapt=None, beta0=None, prox_g1_for_adapt=None):
    """Generic (proximal) ADMM:
        x^{k+1} = argmin f - <lam, Ax> + beta_k/2||Ax + By^k - b||^2
        y^{k+1} = exact:  argmin g - <lam, By> + beta_k/2||Ax^{k+1} + By - b||^2
                  linearized: argmin g - <lam,By> + beta_k/2||..||^2 + 1/2||y-y^k||^2_{q_k I - beta_k B^T B}
                            = prox_g( y^k - (beta_k B^T r^ - B^T lam)/q_k , 1/q_k ),  r^ = Ax^{k+1}+By^k-b
        lam^{k+1} = lam - gamma_k (Ax^{k+1} + By^{k+1} - b)
    """
    a, b, B = prob.a, prob.b, prob.B
    mon.tic()
    x = np.zeros(prob.nx); y = np.zeros(prob.ny); lam = np.zeros(prob.m)
    By = np.zeros(prob.m)
    prox_g = None if exact else prob.make_prox_g()
    solvers = {}
    beta = beta0
    mon.toc()
    k = 0
    try:
        while True:
            k += 1
            mon.tic()
            if beta_rule is not None:
                beta = beta_rule(k)
            gam = gamma_rule(k, beta)
            x = _xstep(prob, lam, By - b, beta)
            Ax = a * x
            if exact:
                key = float(beta)
                if key not in solvers:
                    if len(solvers) > 8:
                        solvers.clear()
                    solvers[key] = prob.make_admm_ysolver(beta)
                y = solvers[key](lam, Ax - b)
            else:
                q = q_rule(k, beta)
                rh = Ax + By - b
                y = prox_g(y - (beta * B.rmv(rh) - B.rmv(lam)) / q, 1.0 / q)
            By = B.mv(y)
            lam = lam - gam * (Ax + By - b)
            used_beta = beta
            if adapt is not None:
                # Tang-Toh (2024) Sec. 5.2: R_p^{k+1}, R_d^{k+1} of the new iterate (part of the method's cost)
                Rp, Rdx, Rdy, _ = kkt_parts(prob, x, y, lam, By, prox_g1_for_adapt)
                beta = adapt(beta, Rp, max(Rdx, Rdy))
            mon.toc()
            mon.record(x, y, lam, By, weight=weight_rule(k, used_beta), penalty=used_beta)
    except StopRun:
        pass


def run_admm(prob, mon, beta=1.0):
    exact = prob.has_exact_y
    mu = prob.mu
    _proximal_admm(prob, mon, beta_rule=lambda k: beta, gamma_rule=lambda k, bt: bt,
                   q_rule=lambda k, bt: bt * mu,     # q >= beta ||B||^2 since ||B||^2 <= mu after scaling
                   exact=exact, weight_rule=lambda k, bt: 1.0)


def run_xu_fixed(prob, mon, c=0.5):
    """Xu (2017) journal Thm 2.13 / (2.26) with the constants of Xu (2017) Sec. 3.3 ('Nonaccelerated Linearized ADMM'):
    beta=gamma=1/(2||B||^2), Q = I/2 - B^T B/(2||B||^2).  After scaling ||B||^2 <= mu; we use mu
    in place of ||B||^2 (conservative, keeps Q >= 0).  Ergodic: uniform average (journal Thm 2.13)."""
    mu = prob.mu
    beta = c / mu
    _proximal_admm(prob, mon, beta_rule=lambda k: beta, gamma_rule=lambda k, bt: bt,
                   q_rule=lambda k, bt: c, exact=False, weight_rule=lambda k, bt: 1.0)


def run_xu_acc(prob, mon, q0_frac=1.0 / 20.0):
    """Xu (2017) journal Thm 2.14 / (2.27), P = 0, smooth f_Xu = 0, L_f = mu_f = 0, mu_g = mu:
        beta_k = gamma_k = (k+1) gamma,  Q^k = (k+1)(Q - gamma B^T B),  Q = q0 I,
        constraint eta*gamma*B^T B <= Q <= mu/2 I (eta >= 1).  Xu (2017) Sec. 3.3 uses q0 = mu2/20 and
        gamma = q0/||B||^2 (eta = 1); we use the same (||B||^2 replaced by mu, conservative).
        Then beta_k B^T B + Q^k = (k+1) q0 I, i.e. q_k = (k+1) q0 in the linearized y-step.
    Ergodic weights (k + k0 + 1), k0 = ceil(1 + 2(L_f - mu_f)/(mu_g + mu_f)) = 1 (journal Thm 2.14).
    Iteration counter: Xu (2017) uses k = 1,2,... for the parameters (2.27) with initial (y^1, z^1, lam^1)."""
    mu = prob.mu
    q0 = q0_frac * mu
    gam0 = q0 / mu
    k0 = 1
    _proximal_admm(prob, mon, beta_rule=lambda k: (k + 1) * gam0, gamma_rule=lambda k, bt: bt,
                   q_rule=lambda k, bt: (k + 1) * q0, exact=False,
                   weight_rule=lambda k, bt: k + k0 + 1.0)


def _tt_setup(prob, beta1_mult, tt_q="exact"):
    """Shared set-up for Tang-Toh (2024) IADMM / Heu-ADMM.
    UNCERTAIN: Q.  Tang-Toh (2024) Alg. 1 leaves Q >= 0 free; Sec. 5.1 suggests Q = lambda_max(C^T C) I - C^T C to make
    the strongly convex block explicit.  We use Q = 0 (exact y-step) where an exact one-factorization
    y-step exists (P3: Cholesky of H + beta_k B^T B, refactorized whenever beta_k changes; P4: DCT) and
    Q = mu I - C^T C (linearized) for P1, P2.  lambda_max(C^T C + Q) is then ||B||^2 resp. mu.
    UNCERTAIN: beta_1.  Tang-Toh (2024) Sec. 6.2 uses beta_1 = beta = 1/(2||B||^2) (Example 4); Sec. 6.1 sweeps beta_1 in
    [1e-5, 1e5].  Default here: beta_1 = 1/(2 mu) (= 1/(2||B||^2) up to the 1.001 safety factor), times
    beta1_mult in the robustness sweep.
    UNCERTAIN: lower bound beta.  Tang-Toh (2024) states beta = 1e-6 (absolute, unscaled problem); since we run on the
    constraint scaled by s, the equivalent floor is 1e-6 / s^2.
    tt_q = "lin" switches P3 to the linearized y-step (Q = mu I - C^T C, one eigendecomposition of H)."""
    exact = prob.has_exact_y and not (prob.name == "P3" and tt_q == "lin")
    lam_max = prob.normB2 if exact else prob.mu
    beta1 = beta1_mult * 0.5 / prob.mu
    beta_floor = 1e-6 / prob.s ** 2
    return exact, lam_max, beta1, beta_floor


def run_iadmm(prob, mon, beta1_mult=1.0, gamma=1.618, eps=1e-4, tt_q="exact"):
    """Tang-Toh (2024) Algorithm 1 (IADMM) + strategy (35) (journal version, MPC 16, 2024).  Tang-Toh (2024) Sec. 5.2: gamma = 1.618, eps = 1e-4, beta = 1e-6.
    Paper sign: lam_TT = -lam_ours, so step 3 'lam + gamma beta_k r' becomes lam - gamma beta_k r here.
    Ergodic weights beta_k (journal Theorem 1 / Corollary 1: v^k = sum beta_i x^{i+1} / sum beta_i)."""
    exact, lam_max, beta1, floor = _tt_setup(prob, beta1_mult, tt_q)
    requested_beta1 = beta1
    beta1 = max(beta1, floor)  # Algorithm 1 requires the initial penalty to satisfy its floor.
    mon.method_meta = dict(beta1_requested=requested_beta1, beta1_used=beta1,
                           beta_floor=floor, penalty_policy="residual_balance")
    sig = prob.mu
    mu = prob.mu
    mon.tic()
    pg1 = prob.make_prox_g_fixed(1.0)     # needed for R_d in rule (35) (journal version); timed as part of the method
    mon.toc()

    def adapt(bk, Rp, Rd):
        if Rp > Rd:
            return np.sqrt(bk * bk + (1.0 - eps) * sig * bk / lam_max)
        if Rp < Rd / 10.0:
            return max(floor, bk / 1.5)
        return bk
    _proximal_admm(prob, mon, beta_rule=None, gamma_rule=lambda k, bt: gamma * bt,
                   q_rule=lambda k, bt: bt * mu, exact=exact, weight_rule=lambda k, bt: bt,
                   adapt=adapt, beta0=beta1, prox_g1_for_adapt=pg1)


def run_heu_admm(prob, mon, beta1_mult=1.0, gamma=1.618, tt_q="exact"):
    """Tang-Toh (2024) Heu-ADMM, rule (40) (journal version).  UNCERTAIN: Tang-Toh (2024) does not state gamma or Q for Heu-ADMM separately;
    we use the IADMM ones (gamma = 1.618, same Q)."""
    exact, lam_max, beta1, floor = _tt_setup(prob, beta1_mult, tt_q)
    # rule (40) has no interval floor; beta_1 is NOT clamped (recorded for the sweep tables).
    mon.method_meta = dict(beta1_requested=beta1, beta1_used=beta1, beta_floor=None,
                           penalty_policy="heuristic_rule39_no_floor")
    mu = prob.mu
    mon.tic()
    pg1 = prob.make_prox_g_fixed(1.0)
    mon.toc()

    def adapt(bk, Rp, Rd):
        if Rp > 10.0 * Rd:
            return 1.5 * bk
        if Rp < Rd / 10.0:
            return bk / 1.5
        return bk
    _proximal_admm(prob, mon, beta_rule=None, gamma_rule=lambda k, bt: gamma * bt,
                   q_rule=lambda k, bt: bt * mu, exact=exact, weight_rule=lambda k, bt: bt,
                   adapt=adapt, beta0=beta1, prox_g1_for_adapt=pg1)


def run_fast_admm_restart(prob, mon, tau=1.0, eta=0.999):
    """Goldstein et al. (SIIMS 2014) Algorithm 8 'Fast ADMM with restart', roles u <- x (f), v <- y (g):
        x_k   = argmin f - <lamh_k, Ax> + tau/2||b - Ax - B vh_k||^2
        y_k   = argmin g - <lamh_k, By> + tau/2||b - A x_k - By||^2
        lam_k = lamh_k + tau (b - A x_k - B y_k)
        c_k   = ||lam_k - lamh_k||^2/tau + tau ||B(y_k - vh_k)||^2
        if c_k < eta c_{k-1}: alpha_{k+1} = (1+sqrt(1+4 alpha_k^2))/2,
              vh_{k+1} = y_k + (alpha_k-1)/alpha_{k+1} (y_k - y_{k-1}),  lamh likewise
        else: alpha_{k+1} = 1, vh_{k+1} = y_{k-1}, lamh_{k+1} = lam_{k-1}, c_k <- c_{k-1}/eta.
    UNCERTAIN: tau (no rule in Goldstein et al. for this setting); we use tau = 1 on the scaled problem,
    the same default as admm.  Only the restart variant is used (Alg. 7 needs both blocks strongly convex)."""
    a, b, B = prob.a, prob.b, prob.B
    mon.tic()
    ysolve = prob.make_admm_ysolver(tau)
    if ysolve is None:
        raise RuntimeError("fast_admm_r requires an exact y-step")
    y_prev = np.zeros(prob.ny); By_prev = np.zeros(prob.m); lam_prev = np.zeros(prob.m)
    vh = y_prev.copy(); Bvh = By_prev.copy(); lamh = lam_prev.copy()
    alpha = 1.0
    c_prev = np.inf
    mon.toc()
    try:
        while True:
            mon.tic()
            x = _xstep(prob, lamh, Bvh - b, tau)
            Ax = a * x
            y = ysolve(lamh, Ax - b)
            By = B.mv(y)
            lam = lamh - tau * (Ax + By - b)
            ck = float(np.sum((lam - lamh) ** 2)) / tau + tau * float(np.sum((By - Bvh) ** 2))
            restarted = not (ck < eta * c_prev)
            if ck < eta * c_prev:
                alpha1 = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * alpha * alpha))
                w = (alpha - 1.0) / alpha1
                vh = y + w * (y - y_prev); Bvh = By + w * (By - By_prev); lamh = lam + w * (lam - lam_prev)
                alpha = alpha1
                c_prev = ck
            else:
                alpha = 1.0
                vh = y_prev.copy(); Bvh = By_prev.copy(); lamh = lam_prev.copy()
                c_prev = c_prev / eta
            y_prev, By_prev, lam_prev = y, By, lam
            mon.toc()
            mon.record(x, y, lam, By, weight=1.0, penalty=tau, restart=restarted)
    except StopRun:
        pass


METHODS = {
    "aadmm": run_aadmm,
    "admm": run_admm,
    "xu_fixed": run_xu_fixed,
    "xu_acc": run_xu_acc,
    "iadmm": run_iadmm,
    "heu_admm": run_heu_admm,
    "fast_admm_r": run_fast_admm_restart,
}
