"""High-accuracy reference solve for F* (independent code path, not AADMM):
restarted accelerated proximal gradient (FISTA + O'Donoghue-Candes gradient restart) on the
Lagrange dual   max_lam d(lam) = -f^*(A^T lam) + <lam, b> - g^*(B^T lam),
with grad g^*(B^T lam) = B y(lam), Lipschitz const ||B||^2/mu <= 1 after scaling (step = 1).
Certificate:
  P1, P2, P4: UB = F(x_feas(y(lam)), y(lam)) (primal feasible), LB = d(lam);  gap = UB - LB.
  P3       : repair y toward the generator's strictly feasible anchor to obtain a feasible UB.
  All gaps are floating-point primal-dual estimates, not interval certificates.
"""
import time
import numpy as np


def prox_fconj_a(prob, u_lam, t):
    """prox of lam -> t * f^*(a lam):  lam = prox_{t a^2 f^*}(a u)/a, Moreau: prox_{s f^*}(w) = w - s prox_f(w/s, 1/s)."""
    a = prob.a
    s = t * a * a
    w = a * u_lam
    return (w - s * prob.prox_f(w / s, 1.0 / s)) / a


def reference_solve(prob, tol=1e-13, maxit=300000, time_budget=600.0):
    t0 = time.perf_counter()
    gstar = prob.make_gstar_grad()
    B, b = prob.B, prob.b
    if not prob.has_feas:
        anchor = prob.feasible_anchor
        anchor_x = (b - B.mv(anchor)) / prob.a
    L = max(prob.normB2 / prob.mu, 1e-12)
    step = 1.0 / L
    lam = np.zeros(prob.m)
    yk = lam.copy()
    th = 1.0
    best_lb, best_ub = -np.inf, np.inf
    best_lam = lam.copy(); best_y = None
    hist = []
    it = 0
    gap_rel = np.inf
    status = "maxit"
    while it < maxit:
        it += 1
        yv = gstar(B.rmv(yk))                 # y(yk)
        grad = B.mv(yv)                       # grad Psi(yk)
        # prox-grad step on phi = Psi + Phi, Phi(lam) = f^*(a lam) - <lam, b>
        lam_new = prox_fconj_a(prob, yk - step * grad + step * b, step)
        # gradient restart (O'Donoghue-Candes)
        if float((yk - lam_new) @ (lam_new - lam)) > 0:
            th = 1.0
            yk = lam_new.copy()
        else:
            th1 = 0.5 * (1 + np.sqrt(1 + 4 * th * th))
            yk = lam_new + ((th - 1) / th1) * (lam_new - lam)
            th = th1
        lam = lam_new
        if it % 10 == 0 or it < 50:
            w = B.rmv(lam)
            y = gstar(w)
            By = B.mv(y)
            lb = -prob.fconj(prob.a * lam) - prob.gconj_val(w, y) + float(lam @ b)
            if lb > best_lb:
                best_lb, best_lam, best_y = lb, lam.copy(), y.copy()
            xf = prob.x_feas(y, By)
            if xf is not None:
                ub = prob.F(xf, y)
            else:
                xr = (b - By) / prob.a
                direction = xr - anchor_x
                ratios = np.full_like(direction, np.inf)
                np.divide(prob.hi-anchor_x, direction, out=ratios, where=direction > 0)
                np.divide(prob.lo-anchor_x, direction, out=ratios, where=direction < 0)
                fraction = min(1.0, float(ratios.min()))
                if fraction < 1.0:
                    fraction = max(0.0, fraction * (1 - 32 * np.finfo(float).eps))
                yf = anchor + fraction * (y-anchor)
                xf = (b - B.mv(yf)) / prob.a
                ub = prob.F(xf, yf)
            best_ub = min(best_ub, ub)
            gap = best_ub - best_lb
            if gap < -1e-10 * (1 + abs(best_lb) + abs(best_ub)):
                raise ArithmeticError("Reference primal upper bound is below the dual lower bound")
            gap_rel = abs(gap) / (1.0 + abs(best_lb))
            if it % 1000 == 0:
                hist.append((it, time.perf_counter() - t0, best_lb, best_ub, gap_rel))
            if gap_rel <= tol:
                status = "tol"
                break
        if time.perf_counter() - t0 > time_budget:
            status = "time_budget"
            break
    w = B.rmv(best_lam)
    y = gstar(w)
    By = B.mv(y)
    Fstar = best_ub
    return dict(Fstar=Fstar, LB=best_lb, UB=best_ub, gap_rel=gap_rel, iters=it, status=status,
                time=time.perf_counter() - t0, lam_star=best_lam, y_star=y, By_star=By,
                hist=np.array(hist) if hist else np.zeros((0, 5)))
