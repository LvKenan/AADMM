"""(b) Problems with iterative x-subproblems: the paper's own B = I, g = 1/2||.||^2 splittings (sn-article.tex
Sec. 4.2 LASSO model (model-experiment), Sec. 4.3 TV dual model (DM)), in the form
        min f(x) + 1/2||y||^2   s.t.  A x + y = b       (B = I, mu = 1: normalization (5) holds with equality).
  LASSO: f = tau||x||_1, A in R^{m x n} standard normal (as in the paper), x0 with 10% nonzeros, noise 0.05
         (sn-article.tex Table "LASSO: standard ADMM vs AADMM": sparsity 0.1, noise level 0.05),
         tau = 0.1 ||A^T b||_inf (UNCERTAIN: the paper does not state tau for these sizes).
  TVD  : x = p in R^{2N^2}, f = indicator{ |p_i|_2 <= alpha } (isotropic), A = div = -grad^T (Neumann), b = f0:
         the image f0 of C1 problem P4 (same generator and seed), alpha = 0.1 (paper: alpha = 0.1, noise variance 0.01).
         The multiplier is the image u; d(u) = <u,f0> - 1/2||u||^2 - alpha TV(u) = 1/2||f0||^2 - ROF(u),
         so p* = 1/2||f0||^2 - F*_P4 and u* = y*_P4 (C1-1.2 reference, see refs/).
x-subproblem (both): min_x h(x) = f(x) + beta/2 ||A x - q||^2,  q = b - z + lam/beta  (paper (17a) with B y = z).
Inner certificate (computable upper bound of h(x) - min h, Fenchel duality):
   v = beta (A x - q) (LASSO: scaled into {||A^T v||_inf <= tau});
   gap(x) = h(x) - [ -f^*(-A^T v) - <v, q> - ||v||^2/(2 beta) ]  >= h(x) - min h.
"""
import numpy as np
from c1.problems import Counter, GradOp, P4_TV, soft


class InnerProblem:
    def __init__(self, spec):
        self.spec = dict(spec)
        self.kind = spec["kind"]
        seed = spec["seed"]
        self.seed = seed
        self.mu = 1.0
        if self.kind == "LASSO":
            m, n = spec["m"], spec["n"]
            rng = np.random.default_rng(1000 + seed)
            M = rng.standard_normal((m, n))
            x0 = np.zeros(n)
            nnz = max(1, int(round(0.1 * n)))
            idx = rng.choice(n, nnz, replace=False)
            x0[idx] = rng.standard_normal(nnz)
            b = M @ x0
            b = b + 0.05 * np.linalg.norm(b) / np.sqrt(m) * rng.standard_normal(m)
            self.M = M
            self.b = b
            self.tau = 0.1 * float(np.max(np.abs(M.T @ b)))
            self.nx, self.m = n, m
            sv = np.linalg.svd(M, compute_uv=False)
            self.LA = float(sv[0] ** 2) * (1 + 1e-12)        # ||A||^2 (inner Lipschitz constant / beta)
            self.inst = f"LASSO_m{m}_n{n}_s{seed}"
            self.p4 = None
        elif self.kind == "TVD":
            N = spec["N"]
            p4 = P4_TV(N, seed)          # same generator, same seed as C1 P4 (image f0)
            self.p4inst = p4.inst
            self.N = N
            self.G = GradOp(N, Counter(), c=1.0)
            self.b = p4.f0.copy()
            self.alpha = p4.alpha
            self.nx, self.m = 2 * N * N, N * N
            self.LA = float(self.G.eig_lap.max()) * (1 + 1e-12)   # ||div||^2 = ||grad||^2 (< 8)
            self.inst = f"TVD_N{N}_s{seed}"
        else:
            raise ValueError(self.kind)

    # ---- operators
    def A(self, x):
        return self.M @ x if self.kind == "LASSO" else -self.G.rmv(x)

    def AT(self, v):
        return self.M.T @ v if self.kind == "LASSO" else -self.G.mv(v)

    # ---- f, prox_f, f^*
    def f(self, x):
        if self.kind == "LASSO":
            return self.tau * float(np.abs(x).sum())
        nrm = self._pn(x)
        return 0.0 if float(nrm.max()) <= self.alpha * (1 + 1e-9) else np.inf

    def _pn(self, x):
        NN = self.N * self.N
        return np.sqrt(x[:NN] ** 2 + x[NN:] ** 2)

    def prox_f(self, v, t):
        if self.kind == "LASSO":
            return soft(v, t * self.tau)
        fac = 1.0 / np.maximum(self._pn(v) / self.alpha, 1.0)
        return v * np.concatenate([fac, fac])

    def fconj(self, w):
        if self.kind == "LASSO":
            return 0.0 if float(np.max(np.abs(w))) <= self.tau * (1 + 1e-10) else np.inf
        return self.alpha * float(self._pn(w).sum())

    def dual_feasible(self, lam):
        """lam itself if A^T lam in dom f^*, else the scaled point (LASSO) giving a valid lower bound."""
        if self.kind == "LASSO":
            r = float(np.max(np.abs(self.AT(lam)))) / self.tau
            return (lam / r if r > 1 else lam), r
        return lam, 0.0

    def d(self, lam):
        """Lagrange dual value d(lam) = <lam,b> - 1/2||lam||^2 - f^*(A^T lam)."""
        return float(lam @ self.b) - 0.5 * float(lam @ lam) - self.fconj(self.AT(lam))

    def F(self, x, y):
        return self.f(x) + 0.5 * float(y @ y)

    # ---- inner certificate and solvers
    def inner_gap(self, x, q, beta, Ax=None):
        Ax = self.A(x) if Ax is None else Ax
        res = Ax - q
        v = beta * res
        ATv = self.AT(v)
        if self.kind == "LASSO":
            r = float(np.max(np.abs(ATv))) / self.tau
            if r > 1:
                v = v / r; ATv = ATv / r
        upper = self.f(x) + 0.5 * beta * float(res @ res)
        lower = -self.fconj(-ATv) - float(v @ q) - 0.5 * float(v @ v) / beta
        return upper - lower, upper

    def inner_solve(self, x0, q, beta, mode, tol=None, count=None, maxinner=20000, check_every=1):
        """mode 'count': paper style, fixed number of warm-started steps of ISTA (LASSO, sn-article (FP-ADMM))
        or Chambolle's projection algorithm (TVD, Chambolle 2004, step 1/8).
        mode 'tol'  : warm-started FISTA with gradient restart until inner_gap <= tol (or maxinner).
        Returns x, number of inner iterations, final certificate (computed in both modes)."""
        if mode == "count":
            x = x0.copy()
            if self.kind == "LASSO":
                step = 1.0 / (beta * self.LA)
                for _ in range(count):
                    x = soft(x - step * beta * self.AT(self.A(x) - q), step * self.tau)
            else:
                # min_{|p|<=alpha} 1/2||div p - q||^2, p = alpha pbar:  pbar <- (pbar + s grad(div pbar - q/alpha)) / (1 + s|.|)
                s = 0.125
                pb = x / self.alpha
                for _ in range(count):
                    gr = self.G.mv(-self.G.rmv(pb) - q / self.alpha)
                    nrm = self._pn(gr)
                    den = np.concatenate([1 + s * nrm, 1 + s * nrm])
                    pb = (pb + s * gr) / den
                x = self.alpha * pb
            gap, _ = self.inner_gap(x, q, beta)
            return x, count, gap
        step = 1.0 / (beta * self.LA)
        x = x0.copy(); xp = x.copy(); yv = x.copy(); th = 1.0
        gap, h_prev = self.inner_gap(x, q, beta)
        j = 0
        while gap > tol and j < maxinner:
            j += 1
            Ay = self.A(yv)
            xn = self.prox_f(yv - step * beta * self.AT(Ay - q), step)
            if float((yv - xn) @ (xn - x)) > 0:      # gradient restart (O'Donoghue-Candes)
                th = 1.0; yv = xn.copy()
            else:
                th1 = 0.5 * (1 + np.sqrt(1 + 4 * th * th))
                yv = xn + ((th - 1) / th1) * (xn - x); th = th1
            x = xn
            if j % check_every == 0 or j == maxinner:
                gap, _ = self.inner_gap(x, q, beta)
        return x, j, gap


def lasso_reference(P, tol=1e-13, maxit=200000, time_budget=1800.0):
    """Independent FISTA + restart on primal LASSO; UB=primal F; LB=d(scaled residual).
    Reference passes only if relative primal-dual gap <= tol. Float64, not interval arithmetic."""
    import time
    t0=time.perf_counter();M,b,tau=P.M,P.b,P.tau;L=P.LA
    x=np.zeros(P.nx);yv=x.copy();th=1.;ubest,lbest=np.inf,-np.inf;bestx=x.copy()
    status='maxit';gap=np.inf
    for it in range(1,maxit+1):
        xn=soft(yv-M.T@(M@yv-b)/L,tau/L)
        if (yv-xn)@(xn-x)>0:th=1.;yv=xn.copy()
        else:
            tn=(1+np.sqrt(1+4*th*th))/2;yv=xn+(th-1)/tn*(xn-x);th=tn
        x=xn
        if it<50 or it%10==0:
            rr=b-M@x;ub=tau*np.abs(x).sum()+.5*(rr@rr);lp,_=P.dual_feasible(rr);lb=P.d(lp)
            if ub<ubest:ubest=ub;bestx=x.copy()
            lbest=max(lb,lbest);gap=(ubest-lbest)/(1+abs(lbest))
            if gap < -1e-12:raise ArithmeticError('reference UB below LB')
            if gap<=tol:status='tol';break
        if time.perf_counter()-t0>time_budget:status='time_budget';break
    ys=b-M@bestx
    return dict(pstar=float(ubest),LB=float(lbest),UB=float(ubest),gap_rel=float(gap),iters=it,status=status,time=time.perf_counter()-t0,lam_star=ys,By_star=ys,x_star=bestx)
