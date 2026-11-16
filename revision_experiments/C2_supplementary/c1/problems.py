"""Test problems for C1 (memo Revision_Suggestions_CN.md, Sec. 4.3.1, problems P1-P4).

All problems are written in the paper's form (sn-article.tex eq. (1)/(6))
    min f(x) + g(y)   s.t.  A x + B y = b,        A = a*I (scalar a),
with g mu-strongly convex, and are scaled as in sn-article.tex Sec. 2.1 (before eq. (5)):
    A <- s A, B <- s B, b <- s b,  s = sqrt(mu / ||B||^2),
so that mu*I - B^T B >= 0 (eq. (5)). Dense operators use a largest-eigenvalue
calculation and GradOp uses its known spectrum, with numerical safety margins.
Scaling the constraint does not change the primal solution set; relative residuals,
objective values and dual values are scale invariant (see README).

Every problem object exposes the closed-form pieces needed by all methods:
    f(x), prox_f(v, t)        prox_f(v,t) = argmin_x f(x) + ||x-v||^2/(2t)
    g(y), make_prox_g()       returns a callable prox_g(v, t) (may own a factorization)
    make_gstar_grad()         returns callable w -> argmin_y g(y) - <w, y>  (AADMM (17c))
    make_admm_ysolver(beta)   exact ADMM y-step  argmin_y g(y)-<lam,By>+beta/2||By+c||^2,
                              or None if no closed form / one-factorization solution exists
    fconj(u), proj_dual(lam)  f^* at u = A^T lam, and projection of lam onto {lam: A^T lam in dom f^*}
    gconj_val(w, ystar)       g^*(w) given ystar = argmin g - <w,.>
    x_feas(y)                 x with A x + B y = b and f(x) < inf, or None
"""
import numpy as np
import scipy.linalg as sla
import scipy.fft as sfft

SAFETY = 1.001


class Counter:
    def __init__(self):
        self.matvec = 0
        self.fact = 0

    def reset(self):
        self.matvec = 0
        self.fact = 0


class DenseOp:
    """B = scale * M (dense); counts products with B and B^T."""

    def __init__(self, M, counter):
        self.M = np.ascontiguousarray(M, dtype=np.float64)
        self.cnt = counter
        self.shape = M.shape

    def mv(self, v):
        self.cnt.matvec += 1
        return self.M @ v

    def rmv(self, w):
        self.cnt.matvec += 1
        return self.M.T @ w

    def scaled(self, s):
        return DenseOp(s * self.M, self.cnt)


class GradOp:
    """B = c * grad (forward differences, Neumann BC) on N x N images, flattened.
    Output layout: [dx (N*N), dy (N*N)]."""

    def __init__(self, N, counter, c=1.0):
        self.N = N
        self.c = c
        self.cnt = counter
        self.shape = (2 * N * N, N * N)
        k = np.arange(N)
        lam1 = 2.0 - 2.0 * np.cos(np.pi * k / N)
        self.eig_lap = lam1[:, None] + lam1[None, :]   # eigenvalues of grad^T grad in DCT-II basis

    def mv(self, u):
        self.cnt.matvec += 1
        N = self.N
        U = u.reshape(N, N)
        dx = np.zeros_like(U)
        dy = np.zeros_like(U)
        dx[:-1, :] = U[1:, :] - U[:-1, :]
        dy[:, :-1] = U[:, 1:] - U[:, :-1]
        return self.c * np.concatenate([dx.ravel(), dy.ravel()])

    def rmv(self, p):
        self.cnt.matvec += 1
        N = self.N
        px = p[:N * N].reshape(N, N)
        py = p[N * N:].reshape(N, N)
        out = np.zeros((N, N))
        # adjoint of forward difference with Neumann BC: -div
        out[:-1, :] -= px[:-1, :]
        out[1:, :] += px[:-1, :]
        out[:, :-1] -= py[:, :-1]
        out[:, 1:] += py[:, :-1]
        return self.c * out.ravel()

    def scaled(self, s):
        return GradOp(self.N, self.cnt, self.c * s)


def norm2_bound(op):
    """Numerical spectral bound; unlike a power iterate, it targets the largest eigenvalue."""
    if isinstance(op, GradOp):
        value = op.c ** 2 * float(op.eig_lap.max())
    elif isinstance(op, DenseOp):
        M = op.M
        gram = M @ M.T if M.shape[0] <= M.shape[1] else M.T @ M
        n = gram.shape[0]
        value = float(sla.eigvalsh(gram, subset_by_index=[n-1, n-1], check_finite=True)[0])
    else:
        raise TypeError("No spectral-bound implementation for this operator")
    if not np.isfinite(value) or value <= 0:
        raise ValueError("C1 normalization requires a finite, nonzero operator")
    # shortcut: this float64 safety margin is not an interval certificate; upgrade for formal bounds.
    return value * (1 + 1e-12)


def soft(v, t):
    return np.sign(v) * np.maximum(np.abs(v) - t, 0.0)


# ----------------------------------------------------------------------------------------
class Problem:
    """Base class. Subclasses set: name, a (scalar for A=a*I), B (operator), b, mu, nx, ny, m."""
    has_feas = True

    def finalize_scaling(self, rng):
        import time
        t0 = time.perf_counter()
        self.normb_orig = float(np.linalg.norm(self.b))
        nb2 = norm2_bound(self.B)
        self.normB2_orig = nb2
        self.s = float(np.sqrt(self.mu / (SAFETY * nb2)))
        self.scaling_time = time.perf_counter() - t0
        self.power_its = 0  # kept for result-schema compatibility; power iteration is no longer used
        self.B = self.B.scaled(self.s)
        self.a = self.a * self.s
        self.b = self.b * self.s
        self.normB2 = nb2 * self.s ** 2      # numerical upper bound (= mu/SAFETY)

    # residual r = A x + B y - b (By supplied to avoid recomputation)
    def resid(self, x, By):
        return self.a * x + By - self.b

    def F(self, x, y):
        return self.f(x) + self.g(y)

    def dual_value(self, lam, gstar_grad):
        """d(lam) = -f^*(A^T lam) - g^*(B^T lam) + <lam, b>, evaluated at proj_dual(lam).
        Returns (d, lam_projected)."""
        lp = self.proj_dual(lam)
        w = self.B.rmv(lp)
        ys = gstar_grad(w)
        return -self.fconj(self.a * lp) - self.gconj_val(w, ys) + float(lp @ self.b), lp

    def info(self):
        return dict(name=self.name, inst=self.inst, nx=self.nx, ny=self.ny, m=self.m, mu=self.mu,
                    s=self.s, normB2_orig=self.normB2_orig, normB2_scaled=self.normB2,
                    scaling_time=self.scaling_time, power_its=self.power_its, norm_method="spectral_with_margin",
                    normb_orig=getattr(self, "normb_orig", float("nan")), **self.meta)


# ----------------------------------------------------------------------------------------
class ElasticNetG:
    """g(y) = mu1 ||y||_1 + mu2/2 ||y||^2  (mu = mu2)."""

    def g(self, y):
        return self.mu1 * np.abs(y).sum() + 0.5 * self.mu2 * float(y @ y)

    def make_prox_g(self):
        mu1, mu2 = self.mu1, self.mu2
        return lambda v, t: soft(v, t * mu1) / (1.0 + t * mu2)

    def make_gstar_grad(self, monitor=False):
        mu1, mu2 = self.mu1, self.mu2
        return lambda w: soft(w, mu1) / mu2

    def gconj_val(self, w, ys):
        # g^*(w) = ||S_mu1(w)||^2 / (2 mu2) = <w,ys> - g(ys)
        return float(w @ ys) - self.g(ys)

    def make_admm_ysolver(self, beta):
        return None   # LASSO-type subproblem: no closed form for general B

    def make_prox_g_fixed(self, t, monitor=False):
        pg = self.make_prox_g()
        return lambda v: pg(v, t)


class P1_SVM(ElasticNetG, Problem):
    """P1: elastic-net SVM  (Xu (2017) Sec. 3.3 eq. (3.7); Tang-Toh (2024) Example 2)
        min (1/m) e^T [x]_+ + mu1 ||y||_1 + mu2/2 ||y||^2   s.t.  x + Bdata y = e
    (paper variables: x = hinge slack in R^m with A = I; y = weight vector in R^p with
    B = Diag(labels) * samples, strongly convex g = elastic net)."""
    name = "P1"
    has_exact_y = False

    def __init__(self, m, p, seed, mu1=0.01, mu2=0.01, rho=0.5):
        rng = np.random.default_rng(seed)
        cnt = Counter()
        s = max(1, p // 10)
        u = np.zeros(p); u[:s] = 1.0
        # Sigma = blockdiag(rho*E + (1-rho)*I, I)  (with rho=0.5 identical to Xu's 'rho E + rho I')
        Ls = np.linalg.cholesky(rho * np.ones((s, s)) + (1 - rho) * np.eye(s))
        npos = m // 2
        lab = np.concatenate([np.ones(npos), -np.ones(m - npos)])
        Z = rng.standard_normal((m, p))
        Z[:, :s] = Z[:, :s] @ Ls.T
        X = Z + lab[:, None] * u[None, :]
        Bdata = lab[:, None] * X
        self.cnt = cnt
        self.B = DenseOp(Bdata, cnt)
        self.a = 1.0
        self.b = np.ones(m)
        self.mu1, self.mu2, self.mu = mu1, mu2, mu2
        self.c = 1.0 / m
        self.nx, self.ny, self.m = m, p, m
        self.inst = f"P1_m{m}_p{p}_s{seed}"
        self.meta = dict(seed=seed, mu1=mu1, mu2=mu2, rho=rho, s_feat=s)
        self.finalize_scaling(rng)

    def f(self, x):
        return self.c * np.maximum(x, 0.0).sum()

    def prox_f(self, v, t):
        tc = t * self.c
        return np.where(v > tc, v - tc, np.where(v < 0.0, v, 0.0))

    def fconj(self, u):
        return 0.0   # valid on dom: 0 <= u <= c (guaranteed by proj_dual)

    def proj_dual(self, lam):
        return np.clip(self.a * lam, 0.0, self.c) / self.a

    def x_feas(self, y, By):
        return (self.b - By) / self.a


class P2_SqrtEN(ElasticNetG, Problem):
    """P2: elastic net with square-root loss (Tran-Dinh (2019) Sec. 5.2; Tang-Toh (2024) Example 3)
        min ||x||_2 + mu1 ||y||_1 + mu2/2 ||y||^2   s.t.  -x + Bdata y = c."""
    name = "P2"
    has_exact_y = False

    def __init__(self, n, p, s_sparse, seed, mu1=0.01, mu2=0.1, noise=1e-3):
        rng = np.random.default_rng(seed)
        cnt = Counter()
        Bdata = rng.standard_normal((n, p)) / np.sqrt(n)
        ytrue = np.zeros(p)
        idx = rng.choice(p, s_sparse, replace=False)
        ytrue[idx] = rng.standard_normal(s_sparse)
        c = Bdata @ ytrue + noise * rng.standard_normal(n)
        self.cnt = cnt
        self.B = DenseOp(Bdata, cnt)
        self.a = -1.0
        self.b = c
        self.mu1, self.mu2, self.mu = mu1, mu2, mu2
        self.nx, self.ny, self.m = n, p, n
        self.inst = f"P2_n{n}_p{p}_s{seed}"
        self.meta = dict(seed=seed, mu1=mu1, mu2=mu2, s_sparse=s_sparse, noise=noise)
        self.finalize_scaling(rng)

    def f(self, x):
        return float(np.linalg.norm(x))

    def prox_f(self, v, t):
        nv = np.linalg.norm(v)
        return np.zeros_like(v) if nv <= t else (1.0 - t / nv) * v

    def fconj(self, u):
        return 0.0   # on dom ||u|| <= 1

    def proj_dual(self, lam):
        u = self.a * lam
        nu = np.linalg.norm(u)
        if nu > 1.0:
            u = u / nu
        return u / self.a

    def x_feas(self, y, By):
        return (self.b - By) / self.a


class P3_QP(Problem):
    """P3: dense box-constrained QP (Tran-Dinh (2019) Sec. 5.1; Tang-Toh (2024) Example 4)
        min 1/2 y^T H y + h^T y + delta_[lo,hi](x)   s.t.  -x + Bdata y = 0.
    H = R R^T/... + mu_g I with R in R^{p x (p//2+1)}, so lambda_min(H) = mu_g exactly."""
    name = "P3"
    has_feas = False
    has_exact_y = True

    def __init__(self, p, seed, mu_g=1.0, n=None):
        rng = np.random.default_rng(seed)
        cnt = Counter()
        n = p if n is None else n
        mr = p // 2 + 1
        R = rng.standard_normal((p, mr)) / np.sqrt(mr)
        h = rng.standard_normal(p)
        Bdata = rng.standard_normal((n, p)) / np.sqrt(n)
        H = R @ R.T + mu_g * np.eye(p)
        ynat = rng.standard_normal(p)
        self.feasible_anchor = ynat.copy()
        Byn = Bdata @ ynat
        self.lo = Byn - rng.random(n)
        self.hi = Byn + rng.random(n)
        self.H = H
        self.h = h
        self.cnt = cnt
        self.B = DenseOp(Bdata, cnt)
        self.a = -1.0
        self.b = np.zeros(n)
        self.mu = mu_g
        self.nx, self.ny, self.m = n, p, n
        self.inst = f"P3_p{p}_mu{mu_g:g}_s{seed}"
        self.meta = dict(seed=seed, mu_g=mu_g)
        self.finalize_scaling(rng)

    def f(self, x):
        # indicator of the box; iterates produced by prox are always inside (tolerance 1e-12)
        viol = max(float(np.max(self.lo - x)), float(np.max(x - self.hi)), 0.0)
        return 0.0 if viol <= 1e-9 * (1 + np.abs(self.hi).max()) else np.inf

    def prox_f(self, v, t):
        return np.clip(v, self.lo, self.hi)

    def fconj(self, u):
        return float(np.maximum(self.lo * u, self.hi * u).sum())

    def proj_dual(self, lam):
        return lam

    def x_feas(self, y, By):
        return None

    def g(self, y):
        return 0.5 * float(y @ (self.H @ y)) + float(self.h @ y)

    def make_prox_g(self):
        # (H + I/t) y = v/t - h ; one eigendecomposition, reused for all t
        self.cnt.fact += 1
        ev, V = np.linalg.eigh(self.H)
        h = self.h

        def prox(v, t):
            rhs = V.T @ (v / t - h)
            return V @ (rhs / (ev + 1.0 / t))
        return prox

    def make_prox_g_fixed(self, t, monitor=False):
        # prox with a fixed t: (H + I/t) y = v/t - h ; one Cholesky.
        # monitor=True (monitoring only, untimed): explicit inverse, cheaper per call than cho_solve.
        self.cnt.fact += 1
        h = self.h
        if monitor:
            Ki = np.linalg.inv(self.H + np.eye(self.ny) / t)
            return lambda v: Ki @ (v / t - h)
        cf = sla.cho_factor(self.H + np.eye(self.ny) / t, lower=True, check_finite=False)
        return lambda v: sla.cho_solve(cf, v / t - h, check_finite=False)

    def make_gstar_grad(self, monitor=False):
        # argmin 1/2 y'Hy + h'y - <w,y>  =>  H y = w - h ; one Cholesky of H
        self.cnt.fact += 1
        if monitor:
            Hi = np.linalg.inv(self.H)
            return lambda w: Hi @ (w - self.h)
        cf = sla.cho_factor(self.H, lower=True, check_finite=False)
        return lambda w: sla.cho_solve(cf, w - self.h, check_finite=False)

    def gconj_val(self, w, ys):
        return float(w @ ys) - self.g(ys)

    def make_admm_ysolver(self, beta):
        """(H + beta B^T B) y = B^T lam - h - beta B^T c ; one Cholesky per beta value."""
        self.cnt.fact += 1
        M = self.B.M
        K = self.H + beta * (M.T @ M)
        cf = sla.cho_factor(K, lower=True, check_finite=False)
        Bop, h = self.B, self.h

        def solve(lam, c):
            rhs = Bop.rmv(lam - beta * c) - h
            return sla.cho_solve(cf, rhs, check_finite=False)
        return solve


def phantom(N, rng):
    """Piecewise-constant synthetic image in [0,1] (random rectangles and disks)."""
    img = np.zeros((N, N))
    yy, xx = np.mgrid[0:N, 0:N] / N
    for _ in range(6):
        cx, cy, r = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8), rng.uniform(0.08, 0.25)
        img[(xx - cx) ** 2 + (yy - cy) ** 2 < r ** 2] = rng.uniform(0.2, 1.0)
    for _ in range(4):
        x0, y0 = rng.uniform(0.05, 0.7, size=2)
        w, hgt = rng.uniform(0.1, 0.3, size=2)
        img[(xx > x0) & (xx < x0 + w) & (yy > y0) & (yy < y0 + hgt)] = rng.uniform(0.0, 1.0)
    return img


class P4_TV(Problem):
    """P4: isotropic TV denoising, primal splitting (memo Sec. 4.3.1)
        min alpha ||x||_{2,1} + 1/2 ||y - f0||^2   s.t.  x - grad y = 0
    (x = w in R^{2N^2}, A = I; y = u image, B = -grad, b = 0, mu = 1)."""
    name = "P4"
    has_exact_y = True

    def __init__(self, N, seed, alpha=0.1, sigma=0.1):
        rng = np.random.default_rng(seed)
        cnt = Counter()
        clean = phantom(N, rng)
        f0 = clean + sigma * rng.standard_normal((N, N))
        self.N = N
        self.f0 = f0.ravel()
        self.alpha = alpha
        self.cnt = cnt
        self.B = GradOp(N, cnt, c=-1.0)
        self.a = 1.0
        self.b = np.zeros(2 * N * N)
        self.mu = 1.0
        self.nx, self.ny, self.m = 2 * N * N, N * N, 2 * N * N
        self.inst = f"P4_N{N}_s{seed}"
        self.meta = dict(seed=seed, alpha=alpha, sigma=sigma)
        self.finalize_scaling(rng)

    def _pnorm(self, x):
        NN = self.N * self.N
        return np.sqrt(x[:NN] ** 2 + x[NN:] ** 2)

    def f(self, x):
        return self.alpha * float(self._pnorm(x).sum())

    def prox_f(self, v, t):
        nv = self._pnorm(v)
        fac = np.maximum(1.0 - t * self.alpha / np.maximum(nv, 1e-300), 0.0)
        return v * np.concatenate([fac, fac])

    def fconj(self, u):
        return 0.0   # on dom: pointwise ||u_ij|| <= alpha

    def proj_dual(self, lam):
        u = self.a * lam
        nu = self._pnorm(u)
        fac = 1.0 / np.maximum(nu / self.alpha, 1.0)
        return u * np.concatenate([fac, fac]) / self.a

    def x_feas(self, y, By):
        return (self.b - By) / self.a

    def g(self, y):
        d = y - self.f0
        return 0.5 * float(d @ d)

    def make_prox_g(self):
        f0 = self.f0
        return lambda v, t: (v / t + f0) / (1.0 + 1.0 / t)

    def make_prox_g_fixed(self, t, monitor=False):
        pg = self.make_prox_g()
        return lambda v: pg(v, t)

    def make_gstar_grad(self, monitor=False):
        f0 = self.f0
        return lambda w: f0 + w

    def gconj_val(self, w, ys):
        return float(w @ ys) - self.g(ys)

    def make_admm_ysolver(self, beta):
        """(I + beta B^T B) y = f0 + B^T(lam - beta c), B^T B = c^2 grad^T grad: DCT-II diagonalizes."""
        N, Bop, f0 = self.N, self.B, self.f0
        den = 1.0 + beta * Bop.c ** 2 * Bop.eig_lap

        def solve(lam, c):
            rhs = (f0 + Bop.rmv(lam - beta * c)).reshape(N, N)
            Y = sfft.idctn(sfft.dctn(rhs, type=2, norm="ortho") / den, type=2, norm="ortho")
            return Y.ravel()
        return solve


def make_problem(spec):
    """spec: dict with key 'P' and size/seed parameters (see tasks.py)."""
    P = spec["P"]
    if P == "P1":
        return P1_SVM(spec["m"], spec["p"], spec["seed"])
    if P == "P2":
        return P2_SqrtEN(spec["n"], spec["p"], spec["s"], spec["seed"])
    if P == "P3":
        return P3_QP(spec["p"], spec["seed"], mu_g=spec["mu"])
    if P == "P4":
        return P4_TV(spec["N"], spec["seed"])
    raise ValueError(P)
