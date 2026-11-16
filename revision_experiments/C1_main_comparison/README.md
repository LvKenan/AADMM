# C1: AADMM vs. representative (accelerated / adaptive) ADMMs, same settings

Package for the JSC major revision of *Fixed-Penalty Accelerated ADMM with an O(1/k^2) Dual Gap for
Semi-Strongly Convex Separable Optimization* (Section 4.5 of the revised paper).
 Python 3 with **numpy and scipy only** (matplotlib only for `plot_c1.py`),
float64, fixed seeds, **single-threaded BLAS** (set in `run_c1.py` before numpy is imported).

## C1-1.2 at a glance (see CHANGELOG.md)

938 tasks (`RUN_LIST.json`): 70 references, 670 main (7 methods with default penalties + `admm` beta=10, beta=100
+ `aadmm` beta=1.5 on 70 instances; Goldstein only on P3/P4), 198 sweep. P4 reference budget 2400 s (a cap only; converged references stop at tol).
Do not mix results with C1-1.0 / C1-1.1-audit: `run` stops on result files from another version or code hash.

## How to run (Colab CPU runtime or any Linux machine)

```bash
cd revision_experiments/C1_main_comparison
python run_c1.py env                       # environment record
python run_c1.py smoke                     # < 1 min; must end with "SMOKE PASS"
python run_c1.py run --group ref    --out OUT > ref.log 2>&1 &   # references (second process)
python run_c1.py run --group nonref --out OUT                    # main + sweep, sequential (timing)
python run_c1.py status --out OUT
python run_c1.py checkpoint --out OUT      # optional: write C1_partial_v12_latest.zip now
python run_c1.py pack --out OUT            # -> C1_results_v12_<YYYYMMDD_HHMM>.zip, strict figures included
```

Resume: rerun the same `run` commands; finished tasks (`OUT/results/<task_id>.npz`, written atomically) are skipped.
Failures are in `OUT/logs/FAILED__<task>.txt` and are retried on the next run; report them, do not patch code.
`run` writes a checkpoint zip every 25 tasks or 15 minutes. `pack` exits non-zero if tasks are missing or
strict plotting fails (it still writes the zip for diagnosis). `C1_v12_colab.ipynb` contains the Colab cells.
Do not edit code or `c1/config.json`.

## Contents

| file | role |
|---|---|
| `run_c1.py` | driver: `env`, `list`, `smoke`, `run`, `status`, `checkpoint`, `pack` |
| `RUN_LIST.json` | the 938 task ids with labels and theory scope |
| `CHANGELOG.md` | 1.0 → 1.1-audit → 1.2 |
| `C1_v12_colab.ipynb` | Colab cells |
| `c1/config.json` | instances, seeds, tolerances, iteration and time caps (fixed) |
| `c1/problems.py` | data generators and closed-form subproblem pieces for P1–P4; constraint scaling (paper eq. (5)) |
| `c1/methods.py` | AADMM-β (Alg. 2) and the comparison methods |
| `c1/metrics.py` | per-iteration metrics, stopping rule, logging |
| `c1/reference.py` | independent high-accuracy reference solve for F* (with certificate) |
| `c1/tasks.py` | task list (reference solves + method runs) |
| `tests/sanity.py` | implementation checks (run inside `smoke`) |
| `plot_c1.py` | figures and tables from results |

## Test problems (Section 4.5; all non-degenerate, all subproblems closed form or one factorization)

Paper form: `min f(x)+g(y)  s.t.  Ax+By=b`, `g` μ-strongly convex, `A = a·I`.

| id | problem | f, A | g (strongly convex), B | data / sizes (5 seeds each) |
|---|---|---|---|---|
| P1 | elastic-net SVM (Xu (2017) §3.3, Tang–Toh (2024) Example 2) | (1/m)eᵀ[x]₊, A=I | μ₁‖y‖₁+μ₂/2‖y‖², B=Diag(labels)·samples | generator of Xu (2017) §3.3, μ₁=μ₂=0.01, ρ=0.5; (m,p) ∈ {(100,500),(300,1000),(1000,3000)} |
| P2 | square-root-loss elastic net (Tran-Dinh (2019) §5.2, Tang–Toh (2024) Example 3) | ‖x‖₂, A=−I | elastic net, μ₁=0.01, μ₂=0.1, B Gaussian/√n | generator of Tran-Dinh (2019) §5.2 (σ=1e-3); (n,p,s) ∈ {(350,1000,100),(700,2000,200),(1750,5000,500)} |
| P3 | dense box QP (Tran-Dinh (2019) §5.1, Tang–Toh (2024) Example 4) | δ_[lo,hi], A=−I | ½yᵀHy+hᵀy, λ_min(H)=μ_g exactly, B Gaussian/√n | generator of Tran-Dinh (2019) §5.1; p=n ∈ {500,1000,2000}, μ_g ∈ {1, 0.1} (2000: μ_g=1 only) |
| P4 | isotropic TV denoising, primal splitting | α‖·‖₂,₁, A=I | ½‖y−f₀‖², B=−∇ (Neumann) | synthetic piecewise-constant image + N(0,0.1²) noise, α=0.1; N ∈ {64,128,192} |

All methods run on the same normalized problem, A,B,b ← s·(A,B,b).
The patch computes the largest eigenvalue of the smaller dense Gram matrix, or uses the known
DCT spectrum for P4, adds a numerical margin, and sets s=√(μ/(1.001·norm_bound)).
This replaces the unverified assertion that a power estimate times 1.001 is always an upper bound.
The calculation is float64, not an interval certificate; preprocessing time is recorded.

Objective/dual values and the recorded KKT residuals are invariant under the paired transformation
λ_scaled=λ_original/s. Standard ADMM also has the usual penalty transformation β_scaled=β_original/s²
when its proximal terms are transformed consistently. No equivalence of the complete AADMM
prediction–correction trajectory before and after scaling is claimed.

## Methods and penalty choices

| name | method | penalty |
|---|---|---|
| `aadmm` | AADMM-β, sn-article.tex **Algorithm 2**, eqs. (17a)–(17c), (18), (19a)–(19c); recorded iterate = predictor (x̃ᵏ,ỹᵏ,λ̃ᵏ) | β = 1 (paper default); main also β = 1.5 (separate variant); sweep β ∈ {0.25, 0.5, 1.5, 1.9} |
| `admm` | classical ADMM, eq. (4) | β = 1 (scaled problem); main also β = 10 and β = 100 (separate variants); sweep β ∈ {0.01, 0.1, 10, 100}. y-step exact for P3 (Cholesky of H+βBᵀB, one factorization) and P4 (DCT); **linearized** for P1, P2 (proximal ADMM, Q=β(μI−BᵀB)), because the exact y-step is a LASSO with no closed form |
| `xu_fixed` | Xu (2017) Alg. 2, journal Theorem 2.13 / (2.26), as "Nonaccelerated Linearized ADMM" in Xu (2017) §3.3 | β=γ=1/(2μ), P=0, Q=I/2−BᵀB/(2μ) |
| `xu_acc` | Xu (2017) Alg. 2, journal Theorem 2.14 / (2.27), as "Accelerated Linearized ADMM" in Xu (2017) §3.3 | β_k=γ_k=(k+1)γ, Q^k=(k+1)(q₀I−γBᵀB), q₀=μ/20, γ=q₀/μ, P=0 |
| `iadmm` | Tang–Toh (2024, MPC) Alg. 1 (IADMM) + strategy (35) | γ=1.618, ε=1e-4, requested β₁=1/(2μ); sweep × {1e-3, 1e-1, 1e1, 1e3}; IADMM clamps to its floor |
| `heu_admm` | Tang–Toh (2024, MPC) Heu-ADMM, rule (40) | same as `iadmm` |
| `fast_admm_r` | Goldstein et al. (SIIMS 2014) fast ADMM with restart (Alg. 8), η=0.999 | τ=1 (scaled problem); only P3, P4 (needs exact y-step) |

Ergodic (averaged) iterates: AADMM t_k-weighted (Corollary 8 of the revised paper), `xu_acc` weights (k+k₀+1), k₀=1 (Xu journal Thm 2.14),
`iadmm` β_k-weighted (Tang–Toh, journal Theorem 1); the same weights for
`heu_admm` are only descriptive, since rule (40) need not satisfy Algorithm 1's interval.
The residual-balanced `iadmm` has an O(1/S_k) ergodic guarantee, not an automatic O(1/k²) guarantee.
`weight_sum` records S_k for these two methods; examine β_k/k and S_k/k² rather than inferring
quadratic cumulative growth from the method name. Taking the admissible upper endpoint at every
step defines a separate theoretical schedule; this patch does not silently replace the balancing rule.
Other averages are descriptive unless supported by the corresponding theorem.

### Guarantee scope of each method as run here (as stated in Section 4.5)

No "theory-version" variants are added (no Xu run with P≻0, η>1; no Tang–Toh run that always takes the
upper endpoint). Each curve may be described only with the guarantee listed below; anything else observed
(e.g. fast last-iterate decay) is an empirical observation, not a theorem.

| method | guarantee that applies to this exact configuration | what must NOT be claimed |
|---|---|---|
| `aadmm` (β=1, and main variant β=1.5) | fixed β∈(0,2) on the normalized problem: dual gap v_k=O(1/k²) at the predictor (Theorem 3), O(1/k²) objective error/feasibility of the t_k-weighted average (Corollary 8), O(1/k) last-predictor bounds (Proposition 9) | oracle-complexity optimality; scale invariance of the trajectory |
| `admm` (β=1, and main variants β=10, β=100) | fixed-penalty (proximal/linearized on P1, P2; exact on P3, P4) ADMM: standard non-accelerated O(1/k) ergodic theory only; β=10/100 are tuned baselines | any accelerated rate; no O(1/k²) guarantee |
| `xu_fixed` | Xu (2017) Theorem 2.13: O(1/k) for the uniform average | accelerated rate |
| `xu_acc` | Xu (2017) Theorem 2.14, (2.30a)–(2.30b): O(1/k²) objective error and feasibility of the (k+k₀+1)-weighted average | boundedness or last-iterate bounds (2.32a)–(2.32b): these need P≻0 and η>1, but P=0 here |
| `iadmm` | Tang–Toh journal Theorem 1 / Corollary 1: O(1/S_k) for the β_k-weighted average (hence ≥ O(1/k)); whole-sequence convergence (journal Theorem 2) since β_k ≥ floor > 0 and A=aI | O(1/k²): residual balancing does not guarantee S_k=Ω(k²); report the measured S_k/k² only as a diagnostic |
| `heu_admm` | none from Tang–Toh: rule (40) can leave Algorithm 1's interval and has no floor | any theorem of Tang–Toh; its β_k-weighted average is descriptive |
| `fast_admm_r` | Goldstein et al. (2014) Theorem 3: combined residual c_k → 0 | the O(1/k²) of their Theorem 2 (needs no restart, both blocks strongly convex, G quadratic, small τ) |

### Reporting the penalty actually used

IADMM starts from beta1_used = max(beta1_requested, floor), floor = 1e-6/s² (Algorithm 1 requires β₁ ≥ floor).
Each result file stores `method_settings` = {beta1_requested, beta1_used, beta_floor, penalty_policy}
(Heu-ADMM: beta1_used = beta1_requested, beta_floor = None, no clamping), and the per-iteration
`penalty` and cumulative `weight_sum` (= S_k) columns. `plot_c1.py` writes the used β₁ to the
`beta1_used` column of `table_sweep.csv`. In the paper, sweep points must be labelled by the β₁ actually
used (e.g. on P1 the requested 1e-3 multiplier is raised to the floor), and the β_k range and S_k/k² of the
default runs should be reported.

## Metrics (every iteration; logged at every iteration up to k=100, then about 50 points per factor e)

R_p and R_d as in Tang–Toh (2024) §5.2 (sign of λ flipped to the paper's convention), KKT = max(R_p, R_d);
F = f(x)+g(y); F_feas = F(x_feas(y), y) for P1, P2, P4 (x_feas solves Ax+By=b); Lagrange dual value d(Π(λ))
(a lower bound on F\*); `rp_abs` = ‖A₀x+B₀y−b₀‖; the same for the ergodic iterate; for AADMM also
L(x̃,ỹ,λ̃) (equals d(λ̃), Proposition 4 of the revised paper) and a post-processed point used only as a diagnostic (‖Ĝ‖ and its KKT);
the penalty in use; the counts of products with B or Bᵀ and of matrix factorizations; algorithm time `t_alg`.
`t_alg` excludes monitoring. The 120 s cap is wall time including monitoring, so it can leave
different amounts of algorithm time for different methods; report both times and do not interpret
this as an equal algorithm-time cap. AADMM postprocessing certificates require extra untimed work. The residuals that IADMM/Heu-ADMM need for their penalty rules *are* timed,
since they are part of those methods.
References: tolerance 1e-12, time budget 300 s (P4: 2400 s). Stopping: KKT ≤ 1e-8, or 20000 iterations, or 120 s wall per run. Thresholds 1e-4 … 1e-8 are read off afterwards.
F\*: `c1/reference.py`, restarted FISTA on the Lagrange dual (separate code path), certified by a primal
feasible upper bound and dual lower bound (relative gap target 1e-12). For P3, the reference solver
repairs the dual-recovered point along the segment to the generator's strictly feasible anchor.
Fstar stores the best feasible upper estimate, not a mathematically exact optimum. Plotting rejects
references that did not reach the gap tolerance. The bounds are floating-point estimates.

## Implementation uncertainties (please check before using numbers in the paper)

1. **Tang–Toh Q matrix.** Alg. 1 leaves Q ⪰ 0 free, and the paper does not say which Q was used in Examples 2–4. We use
   Q=0 (exact y-step) for P3/P4 and Q=μI−CᵀC (linearized, as suggested in Tang–Toh (2024) §5.1) for P1/P2. For P3 with Q=0,
   every change of β_k triggers a new Cholesky factorization (counted). The sweep group also runs P3 with the
   linearized variant (`tt_q=lin`).
2. **Tang–Toh (2024) Heu-ADMM** step length γ and Q are not stated separately; we use the IADMM values (γ=1.618, same Q).
3. **Tang–Toh lower bound β = 1e-6** maps to 1e-6/s² on the normalized constraint.
   IADMM now uses max(requested β₁, floor); requested/used β₁ and the floor are saved in metadata.
   The Heu-ADMM rule has no such floor and is not clamped. Sweep labels denote requested multipliers.
   Algorithm 1's floor < 1 convention applies in original coordinates: beta_0=s² beta_s,
   lambda_0=s lambda_s, Q_0=Q_s/s². The original floor remains 1e-6 even if the scaled floor exceeds 1.
4. **Tang–Toh (2024) R_p, R_d** have denominators that can vanish; we guard them with 1e-300 (never active in tests).
5. **Tang–Toh (2024) β₁** in Examples 1–3 is swept over 1e-5…1e5; we use 1/(2μ) as a common normalized baseline plus a sweep. Their Example 4 uses
   β₁=floor=1/(2‖B‖²); combining our baseline with the general floor 1e-6/s² is a C1 choice.
6. **Xu source**: checked against the SIOPT journal version: Algorithm 2 (1.15), Theorems 2.13–2.14,
   (2.26)–(2.32), and §3.3 p.1479. The old references to (78)/(79) did not identify equations in the
   supplied arXiv v1. Use smooth f_Xu=0 and put the complete strongly convex block in g_Xu;
   q₀=μ/20, γ=q₀/μ, P=0. This satisfies the ergodic theorem, but P=0 excludes its (2.32)
   last-iterate guarantee. μ is a conservative upper bound for ‖B‖².
7. **Xu counter offset**: journal (2.27) is stated for k ≥ 1 with initial point (y¹, z¹, λ¹); we use β_k=(k+1)γ at the k-th iteration.
8. **Goldstein et al.** penalty τ=1 is an empirical common baseline. The SIIMS journal Algorithm 8
   (p.1603) is implemented literally, with c₀=∞ to accept the first trial. Every trial is counted;
   `restart` marks rejected trials. Theorem 3 proves c_k→0, not an O(1/k²) rate for this restart
   variant on P3/P4. Its uniform trial average is descriptive.
9. **Classical ADMM on P1/P2** is the linearized (proximal) variant, so with these penalties it belongs to the same
   family as `xu_fixed` (same update, different β).
10. **Data generators**: P1 uses Σ = blockdiag(ρE+(1−ρ)I, I); Xu (2017) prints "ρE+ρI", which is the same matrix for ρ=0.5.
   P2 follows Tran-Dinh arXiv v3 §5.2; P3 follows §5.1 (with μ_g=0.1 also tested). P4 uses a synthetic image (no image file needed).
11. Tang–Toh formulas are checked against the journal version (Math. Program. Comput. 16, 113–150, 2024):
    Algorithm 1, rule (35) (IADMM residual balancing, p. 128) and rule (40) (Heu-ADMM, p. 131), Theorem 1,
    Corollary 1 and Theorem 2. These rules are word-for-word identical to rules (34)/(39) of arXiv:2310.00376v1,
    which the code was originally checked against; only the numbering differs.

## Notes on Algorithm 2 found while implementing (also reported in tests/sanity.py)

* Alg. 1 (eqs. (13)–(15)) and Alg. 2 with β=1 give identical iterates (S4; max difference 5e-13).
* The structural degeneracy of the LASSO/TV splittings of Secs. 4.2–4.4 (Remark 11(i) of the revised paper) is reproduced (S3): with g=½‖y‖², B=I, β=1 and no rescaling,
  λᵏ−zᵏ=0 and Ax̃ᵏ+Bỹᵏ−b=0 to machine precision. With β=0.5 or BᵀB≠μI this does not happen.
* For B=0, S5 now asserts agreement with the revised manuscript's formal limit in s=λ−z
  (main source lines 553–572), and disagreement with He–Yuan (2010), (3.4b). The latter was
  the old manuscript's incorrectly identified equation (16), not the revised limit.
* S6 checks spectral normalization and nondegeneracy on small P1–P4 instances, a known-optimum
  P3 reference bracket, the IADMM initial floor, and cumulative β weights.

## Audit patch usage

Apply this patch before collecting new results, using a fresh output directory. Do not merge C1-1.0
results with C1-1.1-audit results: normalization, P3 reference values and the lowest IADMM sweep
initializations can change. `config.json` and experiment sizes are unchanged.
Run `python run_c1.py smoke` after applying the patch. A smoke pass means the code executed and
sanity checks passed; it does not mean every method reached 1e-8 within the small-run cap.
