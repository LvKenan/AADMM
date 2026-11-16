# C2_supp: supplementary experiments for the JSC AADMM revision

Package for the major revision of *Fixed-Penalty Accelerated ADMM with an O(1/k^2) Dual Gap for
Semi-Strongly Convex Separable Optimization*. Python 3 with **numpy and scipy only** (matplotlib
only for `plot_c2.py` and `toy_fig.py`), float64, fixed seeds, **single-threaded BLAS** (set in
`run_c2.py` before numpy is imported).

This package does **not** rerun the C1-1.2 comparison of Section 4.5. The official C1-1.2 results
already contain algorithm time and objective values; those were tabulated separately (not in this
zip). C2_supp covers the remaining supplementary experiments:

- **(a)** C7(iii) / Remark rem:dualfista: AADMM-β versus FISTA-β (and proximal gradient) on the dual,
  including steps β>1. Structural identities and a β-range study.
- **(b)** C5: sensitivity of AADMM / ADMM to inexact *x*-subproblems on the paper's own LASSO and TV
  denoising splittings (inner ISTA or Chambolle, tolerance schedules and fixed iteration counts).
- **(c)** L2: regenerate the toy Figure F_E1 of Section 4.1 with the corrected updates, plus a
  non-degenerate variant *f(x)=x²/2* (same solution, β need not be 1). Deterministic; no data.

`c1/` is a copy of the C1-1.2 generators (identical code; in this public release only comments and descriptive strings were updated to the journal numbering of Tang–Toh, see `../CODE_CHANGES_vs_run_packages.diff`), operators, Monitor and AADMM. New code is
in `c2/`, `run_c2.py`, `plot_c2.py`, `toy_fig.py` and `tests/sanity_c2.py`. Do not edit code or
`c1/config.json` / `c2/config.json`. Do not mix results with a different package hash: `run` refuses
result files from another version or code hash.

## C2-supp-1.0 at a glance (see CHANGELOG.md)

846 tasks (`RUN_LIST.json`):

| group | n | what |
|---|---|---|
| `ref` | 7 | LASSO references for (b): primal FISTA + restart, primal-dual certificate ≤ 1e-13. (a) and TVD (b) use the shipped C1-1.2 references in `refs/c1v12_refs.npz` (sha256 checked; instance fingerprints checked). |
| `a_struct` | 36 | AADMM-β with the compatible initialization vs FISTA-β on the dual, K=300, β∈{0.5,1,1.5}; P1 m=100, P3 μ∈{1,0.1}, P4 N=64; seeds 0–2. |
| `a_range` | 504 | AADMM-β vs dual FISTA vs dual proximal gradient, β∈{0.5,1,1.25,1.5,1.75,1.9,2.5,4}, 7 C1 instance classes × 3 seeds; stop at KKT ≤ 1e-8 or 5000 iterations; divergence guard ‖λ‖ > 1e8(1+‖λ*‖). |
| `b` | 299 | Inexact *x*-subproblems on LASSO (200×2000 and 500×5000, seeds 0–2; 1000×10000, seed 0) and TV denoising (N=64,128, seeds 0–2): AADMM β∈{1,1.5} with 9 inner-accuracy rules, ADMM β=1 with 5 rules; 500 outer iterations or relative dual gap ≤ 1e-10. |

Per-task wall-time caps: 3600 s for method tasks, 1800 s for a LASSO reference (a cap only; a reference that meets the tolerance stops earlier).

## How to run (Colab CPU runtime or any Linux machine)

```bash
cd revision_experiments/C2_supplementary
python run_c2.py env                       # environment record
python run_c2.py smoke                     # < 3 min; must end with "SMOKE PASS"
python run_c2.py run --group all --out OUT # all 846 tasks, sequential (timing); refs first
python run_c2.py status --out OUT
python run_c2.py checkpoint --out OUT      # optional: write C2_partial_latest.zip now
python run_c2.py pack --out OUT            # -> C2_results_<YYYYMMDD_HHMM>.zip, strict figures included
```

Groups may also be run separately, in this order: `ref`, then `a_struct` / `a_range` (independent of
the new LASSO refs) and `b` (LASSO tasks in `b` need the `ref` group to have finished). `run --group
all` already puts references first.

Resume: rerun the same `run` command; finished tasks (`OUT/results/<task_id>.npz`, written atomically)
are skipped. Failures are in `OUT/logs/FAILED__<task>.txt` and are retried on the next run; report
them, do not patch code. `run` writes a checkpoint zip every 25 tasks or 15 minutes, next to the
output directory (`C2_partial_latest.zip`). `pack` exits 0 only if every task is present and strict
plotting succeeds; it still writes the zip for diagnosis if not (exit code 2). `C2_supp_colab.ipynb`
contains the Colab cells. Run a single `run` process against a given output directory.

## Estimated runtime (Colab CPU, Intel Xeon 2.20 GHz class, single-thread BLAS)

Probes on a faster box, scaled to the C1-1.2 Colab machine, plus the 3600 s cap:

- `smoke`: under 3 minutes. Must print `SMOKE PASS`.
- `ref` (7 LASSO references): about 1 minute. The 1000×10000 reference reached a 1e-13 gap in a few seconds in the probe; the 1800 s cap is only an upper bound.
- `a_struct` (36): a few minutes.
- `a_range` (504): typically 20–45 minutes (5000-iteration cap; P4 N=128 and the P1 class, which often run to the iteration cap, dominate). Dual FISTA with β>1 may run all 5000 iterations without hitting the divergence guard (AADMM with β=4 did diverge in the smoke test).
- `b` (299): this is the long group. Fixed inner-iteration counts, and AADMM with β=1 on the B=I LASSO splitting (the exact inner solve is optimal after one outer step), are short. The expensive tasks are AADMM with β=1.5 and the tight inner-tolerance schedules (`c=1e-11` and, on the larger instances, `c=1e-7`), especially LASSO 500×5000 / 1000×10000 and TV denoising with N=128. A few of the largest tight-tolerance tasks may run to the 3600 s cap.

**Measured in the official run: about 80 min on a Colab CPU runtime.** Planning estimate before the run: **typically about 6–10 hours; worst case about 12–16 hours** if several large tight-tolerance tasks reach the 3600 s cap. The run is resumable after a disconnect. If a single task produces no log line for more than 70 minutes, report it (the cap is 3600 s plus a modest margin for monitoring).

## Contents

| file | role |
|---|---|
| `run_c2.py` | driver: `env`, `list`, `smoke`, `run`, `status`, `checkpoint`, `pack` |
| `RUN_LIST.json` | the 846 task ids with labels and theory scope |
| `CHANGELOG.md` | C2-supp-1.0 |
| `C2_supp_colab.ipynb` | Colab cells |
| `c2/config.json` | instances, seeds, β grids, inner-accuracy rules, caps (fixed) |
| `c2/methods.py` | AADMM-β (optional compatible init), dual FISTA, dual proximal gradient, inexact AADMM / ADMM |
| `c2/problems.py` | LASSO and TVD inner problems (paper splittings), inner certificates, LASSO reference |
| `c2/refs.py` | shipped C1-1.2 references and fingerprint check; TVD duality identity |
| `c2/tasks.py` | task list |
| `c1/` | unmodified C1-1.2 copy (generators, AADMM, Monitor) |
| `refs/c1v12_refs.npz` | C1-1.2 references used by (a) and by TVD in (b) |
| `tests/sanity_c2.py` | implementation checks (run inside `smoke`) |
| `plot_c2.py` | figures and tables from results |
| `toy_fig.py` | regenerated Fig. F_E1 and the *f(x)=x²/2* variant |

## Notes on what is and is not claimed

* With β=1, an affine ∇Ψ and the compatible initialization, the AADMM-1 predictor multipliers coincide with dual FISTA (Remark rem:dualfista (i)); this is checked in `tests/sanity_c2.py` on small P3 and P4 instances. For β≠1 or a nonaffine ∇Ψ the two schemes differ; that is a fact to be measured, not a claim that dual FISTA diverges.
* AADMM-β has an *O(1/k²)* dual-gap guarantee only for β∈(0,2) on the normalized problem, and only for exact subproblems. Tasks with β≥2, and all inexact runs in (b), are outside that guarantee; the theory scope is stored with each task. Divergence is recorded only when the guard is triggered or a residual is non-finite.
* The LASSO and TVD problems in (b) use the paper's *B=I*, *g=½‖·‖²* splittings, so that the inexactness is in the *x*-subproblem (ISTA / Chambolle), matching Sections 4.2–4.3. The strongly convex *y*-subproblem remains closed form. The C1-1.2 problems of Section 4.5 are not reused here as inexact-subproblem tests; those already have exact (or once-factorized) subproblems.
* The LASSO regularization parameter for the sizes in (b) is τ=0.1‖Aᵀb‖_∞. The paper does not state τ for these sizes; this choice is recorded in `c2/problems.py`.

## Pack output

`pack` writes `C2_results_<YYYYMMDD_HHMM>.zip` (timestamp in Japan time) containing `results/`, `logs/`,
`env.txt`, `manifest.json`, `run_summary.md` / `.json`, `provenance/`, and, if strict plotting
succeeds, `figs/` (including the toy figure). Exit code 0 means the run is complete and the figures
passed; exit code 2 means missing tasks or a strict-plot failure — return the zip anyway and report it.
