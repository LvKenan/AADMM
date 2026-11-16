# CHANGELOG

## C2-supp-1.0 (2026-11-15 JST)

First version. Supplementary experiments for claims of the revised manuscript / response letter that had no
experimental support yet. Code of C1-1.2 is copied **unchanged** into `c1/` (problem generators, C1 monitor, AADMM);
new code is in `c2/`, `run_c2.py`, `plot_c2.py`, `toy_fig.py`, `tests/sanity_c2.py`.

Groups (846 tasks, `RUN_LIST.json`)
- `ref` (7): LASSO references for (b) (primal FISTA + restart, primal-dual certificate <= 1e-13). The (a) and TVD (b)
  references are the C1-1.2 references (`refs/c1v12_refs.npz`, sha256 checked, instance fingerprints checked).
- `a_struct` (36): Remark rem:dualfista (i)-(iii): AADMM-beta with the compatible initialization vs FISTA-beta on the dual,
  K = 300 iterations, beta in {0.5, 1, 1.5}; P1 m100, P3 mu=1, P3 mu=0.1, P4 N64; seeds 0-2.
- `a_range` (504): AADMM-beta vs dual FISTA vs dual proximal gradient, beta in {0.5, 1, 1.25, 1.5, 1.75, 1.9, 2.5, 4},
  7 C1 instance classes x 3 seeds, stop at KKT <= 1e-8 or 5000 iterations; divergence guard ||lam|| > 1e8 (1+||lam*||).
- `b` (299): inexact x-subproblems on LASSO (200x2000, 500x5000: seeds 0-2; 1000x10000: seed 0) and TV denoising
  (N = 64, 128; seeds 0-2): AADMM beta in {1, 1.5} with 9 inner-accuracy rules, ADMM beta=1 with 5 rules;
  500 outer iterations or relative dual gap <= 1e-10.
- `toy_fig.py` (no task): Fig. F_E1 with the corrected updates of Sec. 4.1 and a non-degenerate variant (f = x^2/2).

Caps: method wall time 3600 s per task (no box run came near it: longest 117 s); LASSO reference 1800 s.
