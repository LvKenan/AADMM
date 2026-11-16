# CHANGELOG

## C1-1.2 (2026-11-15 JST)

Full rerun package (decisions R5-D1/D2/D3). Results from C1-1.0 or C1-1.1-audit must not be mixed with C1-1.2;
`run` refuses existing result files whose `version` or `package_sha256` differs from this package.

Experiment design
- Main group gets three extra variants on all 70 main instances, with unique task ids, labels and theory scopes:
  - `admm` beta=10 (`main__<inst>__admm__beta10`) and beta=100 (`main__<inst>__admm__beta100`):
    tuned baselines, **no O(1/k^2) guarantee**;
  - `aadmm` beta=1.5 (`main__<inst>__aadmm__beta1.5`): fixed beta in (0,2), covered by the O(1/k^2) dual-gap theorem.
- P4 reference time budget 300 s -> 2400 s (`reference.time_budget_by_problem.P4`; a cap only, converged references stop at tol; hardest box test P4_N192_s3 needed 1092 s); others stay 300 s.
  Tolerance 1e-12 and strict reference checking unchanged. Method maxit stays 20000, wall cap 120 s.
- Task count: 70 references + 670 main + 198 sweep = 938 (`RUN_LIST.json`).

New recorded quantities (C1-1.0 did not store these)
- per-iteration KKT, penalty and (AADMM) L(predictor) arrays, so first-hit levels and theorem ratios are exact;
- IADMM/Heu-ADMM: S_k, S_k/k^2, penalty first/last/min/max, penalty_last/k, number of penalty changes;
- Goldstein restart: restart iterations, restart count and fraction;
- actual penalty used for every method; run-level `summary` in each result's meta.

Runner
- checkpoint every 25 tasks or 15 min plus at end: `C1_partial_v12_latest.zip` next to the output directory;
- `run --group ref|nonref|main|sweep|all`, so references can run in a second process;
- `pack` writes `C1_results_v12_<YYYYMMDD_HHMM>.zip` with results, logs, env, manifest, run summary,
  provenance (config, run list, changelog) and the **strict** figures/tables (`figs/`). If strict plotting
  fails, no relaxed figures are made; `pack` exits with code 2 and the failure is in `logs/plot_log.txt`.
- log timestamps carry an explicit +09:00 offset.

Plotting
- curves/tables keyed by method variant with labels and theory scope (`theory_scope.csv`, `table_main.md`);
- AADMM theorem-ratio check uses each run's beta (`aadmm_theorem_ratio.csv`);
- reference acceptance also recomputes the gap from UB/LB.

## C1-1.1-audit

Audit patch on C1-1.0: normalization checks, P3 reference repair toward the strictly feasible anchor,
IADMM floor clamping with `method_settings` (beta1_requested/used/floor), guarantee-scope table in README.

## C1-1.0

First package run by Carbon (159 min on 8-core EPYC, single-thread BLAS). Did not store IADMM/Heu S_k,
restart flags, penalty traces or per-iteration first hits; six P4 references stopped at the 300 s limit.
