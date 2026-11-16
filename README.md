# Reproducibility package: revision experiments

This folder belongs to the revised version of

> K. Lv, F. Guan, B. He, J. Yuan, *Fixed-Penalty Accelerated ADMM with an O(1/k²) Dual Gap for
> Semi-Strongly Convex Separable Optimization*, submitted to the Journal of Scientific Computing (major revision).

It contains the code, run lists, summary tables and analysis scripts for the numerical experiments added
in the revision:

| Paper item | Folder |
|---|---|
| Sec. 4.5 (comparison with accelerated and adaptive ADMM methods): Table 9, Tables 10–11, Figs. 10–12; Appendix A (Figs. A1–A2) | `revision_experiments/C1_main_comparison/` |
| Appendix B.1 (AADMM-β vs. FISTA / proximal gradient on the dual): Table B1, Figs. B1–B2 | `revision_experiments/C2_supplementary/` (group `a_*`) |
| Appendix B.2 (inexact x-subproblems): Table B2, Fig. B3 | `revision_experiments/C2_supplementary/` (group `b`) |
| Sec. 4.1, Fig. 1 (toy example, regenerated) | `revision_experiments/C2_supplementary/toy_fig.py` |

The code of the original submission (Secs. 4.2–4.4) is not changed by this package.

## Layout

```
revision_experiments/
  C1_main_comparison/          code that produced the results of Sec. 4.5 and Appendix A (package C1-1.2)
    run_c1.py, c1/, tests/, plot_c1.py, RUN_LIST.json, README.md, CHANGELOG.md
    results_summary/           summary of the official run (no raw trajectories)
      per_run_summary.csv      one row per task: stop reason, iterations, algorithm time, final KKT, first hits
      C1_full_14config_table.md the full 14-configuration table referred to in Sec. 4.5
      C1_full_run_table.json    one record per main run (first hits, dual-gap hits, theorem ratio, S_k, restarts)
      table10_11_time_objerr.csv / .md    data of Tables 10 and 11
      figs/                    objective-error curves against iterations and algorithm time for all methods
                               (fig_c1_iter_objerr, fig_c1_time_objerr; also ergodic objective error, dual gap and
                               KKT residual against time) on the four configurations of Table 9, written by
                               paper_analysis/analyze_c1_time_objerr.py from the raw results
      figs_tables/             tables written by plot_c1.py (main table, sweeps, reference certificates, theorem ratio
                               of the 140 main runs); aadmm_theorem_ratio_sweep.csv adds the 48 penalty-sweep runs
      paper_tables/            LaTeX of Tables 9, 10, 11 as included in the paper
      run_summary.*, manifest.json, env.txt   run record (versions, package hash, hardware)
    paper_analysis/            scripts that produced the paper's tables and figures from the raw results
  C2_supplementary/            code that produced Appendix B and Fig. 1 (package C2-supp-1.0)
    run_c2.py, c2/, c1/ (copy of the C1 code it builds on), refs/c1v12_refs.npz, toy_fig.py, plot_c2.py, tests/
    results_summary/           per_run_summary.csv, figs_tables/, paper_tables/ (Tables B1, B2), run record
    paper_analysis/analyze_c2_r9.py
  CODE_CHANGES_vs_run_packages.diff   differences between this release and the packages that were run
```

## Environment

Python 3 with NumPy and SciPy only; matplotlib is needed only for figures (`requirements.txt` in each folder).
All computations are in float64 with fixed seeds and single-threaded BLAS (set by the drivers before NumPy is imported).
The official runs used Python 3.13, NumPy 2.4.6, SciPy 1.16.3 on Google Colab CPU runtimes with single-threaded BLAS
(C1: Intel Xeon @ 2.20 GHz; C2: AMD EPYC 7B12); see `results_summary/env.txt` for the exact record.
Timings depend on the machine; iteration counts and gaps do not, up to floating-point effects.

```bash
pip install -r revision_experiments/C1_main_comparison/requirements.txt
```

## Reproducing Section 4.5 and Appendix A (C1)

```bash
cd revision_experiments/C1_main_comparison
python run_c1.py smoke                                   # < 1 min, must end with "SMOKE PASS"
python run_c1.py run --group ref    --out C1_out > ref.log 2>&1 &   # reference solutions (separate process)
python run_c1.py run --group nonref --out C1_out         # 868 method runs, sequential (for timing)
python run_c1.py pack --out C1_out                       # tables and figures
```

938 tasks (70 reference solves, 670 main runs, 198 penalty-sweep runs). In the official run the method runs took about 5 h of wall time
in total and the reference solves about 3 h (in a parallel process). Runs are resumable; see `C1_main_comparison/README.md` for details, the problem generators
P1–P4 and the guarantee scope of every method.

## Reproducing Appendix B and Fig. 1 (C2)

```bash
cd revision_experiments/C2_supplementary
python run_c2.py smoke                                   # < 3 min, must end with "SMOKE PASS"
python run_c2.py run --group all --out C2_out            # 846 tasks, about 80 min on Colab CPU
python run_c2.py pack --out C2_out
python toy_fig.py                                        # Fig. 1 (deterministic, no data)
```

C2 uses the C1 reference solutions shipped in `refs/c1v12_refs.npz` (checked by SHA-256 at run time).

## Raw results and the paper's tables and figures

The raw per-iteration results are not stored in this git repository because of their size. They are attached
to the GitHub release of this repository as two assets:

| Asset | Size | MD5 | Content |
|---|---|---|---|
| `C1_results_v12_20261115_1659.zip` | 113 MB | `ab95e7723466513cf36b4457126efa71` | `C1_out/`: 938 result files (`.npz`), logs, provenance, figures |
| `C2_results_20261115_2350.zip` | 37 MB | `f85212ecad0ee419d88a5abd072289de` | `C2_out/`: 846 result files (`.npz`), logs, provenance, figures |

To regenerate the paper's tables and figures from these results without rerunning the experiments:

```bash
cd revision_experiments/C1_main_comparison
unzip /path/to/C1_results_v12_20261115_1659.zip          # creates C1_out/
cd paper_analysis
python analysis.py && python summarize.py && python table_tex.py && python figures.py   # Table 9, Figs. 10-12, A1-A2
python analyze_c1_time_objerr.py                          # Tables 10-11
# (or set C1_OUT=/path/to/C1_out instead of unzipping next to the code)

cd ../../C2_supplementary
unzip /path/to/C2_results_20261115_2350.zip               # creates C2_out/
cd paper_analysis && python analyze_c2_r9.py              # Tables B1-B2, Figs. 1, B1-B3   (or set C2_OUT=...)
```

The summary tables in `results_summary/` were produced from exactly these results; running the scripts above on the
release assets reproduces the files in `paper_tables/` byte for byte (checked before release). Captions and some
wording of these tables were edited afterwards in the paper itself; the numbers are unchanged.

## Provenance

Each result file records the package version and the SHA-256 hash of the code that produced it
(`C1-1.2`: see `C1_main_comparison/results_summary/manifest.json`; `C2-supp-1.0`: see
`C2_supplementary/results_summary/manifest.json`). The code in this release differs from the packages that
were run only in comments and descriptive strings: references to Tang and Toh (2024) now use the numbering
of the journal version (Math. Program. Comput. 16, 113–150; rules (35) and (40), Theorem 1, Corollary 1,
Theorem 2) instead of arXiv:2310.00376v1 (rules (34) and (39)); the rules themselves are identical.
The run instructions in the folder READMEs were adapted to the folder names of this repository.
Tang–Toh example numbers and the Xu (2017) equation number in docstrings were likewise updated to the journal versions
(Examples 2–4; eq. (3.7)). The metadata string `heuristic_rule39_no_floor` recorded by `run_heu_admm` is kept unchanged
so that it matches the official result files; it refers to rule (40) of the journal version.
The two Colab notebooks are the ones used for the official runs and still refer to the run-package names
(`C1_v12_upload.zip`, `/content/C1_v12`; `C2_supp_upload.zip`, `/content/C2_supp`); to use them with this repository,
upload the corresponding folder of `revision_experiments/` instead and adjust the path in the first cell.
See `CODE_CHANGES_vs_run_packages.diff` (regenerated for this release; `RUN_LIST.json` is excluded there because its
only differences are the same descriptive `theory_scope` strings that changed in `c1/tasks.py`). Because the source files changed, the package hash computed by
the drivers differs from the hash stored in the official result files, so `run` will not mix new results
with the official ones; write new runs to a fresh output directory.
