"""Task list.  A task is either a reference solve ('ref') for one problem instance, or one
method run on one instance.  Task ids are file-name safe and stable; each finished task
writes exactly one file  <out>/results/<task_id>.npz  (atomic rename), which is how
'run' knows what to skip after a disconnect."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load_config():
    with open(os.path.join(HERE, "config.json")) as fh:
        return json.load(fh)


def inst_id(P, d, seed):
    if P == "P1":
        return f"P1_m{d['m']}_p{d['p']}_s{seed}"
    if P == "P2":
        return f"P2_n{d['n']}_p{d['p']}_s{seed}"
    if P == "P3":
        return f"P3_p{d['p']}_mu{d['mu']:g}_s{seed}"
    if P == "P4":
        return f"P4_N{d['N']}_s{seed}"
    raise ValueError(P)


def variant_tag(params):
    if not params:
        return "default"
    return "_".join(f"{k}{v:g}" if isinstance(v, (int, float)) else f"{k}{v}" for k, v in sorted(params.items()))


BASE_LABEL = {"aadmm": "AADMM", "admm": "ADMM / proximal ADMM", "xu_fixed": "Xu fixed (Thm 2.13)",
              "xu_acc": "Xu accelerated (Thm 2.14)", "iadmm": "Tang-Toh IADMM", "heu_admm": "Tang-Toh Heu-ADMM",
              "fast_admm_r": "Goldstein fast ADMM (restart)"}


def task_label(method, params):
    """Human-readable label used in plots/tables; distinct for every (method, params) variant."""
    params = params or {}
    lab = BASE_LABEL[method]
    if method == "aadmm":
        return f"AADMM (beta={params.get('beta', 1.0):g})"
    if method == "admm":
        return f"{lab} (beta={params.get('beta', 1.0):g})"
    extra = []
    if "beta1_mult" in params:
        extra.append(f"beta1 x {params['beta1_mult']:g}")
    if params.get("tt_q") == "lin":
        extra.append("linearized y-step")
    return lab + (f" ({', '.join(extra)})" if extra else "")


def theory_scope(method, params):
    """Which guarantee (if any) the cited source gives for exactly this setting.  Stored in every result."""
    params = params or {}
    if method == "aadmm":
        b = float(params.get("beta", 1.0))
        assert 0.0 < b < 2.0
        return (f"this paper, Theorem (AADMM-beta), fixed beta={b:g} in (0,2): dual gap p*-d(lam~^k) <= C/(k+1)^2, "
                "i.e. O(1/k^2); t_k-weighted ergodic primal O(1/k^2); predictor residual O(1/k)")
    if method == "admm":
        b = float(params.get("beta", 1.0))
        tuned = "" if b == 1.0 else "; beta chosen from the C1-1.0 penalty sweep (tuned baseline, descriptive only)"
        return ("classical fixed-penalty (proximal) ADMM: convergence and O(1/k) ergodic rates only, no O(1/k^2) "
                f"guarantee{tuned}")
    if method == "xu_fixed":
        return "Xu (2017) journal Thm 2.13 / (2.26): O(1/k) ergodic (uniform average)"
    if method == "xu_acc":
        return ("Xu (2017) journal Thm 2.14, P=0: ergodic O(1/k^2) bounds (2.29)-(2.30) only; P=0 excludes the "
                "last-iterate bounds (2.32)")
    if method == "iadmm":
        return ("Tang-Toh (2024, MPC) Alg. 1, strategy (35): O(1/S_k) ergodic, S_k = sum of penalties; "
                "O(1/k^2) only if S_k = Omega(k^2), not implied by residual balancing (S_k reported)")
    if method == "heu_admm":
        return ("Tang-Toh (2024, MPC) Heu-ADMM, rule (40): heuristic, no rate theorem; "
                "beta-weighted average is a descriptive diagnostic")
    if method == "fast_admm_r":
        return ("Goldstein et al. (2014) Alg. 8 (restart), tau=1, eta=0.999: Thm 3 gives convergence of the combined "
                "residual c_k -> 0 only; no O(1/k^2) rate for these instances")
    raise ValueError(method)


def ref_time_budget(cfg, P):
    r = cfg["reference"]
    return float(r.get("time_budget_by_problem", {}).get(P, r["time_budget"]))


def _run_task(gid, iid, P, d, seed, mth, prm):
    return dict(id=f"{gid}__{iid}__{mth}__{variant_tag(prm)}", kind="run", group=gid, P=P,
                spec=dict(P=P, seed=seed, **d), inst=iid, method=mth, params=dict(prm),
                variant=variant_tag(prm), label=task_label(mth, prm), theory_scope=theory_scope(mth, prm))


def applicable(method, P, params=None):
    if method == "fast_admm_r":
        return P in ("P3", "P4")
    if params and params.get("tt_q") == "lin":
        return P == "P3"
    return True


def build_tasks(cfg=None, groups=("main", "sweep")):
    cfg = cfg or load_config()
    tasks = []
    seen_ref = set()

    def add_ref(P, d, seed):
        iid = inst_id(P, d, seed)
        if iid not in seen_ref:
            seen_ref.add(iid)
            tasks.append(dict(id=f"ref__{iid}", kind="ref", group="ref", P=P, spec=dict(P=P, seed=seed, **d), inst=iid))
        return iid

    if "main" in groups:
        for P, lst in cfg["instances"].items():
            for d in lst:
                for seed in cfg["seeds"]:
                    iid = add_ref(P, d, seed)
                    for mth in cfg["main_methods"]:
                        if applicable(mth, P):
                            tasks.append(_run_task("main", iid, P, d, seed, mth, {}))
                    for mth, prm in cfg.get("main_variants", []):
                        if applicable(mth, P, prm):
                            tasks.append(_run_task("main", iid, P, d, seed, mth, prm))
    if "sweep" in groups:
        for P, d in cfg["sweep_instances"].items():
            for seed in cfg["sweep_seeds"]:
                iid = add_ref(P, d, seed)
                for mth, prm in cfg["sweep_variants"]:
                    if applicable(mth, P, prm):
                        tasks.append(_run_task("sweep", iid, P, d, seed, mth, prm))
    return tasks


def smoke_tasks():
    """Tiny instances, every method, every problem (used by 'smoke')."""
    specs = [("P1", {"m": 40, "p": 120}), ("P2", {"n": 60, "p": 150, "s": 15}),
             ("P3", {"p": 80, "mu": 0.5}), ("P4", {"N": 16})]
    tasks = []
    for P, d in specs:
        iid = inst_id(P, d, 0)
        tasks.append(dict(id=f"ref__{iid}", kind="ref", group="ref", P=P, spec=dict(P=P, seed=0, **d), inst=iid))
        for mth, prm in [(m, {}) for m in ["aadmm", "admm", "xu_fixed", "xu_acc", "iadmm", "heu_admm", "fast_admm_r"]] \
                + [("admm", {"beta": 10.0}), ("admm", {"beta": 100.0}), ("aadmm", {"beta": 1.5})]:
            if applicable(mth, P, prm):
                tasks.append(_run_task("smoke", iid, P, d, 0, mth, prm))
    return tasks
