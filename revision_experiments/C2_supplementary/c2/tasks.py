"""C2 task list (stable, file-name-safe ids; one result file per task).
Groups:  ref       LASSO reference solves (b) -- TVD and (a) use the shipped C1-1.2 references (refs/)
         a_struct  Remark rem:dualfista (i)-(iii): AADMM-beta (compatible init) vs FISTA-beta, traces of the difference
         a_range   AADMM-beta vs dual FISTA vs dual proximal gradient over a beta grid (C7)
         b         inexact x-subproblems (C5)"""
import json, os
from c1.tasks import inst_id
HERE = os.path.dirname(os.path.abspath(__file__))


def load_config():
    with open(os.path.join(HERE, "config.json")) as fh:
        return json.load(fh)


def b_inst_id(d, seed):
    return f"LASSO_m{d['m']}_n{d['n']}_s{seed}" if d["kind"] == "LASSO" else f"TVD_N{d['N']}_s{seed}"


def sched_tag(s):
    return f"tol_c{s['c']:.0e}_p{s['p']}" if s["mode"] == "tol" else f"count_n{s['n']}"


SCOPE_A = {
    "aadmm": lambda b: ("Theorem Theor-AADMM: dual gap O(1/k^2) for beta in (0,2) on the normalized problem" if b < 2
                        else "OUTSIDE the theory (beta >= 2): no guarantee; recorded to show where the range ends"),
    "dual_fista": lambda b: ("FISTA theory (Beck-Teboulle 2009): step beta <= 1/L_Psi = 1.001" if b <= 1.001
                             else "OUTSIDE standard FISTA theory (step > 1/L_Psi): no guarantee"),
    "dual_pg": lambda b: ("proximal-gradient theory: step beta < 2/L_Psi = 2.002" if b < 2.002 else "OUTSIDE proximal-gradient theory"),
}


def build_tasks(cfg=None):
    cfg = cfg or load_config()
    T = []
    seeds = cfg["seeds"]
    for d in cfg["b_instances"]:
        if d["kind"] == "LASSO":
            for s in d.get("seeds", seeds):
                iid = b_inst_id(d, s)
                T.append(dict(id=f"ref__{iid}", kind="ref", group="ref", inst=iid,
                              spec=dict({k: v for k, v in d.items() if k != "seeds"}, seed=s)))
    for d in cfg["a_struct"]["instances"]:
        for s in seeds:
            spec = dict(d, seed=s); iid = inst_id(d["P"], d, s)
            for b in cfg["a_struct"]["betas"]:
                T.append(dict(id=f"a_struct__{iid}__beta{b:g}", kind="a_struct", group="a_struct", inst=iid, spec=spec,
                              params=dict(beta=b), label=f"AADMM-beta vs FISTA-beta, beta={b:g}",
                              theory_scope="Remark rem:dualfista (i) (beta=1, affine grad Psi: identical), (ii), (iii)"))
    for d in cfg["a_range"]["instances"]:
        for s in seeds:
            spec = dict(d, seed=s); iid = inst_id(d["P"], d, s)
            for m in cfg["a_range"]["methods"]:
                for b in cfg["a_range"]["betas"]:
                    T.append(dict(id=f"a_range__{iid}__{m}__beta{b:g}", kind="a_range", group="a_range", inst=iid, spec=spec,
                                  method=m, params=dict(beta=b), label=f"{m} beta={b:g}", theory_scope=SCOPE_A[m](b)))
    for d in cfg["b_instances"]:
        for s in d.get("seeds", seeds):
            iid = b_inst_id(d, s); spec = {k: v for k, v in d.items() if k != "seeds"}; spec["seed"] = s
            def resolve(sc):
                sc = dict(sc)
                if sc["mode"] == "count":
                    sc["n"] = {"paper": d["n_paper"], "5paper": 5 * d["n_paper"]}.get(sc["n"], sc["n"])
                return sc
            for b in cfg["b_aadmm_betas"]:
                for sc in map(resolve, cfg["b_aadmm_schedules"]):
                    T.append(dict(id=f"b__{iid}__aadmm__beta{b:g}__{sched_tag(sc)}", kind="b", group="b", inst=iid, spec=spec,
                                  method="aadmm", params=dict(beta=b, sched=sc), label=f"AADMM beta={b:g} {sched_tag(sc)}",
                                  theory_scope="exact-subproblem theory only (Remark rem:inexact): no guarantee claimed for inexact runs"))
            for sc in map(resolve, cfg["b_admm_schedules"]):
                T.append(dict(id=f"b__{iid}__admm__beta1__{sched_tag(sc)}", kind="b", group="b", inst=iid, spec=spec,
                              method="admm", params=dict(beta=1.0, sched=sc), label=f"ADMM beta=1 {sched_tag(sc)}",
                              theory_scope="classical ADMM, exact-subproblem O(1/k) ergodic theory; inexact runs: none claimed"))
    ids = [t["id"] for t in T]
    assert len(ids) == len(set(ids))
    return T


def smoke_tasks():
    """Tiny versions of every task kind (run with reduced caps by 'smoke')."""
    T = []
    T.append(dict(id="ref__LASSO_m40_n120_s0", kind="ref", group="ref", inst="LASSO_m40_n120_s0",
                  spec=dict(kind="LASSO", m=40, n=120, n_paper=10, seed=0)))
    sp = dict(P="P3", p=60, mu=1.0, seed=0)
    for b in (1.0, 1.5):
        T.append(dict(id=f"a_struct__P3_p60_mu1_s0__beta{b:g}", kind="a_struct", group="a_struct", inst="P3_p60_mu1_s0", spec=sp,
                      params=dict(beta=b), label="smoke", theory_scope="smoke"))
    sp1 = dict(P="P1", m=20, p=60, seed=0)
    T.append(dict(id="a_struct__P1_m20_p60_s0__beta1", kind="a_struct", group="a_struct", inst="P1_m20_p60_s0", spec=sp1,
                  params=dict(beta=1.0), label="smoke", theory_scope="smoke"))
    sp4 = dict(P="P4", N=16, seed=0)
    for m in ("aadmm", "dual_fista", "dual_pg"):
        for b in (1.0, 1.9, 4.0):
            T.append(dict(id=f"a_range__P4_N16_s0__{m}__beta{b:g}", kind="a_range", group="a_range", inst="P4_N16_s0", spec=sp4,
                          method=m, params=dict(beta=b), label="smoke", theory_scope="smoke"))
    for kind, d in (("LASSO", dict(kind="LASSO", m=40, n=120, n_paper=10, seed=0)), ("TVD", dict(kind="TVD", N=16, n_paper=20, seed=0))):
        iid = b_inst_id(d, 0)
        for meth, b, sc in (("aadmm", 1.0, dict(mode="tol", c=1e-11, p=0)), ("aadmm", 1.5, dict(mode="tol", c=1e-4, p=2)),
                            ("aadmm", 1.5, dict(mode="count", n=10)), ("admm", 1.0, dict(mode="count", n=10))):
            T.append(dict(id=f"b__{iid}__{meth}__beta{b:g}__{sched_tag(sc)}", kind="b", group="b", inst=iid, spec=d,
                          method=meth, params=dict(beta=b, sched=sc), label="smoke", theory_scope="smoke"))
    return T
