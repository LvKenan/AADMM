"""Reference values.  (a) and TVD (b): C1-1.2 references shipped in refs/c1v12_refs.npz (extracted unchanged from the
C1-1.2 results; per-instance source sha256 inside).  The instance is regenerated from the same C1 generator and its
fingerprint (scaling s, ||B||^2, ||b||) must match the C1-1.2 record, otherwise the task FAILS (no fallback).
Smoke-only instances are not in the file: smoke computes them with c1.reference.reference_solve."""
import json, os
import numpy as np
from c1.problems import make_problem, P4_TV
from c1.reference import reference_solve
HERE = os.path.dirname(os.path.abspath(__file__))
REF_FILE = os.path.join(os.path.dirname(HERE), "refs", "c1v12_refs.npz")
REF_SHA256 = "a66d2bf78a73323ce9ec8a0240572823804606bd87bcfdcf0f8ba4bde04ef409"
_cache = {}


def _load():
    if "z" not in _cache:
        import hashlib
        with open(REF_FILE, "rb") as fh:
            h = hashlib.sha256(fh.read()).hexdigest()
        if h != REF_SHA256:
            raise ValueError(f"refs/c1v12_refs.npz sha256 mismatch: {h}")
        z = np.load(REF_FILE, allow_pickle=False)
        _cache["z"] = z
        _cache["meta"] = json.loads(str(z["meta"]))
    return _cache["z"], _cache["meta"]


def fingerprint_check(prob, rec):
    for k in ("s", "normB2_orig", "normb_orig"):
        a, b = float(prob.info()[k]), float(rec["problem"][k])
        if abs(a - b) > 1e-12 * max(1.0, abs(b)):
            raise ValueError(f"instance {prob.inst}: regenerated {k}={a!r} differs from C1-1.2 record {b!r}")


def c1_ref(inst, spec, allow_compute=False):
    """Returns dict(Fstar, lam_star, By_star, y_star, source) for a C1 instance (scaled problem)."""
    z, meta = _load()
    if inst in meta:
        prob = make_problem(spec)
        fingerprint_check(prob, meta[inst])
        return dict(Fstar=meta[inst]["Fstar"], gap_rel=meta[inst]["gap_rel"], lam_star=z[f"{inst}__lam_star"],
                    By_star=z[f"{inst}__By_star"], y_star=z[f"{inst}__y_star"], source="C1-1.2:" + meta[inst]["source_file"])
    if not allow_compute:
        raise KeyError(f"no shipped C1-1.2 reference for {inst}")
    prob = make_problem(spec)
    r = reference_solve(prob, tol=1e-12, maxit=200000, time_budget=60.0)
    if r["status"] != "tol":
        raise RuntimeError(f"smoke reference for {inst} not within tolerance: {r['status']} {r['gap_rel']:.1e}")
    return dict(Fstar=r["Fstar"], gap_rel=r["gap_rel"], lam_star=r["lam_star"], By_star=r["By_star"], y_star=r["y_star"],
                source="computed (smoke only)")


def tvd_ref(P, allow_compute=False):
    """TVD p* = 1/2||f0||^2 - F*_P4,  u* = y*_P4  (same image f0, same seed), By* = q* = u*."""
    N, seed = P.N, P.seed
    r = c1_ref(P.p4inst, dict(P="P4", N=N, seed=seed), allow_compute)
    u = r["y_star"]
    return dict(pstar=0.5 * float(P.b @ P.b) - r["Fstar"], lam_star=u, By_star=u, source=r["source"],
                gap_rel_P4=r["gap_rel"], d_at_ustar=P.d(u))
