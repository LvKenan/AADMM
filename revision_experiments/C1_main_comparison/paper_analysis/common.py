"""Shared loaders for the C1-1.2 (Carbon, 2026-11-15) results."""
import json, glob, os, re
import numpy as np
RES = os.path.join(os.environ.get("C1_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "C1_out")), "results")
def load(path):
    d = np.load(path, allow_pickle=False)
    out = dict(meta=json.loads(str(d["meta"])))
    for k in d.files:
        if k != "meta":
            out[k] = d[k]
    return out
def files(prefix):
    return sorted(glob.glob(os.path.join(RES, prefix + "__*.npz")))
def H(r, name):
    return r["hist"][:, r["meta"]["fields"].index(name)]
def cfgname(inst):
    return re.sub(r"_s\d+$", "", inst)
def vkey(t):
    v = t.get("variant", "default")
    return t["method"] if v == "default" else f"{t['method']}__{v}"
_refs = None
def refs():
    global _refs
    if _refs is None:
        _refs = {}
        for p in files("ref"):
            r = load(p); m = r["meta"]
            _refs[m["task"]["inst"]] = dict(F=m["Fstar"], LB=m["LB"], UB=m["UB"], gap=m["gap_rel"], status=m["status"],
                                            lam=r["lam_star"], By=r["By_star"], iters=m["iters"], time=m["time"], cfg=m["ref_cfg"])
    return _refs
def first_reach(k, v, lev):
    i = np.where(v <= lev)[0]
    return int(k[i[0]]) if len(i) else None
