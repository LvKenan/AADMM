import json, collections, numpy as np
runs = json.load(open("runs_r6.json"))
for r in runs:
    for k, v in list(r["kkt_hit"].items()):
        if v is None: r["kkt_hit"][k] = [None, None]
def med(xs, n=None):
    w = [x for x in xs if x is not None]
    n = len(xs) if n is None else n
    if not w: return f"— (0/{n})"
    s = f"{np.median(w):.4g}"
    return s if len(w) == n else s + f" ({len(w)}/{n})"
L = []
# 1 theorem
L.append("## T1 theorem bound (k+1)^2 v_k <= C, exact per-iteration L(pred)\n")
L.append("| P | group | beta | n | max ratio | median max ratio | violations | weak-duality violations | median slope k>=100 | median slope [100,1000] |\n|---|---|---|---|---|---|---|---|---|---|")
G = collections.defaultdict(list)
for r in runs:
    if r["method"] == "aadmm": G[(r["P"], r["group"], r["beta"])].append(r)
for key in sorted(G):
    g = G[key]
    sl = [x["slope"] for x in g if x["slope"] is not None]; sl2 = [x["slope_100_1000"] for x in g if x["slope_100_1000"] is not None]
    L.append(f"| {key[0]} | {key[1]} | {key[2]} | {len(g)} | {max(x['max_ratio'] for x in g):.3f} | {np.median([x['max_ratio'] for x in g]):.3f} | "
             f"{sum(x['n_viol'] for x in g)} | {sum(x['n_weak_dual_viol'] for x in g)} | {np.median(sl) if sl else float('nan'):.2f} | {np.median(sl2) if sl2 else float('nan'):.2f} |")
na = sum(len(g) for g in G.values()); L.append(f"\nAADMM runs checked: {na}; total violations {sum(x['n_viol'] for g in G.values() for x in g)}\n")
# 2 main table
ORDER = ["aadmm", "aadmm__beta1.5", "admm", "admm__beta10", "admm__beta100", "xu_fixed", "xu_acc", "iadmm", "heu_admm", "fast_admm_r"]
M = collections.defaultdict(list)
for r in runs:
    if r["group"] == "main": M[(r["cfg"], r["v"])].append(r)
cfgs = sorted({k[0] for k in M}, key=lambda c: (c[:2], int(''.join(ch for ch in c.split('_')[1] if ch.isdigit())), c))
L.append("## T2 main table (median over 5 seeds; KKT from exact first_hit; rel. dual gap (F*-d)/(1+|F*|) from logged points)\n")
L.append("| config | method | KKT≤1e-4 | KKT≤1e-6 | KKT≤1e-8 | t_alg@1e-8 [s] | gap≤1e-6 | gap≤1e-8 | gap≤1e-10 | final gap | final KKT | iters | stops |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
T2 = {}
for c in cfgs:
    for v in ORDER:
        g = M.get((c, v))
        if not g: continue
        n = len(g)
        row = dict(k4=med([x["kkt_hit"]["0.0001"][0] for x in g]), k6=med([x["kkt_hit"]["1e-06"][0] for x in g]),
                   k8=med([x["kkt_hit"]["1e-08"][0] for x in g]), t8=med([x["kkt_hit"]["1e-08"][1] for x in g]),
                   d6=med([x["dg_hit"]["1e-06"] for x in g]), d8=med([x["dg_hit"]["1e-08"] for x in g]), d10=med([x["dg_hit"]["1e-10"] for x in g]),
                   dgf=f"{np.median([x['dg_final'] for x in g]):.2e}", kf=f"{np.median([x['final_kkt'] for x in g]):.2e}",
                   it=f"{np.median([x['iters'] for x in g]):.0f}", st=",".join(f"{a}:{b}" for a, b in sorted(collections.Counter(x["stop"] for x in g).items())))
        T2[(c, v)] = row
        L.append(f"| {c} | {v} | " + " | ".join(row[k] for k in ("k4", "k6", "k8", "t8", "d6", "d8", "d10", "dgf", "kf", "it", "st")) + " |")
json.dump({f"{a}|{b}": v for (a, b), v in T2.items()}, open("table_main_r6.json", "w"), indent=0)
# 3 beta 1.5 vs 1 paired
L.append("\n## T3 AADMM beta=1.5 vs beta=1 (main, paired by instance): ratio iters(1.5)/iters(1)\n")
L.append("| P | level | n pairs both reached | median ratio | 1.5 faster in |\n|---|---|---|---|---|")
by = {(r["inst"], r["v"]): r for r in runs if r["group"] == "main"}
for P in ("P1", "P2", "P3", "P4"):
    for nm, getter in [("KKT 1e-6", lambda x: x["kkt_hit"]["1e-06"][0]), ("KKT 1e-8", lambda x: x["kkt_hit"]["1e-08"][0]), ("gap 1e-8", lambda x: x["dg_hit"]["1e-08"]), ("gap 1e-10", lambda x: x["dg_hit"]["1e-10"])]:
        rat = []
        for (inst, v), a in by.items():
            if v != "aadmm" or not inst.startswith(P): continue
            b = by[(inst, "aadmm__beta1.5")]
            if getter(a) and getter(b): rat.append(getter(b) / getter(a))
        if rat: L.append(f"| {P} | {nm} | {len(rat)} | {np.median(rat):.3f} | {sum(x < 1 for x in rat)}/{len(rat)} |")
# 4 tuned ADMM vs AADMM: per config best variant at KKT 1e-6 / gap 1e-8
L.append("\n## T4 per config: fastest variant (median iters) for KKT<=1e-6 and rel gap<=1e-8 (only variants with all 5 seeds reaching)\n")
L.append("| config | fastest KKT 1e-6 | AADMM(1) KKT 1e-6 | fastest gap 1e-8 | AADMM(1) gap 1e-8 | best ADMM β for KKT 1e-6 |\n|---|---|---|---|---|---|")
for c in cfgs:
    def best(get):
        cand = []
        for v in ORDER:
            g = M.get((c, v))
            if g and all(get(x) for x in g): cand.append((np.median([get(x) for x in g]), v))
        return sorted(cand)
    b6 = best(lambda x: x["kkt_hit"]["1e-06"][0]); b8 = best(lambda x: x["dg_hit"]["1e-08"])
    a6 = [x for x in b6 if x[1] == "aadmm"]; a8 = [x for x in b8 if x[1] == "aadmm"]
    adm = [x for x in b6 if x[1].startswith("admm")]
    L.append(f"| {c} | {b6[0][1]} {b6[0][0]:.0f} | " + (f"{a6[0][0]:.0f} (rank {b6.index(a6[0])+1}/{len(b6)})" if a6 else "not all reached"))
    L[-1] += f" | {b8[0][1]} {b8[0][0]:.0f} | " + (f"{a8[0][0]:.0f} (rank {b8.index(a8[0])+1}/{len(b8)})" if a8 else "not all reached") + " | " + (f"{adm[0][1]} {adm[0][0]:.0f}" if adm else "none reached") + " |"
# 5 S_k
L.append("\n## T5 IADMM / Heu-ADMM penalty sum S_k (main, exact per-iteration)\n")
L.append("| P | method | n | median S_k/k^2 @k=100 | @k=1000 | final | median exponent a in S_k~k^a (k>=100) | max over runs of max_k beta_k/k | median beta_last |\n|---|---|---|---|---|---|---|---|---|")
S = collections.defaultdict(list)
for r in runs:
    if r["method"] in ("iadmm", "heu_admm") and r["group"] == "main": S[(r["P"], r["method"])].append(r)
for key in sorted(S):
    g = S[key]
    def mm(kx): 
        w = [x["Sk_k2"].get(str(kx)) for x in g]; w = [x for x in w if x is not None]
        return f"{np.median(w):.3g} ({len(w)})" if w else "—"
    ex = [x["Sk_exponent"] for x in g if x["Sk_exponent"] is not None]
    L.append(f"| {key[0]} | {key[1]} | {len(g)} | {mm(100)} | {mm(1000)} | {np.median([x['Sk_k2_final'] for x in g]):.3g} | {np.median(ex):.2f} [{min(ex):.2f},{max(ex):.2f}] | {max(x['pen_max_over_k'] for x in g):.3g} | {np.median([x['pen_last'] for x in g]):.3g} |")
L.append(f"\nS_k consistency with run_summary: max rel diff {max(x['S_check'] for g in S.values() for x in g):.1e}")
# 6 restarts
L.append("\n## T6 Goldstein fast ADMM restarts (main)\n")
L.append("| config | n | median restarts | range | median restart fraction | stops |\n|---|---|---|---|---|---|")
for c in cfgs:
    g = M.get((c, "fast_admm_r"))
    if not g: continue
    nr = [x["n_restart"] for x in g]
    L.append(f"| {c} | {len(g)} | {np.median(nr):.0f} | {min(nr)}–{max(nr)} | {np.median([x['restart_frac'] for x in g]):.3f} | {','.join(f'{a}:{b}' for a,b in sorted(collections.Counter(x['stop'] for x in g).items()))} |")
open("summary_r6.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
