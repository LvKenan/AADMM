import numpy as np, collections, json
from common import *
S=collections.defaultdict(list)
for p in files('main'):
    r=load(p);t=r['meta']['task'];v=vkey(t)
    if v not in('aadmm','aadmm__beta1.5','xu_acc','iadmm','heu_admm','admm__beta100'):continue
    K=r['kkt_all'];k=np.arange(1,len(K)+1);sel=k>=100
    if sel.sum()<50:continue
    # upper envelope (running max from the right) to avoid oscillation bias
    env=np.maximum.accumulate(K[::-1])[::-1]
    s=np.polyfit(np.log(k[sel]),np.log(env[sel]),1)[0]
    S[(t['P'],v)].append(s)
out={f"{a}|{b}":(round(float(np.median(x)),2),len(x)) for (a,b),x in sorted(S.items())}
json.dump(out,open('kkt_slopes.json','w'),indent=0);print(out)
