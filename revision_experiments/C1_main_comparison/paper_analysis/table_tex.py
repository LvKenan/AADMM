import json, numpy as np
runs=json.load(open('runs_r6.json'))
rep=[('P1_m300_p1000','P1, $m{=}300$'),('P2_n700_p2000','P2, $n{=}700$'),('P3_p1000_mu0.1','P3, $\\mu{=}0.1$'),('P4_N128','P4, $N{=}128$')]
ORDER=['aadmm','aadmm__beta1.5','admm','admm__beta10','admm__beta100','xu_fixed','xu_acc','iadmm','heu_admm','fast_admm_r']
name={'aadmm':'AADMM, $\\beta{=}1$','aadmm__beta1.5':'AADMM, $\\beta{=}1.5$','admm':'ADMM, $\\beta{=}1$','admm__beta10':'ADMM, $\\beta{=}10$','admm__beta100':'ADMM, $\\beta{=}100$',
'xu_fixed':'Xu, fixed','xu_acc':'Xu, accel.','iadmm':'IADMM','heu_admm':'Heu-ADMM','fast_admm_r':'Fast ADMM-r'}
scope={'aadmm':'$v_k{=}O(k^{-2})$','aadmm__beta1.5':'$v_k{=}O(k^{-2})$','admm':'$O(k^{-1})$ erg.','admm__beta10':'$O(k^{-1})$ erg.','admm__beta100':'$O(k^{-1})$ erg.',
'xu_fixed':'$O(k^{-1})$ erg.','xu_acc':'$O(k^{-2})$ erg.','iadmm':'$O(1/S_k)$ erg.','heu_admm':'none','fast_admm_r':'$c_k\\to0$'}
D={}
for r in runs:
    if r['group']=='main':D.setdefault((r['cfg'],r['v']),[]).append(r)
def cell(g,get):
    w=[get(x) for x in g];ok=[x for x in w if x is not None]
    if not ok:return '--'
    s=f"{int(round(np.median(ok)))}"
    return s if len(ok)==len(w) else f"{s}$^{{{len(ok)}}}$"
L=[r'\begin{table}[htbp]',r'\centering',
r"""\caption{Iterations to reach $\max(R_p,R_d)\le10^{-6}$ (KKT) and a relative dual gap $(\widehat F^{*}-d(\lambda^{k}))/(1+|\widehat F^{*}|)\le10^{-8}$ (gap); median over $5$ random instances. Runs stop at $\max(R_p,R_d)\le10^{-8}$ or after $20000$ iterations. A superscript gives the number of instances (out of $5$) that reached the level when fewer than five did (the median is then over those); ``--'': not reached before the run stopped; n/a: not applicable (no exact $y$-subproblem). For AADMM-$\beta$ the gap is evaluated at the predictor $\tilde\lambda^{k}$, for the other methods at their multiplier iterate. Xu: \cite{Xu-2017}; IADMM, Heu-ADMM: \cite{Tang-2024}; Fast ADMM-r: restarted fast ADMM \cite{Goldstein-2014}. Configurations: P1 $(m,p)=(300,1000)$, P2 $(n,p)=(700,2000)$, P3 $p=1000$, P4 $N\times N$ image. The last column states the guarantee that applies to the configuration as run (erg.: ergodic average); see the text.}\label{tab:c1-main}""",
r'\scriptsize\setlength{\tabcolsep}{2.6pt}',
r'\begin{tabular}{l'+'rr'*4+'l}',r'\toprule',
' & '+' & '.join(f'\\multicolumn{{2}}{{c}}{{{t}}}' for _,t in rep)+r' & \\',
''.join(f'\\cmidrule(lr){{{2+2*i}-{3+2*i}}}' for i in range(4)),
'Method & '+' & '.join(['KKT & gap']*4)+r' & Guarantee \\',r'\midrule']
for v in ORDER:
    cells=[]
    for c,_ in rep:
        g=D.get((c,v))
        if not g: cells+=['n/a','n/a'];continue
        cells+=[cell(g,lambda x:(x['kkt_hit']['1e-06'] or [None])[0]),cell(g,lambda x:x['dg_hit']['1e-08'])]
    L.append(name[v]+' & '+' & '.join(cells)+f' & {scope[v]} \\\\')
    if v in('aadmm__beta1.5','admm__beta100'):L.append(r'\addlinespace[2pt]')
L+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
open('table_c1_main.tex','w').write('\n'.join(L)+'\n');print('\n'.join(L))
