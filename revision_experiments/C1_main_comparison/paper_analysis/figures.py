"""Paper-size, vector-PDF + 600 dpi PNG figures, read only from C1-1.2 results.
Width=130.75mm (372pt textwidth in sn-jnl). Reproduce: python figures.py.
"""
import os, json, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import *
OUT = 'paper_figs'; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif':['STIXGeneral','DejaVu Serif'], 'mathtext.fontset':'stix', 'font.size':8,
                     'axes.labelsize':8, 'axes.titlesize':8, 'xtick.labelsize':7, 'ytick.labelsize':7,
                     'legend.fontsize':6.7, 'axes.linewidth':.55, 'lines.linewidth':1.,
                     'pdf.fonttype':42, 'ps.fonttype':42, 'savefig.dpi':600,
                     'xtick.major.width':.5,'ytick.major.width':.5})
W=372/72.27
ORDER=['aadmm','aadmm__beta1.5','admm','admm__beta10','admm__beta100','xu_fixed','xu_acc','iadmm','heu_admm','fast_admm_r']
labels={'aadmm':r'AADMM $\beta=1$', 'aadmm__beta1.5':r'AADMM $\beta=1.5$', 'admm':r'ADMM $\beta=1$',
        'admm__beta10':r'ADMM $\beta=10$', 'admm__beta100':r'ADMM $\beta=100$', 'xu_fixed':'Xu fixed',
        'xu_acc':'Xu accelerated', 'iadmm':'IADMM', 'heu_admm':'Heu-ADMM', 'fast_admm_r':'Goldstein restart'}
colors={'aadmm':'#CC3311','aadmm__beta1.5':'#CC3311', 'admm':'#0077BB','admm__beta10':'#0077BB','admm__beta100':'#0077BB',
        'xu_fixed':'#777777', 'xu_acc':'#EE7733', 'iadmm':'#009988','heu_admm':'#AA4499','fast_admm_r':'#222222'}
styles={'aadmm':'-', 'aadmm__beta1.5':'--', 'admm':':','admm__beta10':'--','admm__beta100':'-.',
        'xu_fixed':':', 'xu_acc':'-','iadmm':'-','heu_admm':'-','fast_admm_r':'--'}
rep=['P1_m300_p1000','P2_n700_p2000','P3_p1000_mu0.1','P4_N128']
titles=[r'(a) P1: $m=300,\ p=1000$',r'(b) P2: $n=700,\ p=2000$',r'(c) P3: $p=1000,\ \mu=0.1$',r'(d) P4: $N=128$']
R=refs()
main=collections.defaultdict(list)
for p in files('main'):
    r=load(p);t=r['meta']['task'];main[(cfgname(t['inst']),vkey(t))].append(r)
def save(fig,name):
    fig.savefig(f'{OUT}/{name}.pdf',metadata={'Title':name, 'Creator':'R6 C1-1.2 analysis'})
    fig.savefig(f'{OUT}/{name}.png'); plt.close(fig)
def prep(ax):
    ax.grid(True, which='major', alpha=.20, linewidth=.5)
    ax.tick_params(which='both',top=False,right=False,pad=2)
    for spine in ('top','right'): ax.spines[spine].set_visible(False)
def legend_all(fig,keys=ORDER,ncol=5,y=.005):
    hs=[Line2D([0],[0],color=colors[v],ls=styles[v],lw=1.2,label=labels[v]) for v in keys]
    fig.legend(handles=hs,loc='lower center',ncol=ncol,bbox_to_anchor=(.5,y),frameon=False,
               columnspacing=.8,handlelength=2.3,handletextpad=.35)
def median_curve(rs,metric):
    # Interpolate logged observations; show only horizon shared by all five seeds.
    end=min(int(H(r,'k')[-1]) for r in rs)
    grid=np.unique(np.rint(np.geomspace(1,end,450)).astype(int))
    vs=[]
    for r in rs:
        k=H(r,'k');t=r['meta']['task'];F=R[t['inst']]['F']
        y=(F-H(r,'dval'))/(1+abs(F)) if metric=='gap' else H(r,'KKT')
        vals=np.interp(np.log(grid),np.log(k),np.log(np.maximum(y,1e-300)))
        vs.append(np.exp(vals))
    return grid,np.median(vs,axis=0)
for metric,name,yl in [('gap','fig_c1_dualgap',r'$(\widehat F^*-d(\lambda^k))/(1+|\widehat F^*|)$'),
                       ('kkt','fig_c1_kkt',r'$\max(R_p,R_d)$')]:
    fig,axs=plt.subplots(2,2,figsize=(W,4.35),sharex=False,sharey=True)
    for ax,c,ttl in zip(axs.flat,rep,titles):
        for v in ORDER:
            rs=main.get((c,v))
            if not rs:continue
            k,y=median_curve(rs,metric)
            mask=y>1e-10 if metric=='gap' else y>0
            ax.loglog(k[mask],y[mask],color=colors[v],ls=styles[v],lw=1.35 if v.startswith('aadmm') else .95,zorder=5 if v.startswith('aadmm') else 2)
        ax.axhline(1e-8,color='.6',lw=.6,ls=':');ax.set_ylim((1e-10,1.) if metric=='gap' else (7e-9,1.))
        ax.set_title(ttl,pad=4); ax.set_xlabel('Iteration $k$',labelpad=1);prep(ax)
    for ax in axs[:,0]:ax.set_ylabel(yl,labelpad=2)
    legend_all(fig)
    fig.subplots_adjust(left=.13,right=.99,bottom=.16,top=.95,hspace=.40,wspace=.13)
    save(fig,name)
# theorem: maximum across available runs at each iteration, all main+sweep, per beta
ratios=collections.defaultdict(list); counts=collections.Counter()
for p in files('main')+files('sweep'):
    r=load(p);t=r['meta']['task']
    if t['method']!='aadmm':continue
    b=float(t['params'].get('beta',1.)); ref=R[t['inst']]
    C=float(ref['lam']@ref['lam'])/b + float(np.sum((ref['lam']-2*ref['By'])**2))/(2-b)
    k=np.arange(1,len(r['lpred_all'])+1);rho=(k+1)**2*(ref['F']-r['lpred_all'])/C
    ratios[(t['P'],b)].append(rho);counts[t['P']]+=1
bcs={.25:'#777777',.5:'#0077BB',1.:'#CC3311',1.5:'#009988',1.9:'#AA4499'}
fig,axs=plt.subplots(2,2,figsize=(W,3.9),sharey=True)
for ax,P in zip(axs.flat,['P1','P2','P3','P4']):
    for b in sorted(bcs):
        rs=ratios.get((P,b),[])
        if not rs:continue
        n=max(map(len,rs));arr=np.full((len(rs),n),np.nan)
        for i,v in enumerate(rs):arr[i,:len(v)]=v
        env=np.nanmax(arr,axis=0); k=np.arange(1,n+1)
        ax.semilogx(k,env,color=bcs[b],lw=1.05,label=rf'$\beta={b:g}$')
    ax.axhline(1,color='k',ls='--',lw=.7);ax.set_ylim(0,1.05);ax.set_title(f'{P}: {counts[P]} runs',pad=3)
    ax.set_xlabel('Iteration $k$',labelpad=1);prep(ax)
for ax in axs[:,0]:ax.set_ylabel(r'Maximum $(k+1)^2\widehat v_k/\widehat C$',labelpad=2)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,.001))
fig.subplots_adjust(left=.13,right=.99,bottom=.15,top=.94,hspace=.45,wspace=.15)
save(fig,'fig_c1_theorem')
# Penalty growth IADMM / Heu on the four representative configurations (5 seeds)
fig,axs=plt.subplots(1,2,figsize=(W,2.5))
pcols={'P1':'#CC3311','P2':'#0077BB','P3':'#AA4499','P4':'#009988'}
for ax,mth in zip(axs,['iadmm','heu_admm']):
    for c in rep:
        P=c[:2];rs=main[(c,mth)];end=min(len(r['pen_all']) for r in rs)
        k=np.arange(1,end+1);q=np.array([np.cumsum(r['pen_all'])[:end]/k**2 for r in rs])
        ax.loglog(k,np.median(q,axis=0),color=pcols[P],label=P)
        ax.fill_between(k,q.min(axis=0),q.max(axis=0),color=pcols[P],alpha=.12,lw=0)
    ax.set_title('IADMM' if mth=='iadmm' else 'Heu-ADMM',pad=4);ax.set_xlabel('Iteration $k$');prep(ax)
axs[0].set_ylabel(r'$S_k/k^2$',labelpad=1)
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,.0))
fig.subplots_adjust(left=.12,right=.99,bottom=.27,top=.87,wspace=.35)
save(fig,'fig_c1_penalty_growth')
# beta effect on paired per-instance hits (main); crosses n/a not invented; 4 family boxplot-ish dot/CI summary
rr=json.load(open('runs_r6.json'));D={(r['inst'],r['v']):r for r in rr if r['group']=='main'}
fig,axs=plt.subplots(1,2,figsize=(W,2.55),sharey=True)
for ax,metric,ttl in zip(axs,['kkt','gap'],[r'(a) KKT $\leq10^{-6}$',r'(b) Relative dual gap $\leq10^{-8}$']):
    for i,P in enumerate(['P1','P2','P3','P4']):
        vals=[]
        for (inst,v),a in D.items():
            if v!='aadmm' or not inst.startswith(P):continue
            b=D[(inst,'aadmm__beta1.5')]
            aa=(a['kkt_hit']['1e-06'] or [None])[0] if metric=='kkt' else a['dg_hit']['1e-08']
            bb=(b['kkt_hit']['1e-06'] or [None])[0] if metric=='kkt' else b['dg_hit']['1e-08']
            if aa and bb:vals.append(bb/aa)
        xs=i+np.linspace(-.12,.12,len(vals));ax.scatter(xs,vals,s=10,color=pcols[P],alpha=.6,lw=0)
        if vals:ax.plot([i-.2,i+.2],[np.median(vals)]*2,color='k',lw=1.2)
        ax.text(i,.60,f'n={len(vals)}',ha='center',fontsize=6.5)
    ax.axhline(1,color='.4',lw=.8,ls='--');ax.set_xticks(range(4),['P1','P2','P3','P4']);ax.set_xlim(-.45,3.45);ax.set_ylim(.58,1.35)
    ax.set_title(ttl);prep(ax)
axs[0].set_ylabel(r'Iterations: $\beta=1.5$ / $\beta=1$')
fig.subplots_adjust(left=.13,right=.99,bottom=.11,top=.89,wspace=.13)
save(fig,'fig_c1_beta_pair')
print('wrote',len([x for x in os.listdir(OUT) if x.endswith('.pdf')]),'PDF+PNG pairs')
