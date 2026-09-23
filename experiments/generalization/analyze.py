import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import NullFormatter
ROOT=Path(__file__).resolve().parent
def read(name):return [json.loads(x) for x in (ROOT/'raw'/(name+'.jsonl')).read_text().splitlines()]
def save(name,rows):
    (ROOT/'results'/(name+'.json')).write_text(json.dumps(rows,indent=2))
    if rows:
        with (ROOT/'results'/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    e1,e2,ref,weak=[read(x) for x in ['e1','e2','reference','weak']]
    assert [len(x) for x in [e1,e2,ref,weak]]==[252,1800,12,36]
    assert len({(x['n'],x['rep'],x['tau'],x['delta']) for x in e2})==1800
    assert max(x['residual'] for x in e1+e2+weak)<1e-9
    d={(x['grid'],x['a'],x['b'],x['tau'],x['delta']):x for x in e1}
    responses=[];griderr=[]
    for a in [0.,.3,.6]:
        for b in [0.,.25,.5]:
            for tau in [.02,.08]:
                base=d[512,a,b,tau,0.]
                for delta in [.005,.02,.1,.25,.5,1.]:
                    v=d[512,a,b,tau,delta];lo=d[256,a,b,tau,delta]
                    responses.append(dict(a=a,b=b,tau=tau,delta=delta,third= v['coefficients'][3],baseline_third=base['coefficients'][3],third_magnitude_ratio=abs(v['coefficients'][3]/base['coefficients'][3]),third_sign_reversal=bool(v['coefficients'][3]*base['coefficients'][3]<0),fundamental_ratio=v['coefficients'][1]/base['coefficients'][1],rkhs_norm_ratio=v['rkhs_norm']/base['rkhs_norm'],baseline_linear_third=base['linear_coefficients'][3]))
                    griderr.append(max(abs(np.array(v['coefficients'])-lo['coefficients'])))
    save('e1_response',responses)
    rd={(x['order'],x['tau'],x['delta']):x for x in ref};stats=[]
    for n in [128,256,512]:
        for tau in [.02,.08]:
            rows=sorted([x for x in e2 if x['n']==n and x['tau']==tau],key=lambda x:x['rep'])
            base=np.array([x['scaled_projections'] for x in rows if x['delta']==0.])
            rng=np.random.RandomState(57000+n+int(tau*1000));idx=rng.randint(0,100,size=(2000,100))
            for delta in [.5,1.]:
                arr=np.array([x['scaled_projections'] for x in rows if x['delta']==delta]);cov=np.cov(arr.T,ddof=1);cov0=np.cov(base.T,ddof=1)
                boot=arr[idx].var(axis=1,ddof=1)/base[idx].var(axis=1,ddof=1)
                theo=np.array(rd[32,tau,delta]['covariance']);theo0=np.array(rd[32,tau,0.]['covariance'])
                for k,label in enumerate(['constant','x1','x2']):
                    ci=np.percentile(boot[:,k],[2.5,97.5]);r=cov[k,k]/cov0[k,k];tr=theo[k,k]/theo0[k,k]
                    stats.append(dict(n=n,tau=tau,delta=delta,projection=label,variance=float(cov[k,k]),reference_variance=float(theo[k,k]),ratio=float(r),ci_low=float(ci[0]),ci_high=float(ci[1]),reference_ratio=float(tr),reference_within_ci=bool(ci[0]<=tr<=ci[1])))
    save('e2_covariance',stats)
    referr=max(float(np.max(np.abs(np.array(rd[24,t,d]['covariance'])-rd[32,t,d]['covariance']))/np.max(np.abs(rd[32,t,d]['covariance']))) for t in [.02,.08] for d in [0.,.5,1.])
    wd={(x['order'],x['tau'],x['delta'],x['epsilon']):x for x in weak}
    weakgrid=max(float(np.max(np.abs(np.array(x['projections'])-wd[24,x['tau'],x['delta'],x['epsilon']]['projections']))) for x in weak if x['order']==32)
    weakslopes=[]
    for t in [.02,.08]:
        for delta in [0.,.5,1.]:
            rr=[wd[32,t,delta,eps] for eps in [.05,.1,.2]]
            weakslopes.append(dict(tau=t,delta=delta,l2_error_slope=float(np.polyfit(np.log([.05,.1,.2]),np.log([x['l2_error'] for x in rr]),1)[0]),relative_error_min=min(x['relative_l2_error'] for x in rr),relative_error_max=max(x['relative_l2_error'] for x in rr)))
    save('weak_signal',weakslopes)
    initial=[x for x in responses if x['delta']==.02];late=[x for x in responses if x['delta']==1.]
    summary=dict(counts=dict(e1=252,e2=1800,reference=12,weak=36),max_residual=max(x['residual'] for x in e1+e2+weak),max_distance_bound=max(x['distance_bound'] for x in e1+e2+weak),e1_max_fourier_grid_difference=max(griderr),reference_max_relative_covariance_difference=referr,weak_max_projection_grid_difference=weakgrid,e1_initial_attenuation_count=sum(x['third_magnitude_ratio']<1 for x in initial),e1_initial_amplification_count=sum(x['third_magnitude_ratio']>1 for x in initial),e1_initial_total=len(initial),e1_large_radius_sign_reversals=sum(x['third_sign_reversal'] for x in late),e2_ratio_ci_excludes_one=sum(x['ci_low']>1 for x in stats),e2_total_ratios=len(stats),e2_reference_ratio_outside_ci=sum(not x['reference_within_ci'] for x in stats),weak_error_slopes=weakslopes)
    (ROOT/'results/summary.json').write_text(json.dumps(summary,indent=2))
    plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,ax=plt.subplots(1,3,figsize=(7.15,2.45),gridspec_kw={'width_ratios':[1.14,1,1]})
    mat=np.array([[next(x['third_magnitude_ratio'] for x in initial if x['a']==a and x['tau']==t and x['b']==b) for b in [0.,.25,.5]] for t in [.02,.08] for a in [0.,.3,.6]])
    ax[0].imshow(mat,cmap='RdBu_r',norm=Normalize(vmin=.3,vmax=1.7),aspect='auto')
    for i in range(6):
        for j in range(3):ax[0].text(j,i,'{:.3f}'.format(mat[i,j]),ha='center',va='center',fontsize=7,color='white' if mat[i,j]<.5 else '#222222')
    ax[0].set(xticks=range(3),xticklabels=['0','.25','.5'],yticks=range(6),yticklabels=['.02, 0','.02, .3','.02, .6','.08, 0','.08, .3','.08, .6'],xlabel=r'Signal mixture $b$',ylabel=r'$(\tau,a)$',title='(a) Third-mode magnitude ratio')
    colors=['#0072B2','#D55E00']
    for t,col,mark in zip([.02,.08],colors,['o','s']):
        rr=[next(x for x in stats if x['n']==n and x['tau']==t and x['delta']==1. and x['projection']=='constant') for n in [128,256,512]]
        vals=np.array([x['ratio'] for x in rr]);err=np.array([[x['ratio']-x['ci_low'] for x in rr],[x['ci_high']-x['ratio'] for x in rr]])
        ax[1].errorbar([128,256,512],vals,yerr=err,color=col,marker=mark,linewidth=1.1,capsize=2,label=r'$\tau={}$'.format(t))
        ax[1].axhline(rr[0]['reference_ratio'],color=col,ls='--',lw=1)
        rr=[wd[32,t,1.,eps] for eps in [.05,.1,.2]]
        ax[2].loglog([.05,.1,.2],[x['l2_error'] for x in rr],color=col,marker=mark,lw=1.1,label=r'$\tau={}$'.format(t))
    ax[1].set(xlabel=r'Sample size $n$',ylabel='Projection variance ratio',title='(b) Balanced RBF fits',xticks=[128,256,512])
    ax[1].legend(frameon=False);ax[1].grid(alpha=.2)
    ax[2].set(xlabel=r'Signal strength $\epsilon$',ylabel=r'$L^2(P_X)$ error',title='(c) Weak-signal RBF fits',xticks=[.05,.1,.2],xticklabels=['.05','.1','.2'])
    ax[2].xaxis.set_minor_formatter(NullFormatter())
    ax[2].legend(frameon=False);ax[2].grid(alpha=.2)
    fig.tight_layout(pad=.4,w_pad=1.1)
    for ext in ['pdf','svg','png']:fig.savefig(ROOT/'figures'/('generalization.'+ext),dpi=240,bbox_inches='tight')
    plt.close(fig)
    # All E1 radius paths, unselected across the full factorial design.
    fig,axes=plt.subplots(2,3,figsize=(7.15,4.15),sharex=True)
    for ti,t in enumerate([.02,.08]):
        for ai,a in enumerate([0.,.3,.6]):
            aa=axes[ti,ai]
            for b,col,mark in zip([0.,.25,.5],['#0072B2','#D55E00','#009E73'],['o','s','^']):
                rr=[d[512,a,b,t,delta] for delta in [0.,.005,.02,.1,.25,.5,1.]]
                aa.plot([x['delta'] for x in rr],[x['coefficients'][3] for x in rr],color=col,marker=mark,ms=3,lw=1,label=r'$b={}$'.format(b))
            aa.axhline(0,color='#777777',lw=.6);aa.set_title(r'$a={},\ \tau={}$'.format(a,t));aa.ticklabel_format(axis='y',style='sci',scilimits=(0,0));aa.grid(alpha=.15)
            if ti==1:aa.set_xlabel(r'Radius $\delta$')
            if ai==0:aa.set_ylabel('Signed third coefficient')
    axes[0,0].legend(frameon=False);fig.tight_layout()
    for ext in ['pdf','svg','png']:fig.savefig(ROOT/'figures'/('e1_all_paths.'+ext),dpi=220,bbox_inches='tight')
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
