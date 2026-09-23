import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def read(path):return [json.loads(x) for x in path.read_text().splitlines()]
def save(name,rows):
    (ROOT/'results'/(name+'.json')).write_text(json.dumps(rows,indent=2))
    with (ROOT/'results'/(name+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def interval(v,ix):return [float(x) for x in np.percentile(v[ix].mean(axis=1),[2.5,97.5])]
rbf=read(ROOT/'raw/rbf.jsonl');bw=read(ROOT/'raw/bandwidth.jsonl');real=read(ROOT/'raw/real.jsonl')
assert len(rbf)==3000 and len(bw)==400 and len(real)==600
assert max(r['residual'] for r in rbf+bw+real)<1e-9
ref=read(ROOT.parent/'generalization/raw/reference.jsonl');bref=read(ROOT/'raw/bandwidth_reference.jsonl')
rng=np.random.RandomState(881133);ix=rng.randint(100,size=(2000,100));rows=[]
for dataset,bands in [(rbf,[.7]),(bw,[.4,1.2])]:
    for b in bands:
        for n,tau,delta in sorted(set((r['n'],r['tau'],r['delta']) for r in dataset if r.get('bandwidth',.7)==b)):
            rr=sorted([r for r in dataset if (r['n'],r['tau'],r['delta'],r.get('bandwidth',.7))==(n,tau,delta,b)],key=lambda r:r['rep']);assert len(rr)==100
            R=next(r for r in (ref if b==.7 else bref) if r['order']==32 and (r.get('tau',.02),r.get('delta',1.),r.get('bandwidth',.7))==(tau,delta,b))
            true=np.array([r['true'] for r in rr]);lin=np.array([r['linear'] for r in rr]);cond=np.array([np.diag(r['conditional']) for r in rr]);limit=np.diag(R['covariance'])
            for j in range(3):
                nonlin=true[:,j]**2-lin[:,j]**2;design=cond[:,j]-limit[j];mc=lin[:,j]**2-cond[:,j];cv=cond[:,j]+nonlin
                row=dict(bandwidth=b,n=n,tau=tau,delta=delta,probe=j,population=float(limit[j]),true_second_moment=float(np.mean(true[:,j]**2)),centered_true_variance=float(np.var(true[:,j],ddof=1)),conditional_mean=float(cond[:,j].mean()),nonlinear_bias=float(nonlin.mean()),design_bias=float(design.mean()),label_mc=float(mc.mean()),control_variate_covariance=float(cv.mean()),remainder_rms=float(np.sqrt(np.mean([r['rkhs_remainder']**2 for r in rr]))))
                for name,v in [('nonlinear',nonlin),('design',design),('control_variate',cv),('true_second',true[:,j]**2)]:
                    ci=interval(v,ix);row[name+'_ci_low'],row[name+'_ci_high']=ci
                assert abs(row['true_second_moment']-limit[j]-row['nonlinear_bias']-row['design_bias']-row['label_mc'])<1e-10
                rows.append(row)
save('decomposition',rows)
ratios=[]
for n in [128,256,512,1024,2048]:
    for tau in [.02,.08]:
        base=sorted([r for r in rbf if (r['n'],r['tau'],r['delta'])==(n,tau,0.)],key=lambda r:r['rep'])
        for delta in [.5,1.]:
            rr=sorted([r for r in rbf if (r['n'],r['tau'],r['delta'])==(n,tau,delta)],key=lambda r:r['rep'])
            for j in range(3):
                a=np.array([r['true'][j]**2 for r in rr]);b=np.array([r['true'][j]**2 for r in base]);c=np.array([r['conditional'][j][j] for r in rr]);d=np.array([r['conditional'][j][j] for r in base])
                va=c+a-np.array([r['linear'][j]**2 for r in rr]);vb=d+b-np.array([r['linear'][j]**2 for r in base]);ci=np.percentile(a[ix].mean(axis=1)/b[ix].mean(axis=1),[2.5,97.5])
                num=next(r for r in rows if (r['bandwidth'],r['n'],r['tau'],r['delta'],r['probe'])==(.7,n,tau,delta,j))['population'];den=next(r for r in rows if (r['bandwidth'],r['n'],r['tau'],r['delta'],r['probe'])==(.7,n,tau,0.,j))['population']
                ratios.append(dict(n=n,tau=tau,delta=delta,probe=j,true_ratio=float(a.mean()/b.mean()),true_ratio_ci_low=float(ci[0]),true_ratio_ci_high=float(ci[1]),conditional_ratio=float(c.mean()/d.mean()),control_variate_ratio=float(va.mean()/vb.mean()),limiting_ratio=num/den))
save('covariance_ratios',ratios)
realrows=[]
for mode in ['true','randomized']:
    for delta in [0.,.5,1.]:
        rr=sorted([r for r in real if r['mode']==mode and r['delta']==delta],key=lambda r:r['rep']);F=np.array([r['scores'] for r in rr]);L=np.array([r['linear_scores'] for r in rr]);n=rr[0]['n']
        row=dict(mode=mode,delta=delta,n=n,test_n=F.shape[1],mean_accuracy=float(np.mean([r['accuracy'] for r in rr])),mean_logloss=float(np.mean([r['logloss'] for r in rr])),mean_brier=float(np.mean([r['brier'] for r in rr])),mean_prediction_variance=float(np.var(F,axis=0,ddof=1).mean()),scaled_true_second=float(n*np.mean(F*F)),scaled_linear_second=float(n*np.mean(L*L)),conditional_mean=float(np.mean([r['conditional_diagonal'] for r in rr])))
        for metric in ['accuracy','logloss','brier']:
            v=np.array([r[metric] for r in rr]);row[metric+'_resampling_sd']=float(np.std(v,ddof=1))
        if mode=='randomized':
            dif=n*np.mean(F*F-L*L,axis=1);row['nonlinear_bias']=float(dif.mean());row['nonlinear_ci_low'],row['nonlinear_ci_high']=interval(dif,ix)
        else:row.update(nonlinear_bias='',nonlinear_ci_low='',nonlinear_ci_high='')
        realrows.append(row)
save('real_summary',realrows)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(6.7,2.55),sharex=True)
for ax,tau in zip(axs,[.02,.08]):
    rr=[r for r in rows if r['bandwidth']==.7 and r['tau']==tau and r['delta']==1 and r['probe']==0];ns=np.array([r['n'] for r in rr]);lim=rr[0]['population']
    for key,label,color,marker in [('nonlinear','Nonlinear bias','#c95a36','o'),('design','Random-design bias','#237fa4','s')]:
        v=np.array([r[key+'_bias'] for r in rr])/lim;lo=np.array([r[key+'_ci_low'] for r in rr])/lim;hi=np.array([r[key+'_ci_high'] for r in rr])/lim
        ax.plot(ns,v,marker=marker,color=color,label=label);ax.fill_between(ns,lo,hi,color=color,alpha=.17)
    ax.axhline(0,color='.55',lw=.7);ax.set_xscale('log',basex=2);ax.set_xticks(ns);ax.set_xticklabels(ns);ax.set_xlabel('Sample size $n$');ax.set_title(r'$\tau={}$, $\delta=1$'.format(tau));ax.grid(alpha=.18)
axs[0].set_ylabel('Bias / limiting probe variance');axs[0].legend(frameon=False,fontsize=8,loc='lower right')
fig.tight_layout();fig.savefig(str(ROOT/'figures/finite_sample_decomposition.pdf'),bbox_inches='tight');fig.savefig(str(ROOT/'figures/finite_sample_decomposition.png'),dpi=220,bbox_inches='tight');plt.close(fig)
qa=dict(fits=len(rbf)+len(bw)+len(real),rbf_fits=len(rbf),bandwidth_fits=len(bw),real_fits=len(real),max_gradient=max(r['residual'] for r in rbf+bw+real),max_full_gradient=max(r['full_residual_bound'] for r in rbf+bw),max_distance_bound=max(r['distance_bound'] for r in rbf+bw),bootstrap_repeats=2000)
(ROOT/'results/qa.json').write_text(json.dumps(qa,indent=2));print(json.dumps(qa,indent=2))


