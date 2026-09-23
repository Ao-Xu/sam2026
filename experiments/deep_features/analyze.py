"""Analyze every declared setting; intervals resample paired label/bootstrap repeats."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
def interval_ratio(a,b,indices):
 r=a[indices].mean(axis=1)/b[indices].mean(axis=1)
 return np.quantile(r,[.025,.975]).tolist()
def main():
 out=R/'results';out.mkdir(exist_ok=True)
 summary={};signal_rows=[];noise_rows=[];real_rows=[]
 fig,axes=plt.subplots(2,2,figsize=(7.0,5.1),constrained_layout=True)
 colors=['#315B83','#B44948','#598B69'];markers=['o','s','^']
 for mi,name in enumerate(['resnet18','vit_b_16']):
  sig=np.load(R/'raw'/f'{name}_signal.npz');noise=np.load(R/'raw'/f'{name}_noise.npz');radii=sig['radii'];eps=sig['epsilon']
  fit=sig['fit'];ref=sig['reference'];probes=sig['probes']
  norms=np.linalg.norm(fit-ref,axis=1)/np.linalg.norm(ref,axis=1)
  sc=np.einsum('dp,rdk->rpk',probes,fit);sr=np.einsum('dp,rdk->rpk',probes,ref)
  gainerrors=[]
  for ei,e in enumerate(eps):
   for j,rank in enumerate([1,8,32]):
    k=3*ei+j
    for di,delta in enumerate(radii):
     pred=sr[di,j,k]/sr[0,j,k];obs=sc[di,j,k]/sc[0,j,k]
     signal_rows.append(dict(model=name,epsilon=float(e),rank=rank,radius=float(delta),predicted_gain=float(pred),observed_gain=float(obs),relative_vector_error=float(norms[di,k])))
     if di:gainerrors.append(abs(obs/pred-1))
    # All eps plotted with marker sizes; no selection of better-fitting eps.
    axes[mi,0].plot(sr[1:,j,k]/sr[0,j,k],sc[1:,j,k]/sc[0,j,k],linestyle='none',marker=markers[j],color=colors[j],ms=3+ei,alpha=.7)
  w=noise['fit'];lin=noise['reference'];p=noise['probes'];n=len(noise['subset'])
  c=np.einsum('dp,rdb->rpb',p,w);cl=np.einsum('dp,rdb->rpb',p,lin)
  moments=n*c*c;lm=n*cl*cl;cond=noise['conditional'];rng=np.random.default_rng(9142200);boot=rng.integers(100,size=(2000,100));errs=[];cis=[];nonlin=[]
  for j,rank in enumerate([1,8,32]):
   pred=cond[:,j]/cond[0,j];obs=moments[:,j].mean(axis=1)/moments[0,j].mean()
   lo=[];hi=[]
   for di,delta in enumerate(radii):
    ci=interval_ratio(moments[di,j],moments[0,j],boot)
    centered=n*np.var(c[di,j],ddof=1)
    nl=(moments[di,j]-lm[di,j]).mean()/cond[di,j]
    noise_rows.append(dict(model=name,rank=rank,radius=float(delta),predicted_ratio=float(pred[di]),observed_ratio=float(obs[di]),ci_low=ci[0],ci_high=ci[1],second_moment=float(moments[di,j].mean()),centered_variance=float(centered),conditional_reference=float(cond[di,j]),realized_linearized_second_moment=float(lm[di,j].mean()),nonlinear_bias_relative=float(nl)))
    lo.append(obs[di]-ci[0]);hi.append(ci[1]-obs[di]);nonlin.append(abs(nl))
    if di:errs.append(abs(obs[di]/pred[di]-1));cis.append(ci[0]>1)
   axes[mi,1].errorbar(pred[1:],obs[1:],yerr=[lo[1:],hi[1:]],fmt=markers[j],color=colors[j],ms=4,capsize=2,label=f'Rank {rank}')
  for ax in axes[mi]:
   ax.set_xlabel('Predicted gain');ax.set_ylabel('Observed gain');ax.grid(alpha=.15)
   lim=ax.get_xlim();yl=ax.get_ylim();low=min(lim[0],yl[0]);high=max(lim[1],yl[1]);ax.plot([low,high],[low,high],color='.5',lw=.8,ls='--',zorder=0)
  title='ResNet-18' if mi==0 else 'ViT-B/16'
  axes[mi,0].set_title(title+': weak signal');axes[mi,1].set_title(title+': label noise')
  summary[name]={'signal_max_relative_gain_error':float(max(gainerrors)),'signal_max_relative_vector_error':float(norms.max()),'noise_max_relative_ratio_error':float(max(errs)),'noise_positive_ratio_intervals_above_one':int(sum(cis)),'noise_positive_comparisons':len(cis),'noise_max_absolute_relative_nonlinear_bias':float(max(nonlin))}
  for pair in ['3_5','1_9','0_8']:
   real=np.load(R/'raw'/f'{name}_real_{pair}.npz');scores=real['scores'];y=real['test_labels'];bs=np.random.default_rng(9142400).integers(30,size=(2000,30))
   for di,delta in enumerate(radii):
    loss=np.logaddexp(0,-y[:,None]*scores[di]).mean(axis=0);acc=((scores[di]>=0)==(y[:,None]>0)).mean(axis=0)
    var=np.var(scores[di],axis=1,ddof=1).mean()
    lci=np.quantile(loss[bs].mean(axis=1),[.025,.975]);aci=np.quantile(acc[bs].mean(axis=1),[.025,.975])
    # Bootstrap resampling variance is descriptive conditional on a fixed training pool and test set.
    vboot=np.array([np.var(scores[di][:,ii],axis=1,ddof=1).mean() for ii in bs]);vci=np.quantile(vboot,[.025,.975])
    real_rows.append(dict(model=name,pair=pair,radius=float(delta),logloss=float(loss.mean()),logloss_ci_low=float(lci[0]),logloss_ci_high=float(lci[1]),accuracy=float(acc.mean()),accuracy_ci_low=float(aci[0]),accuracy_ci_high=float(aci[1]),score_variance=float(var),variance_ci_low=float(vci[0]),variance_ci_high=float(vci[1])))
 axes[0,1].legend(fontsize=7,loc='upper left')
 fig.savefig(out/'deep_transfer.pdf');fig.savefig(out/'deep_transfer.png',dpi=220);fig.savefig(out/'deep_transfer.svg');plt.close(fig)
 compact,axs=plt.subplots(1,2,figsize=(7,2.3),constrained_layout=True)
 for mi,name in enumerate(['resnet18','vit_b_16']):
  col=['#315B83','#B44948'][mi];label=['ResNet-18','ViT-B/16'][mi]
  sr=[r for r in signal_rows if r['model']==name and r['radius']>0]
  nr=[r for r in noise_rows if r['model']==name and r['radius']>0]
  for rank,marker in zip([1,8,32],['o','s','^']):
   rows=[r for r in sr if r['rank']==rank]
   axs[0].scatter([r['predicted_gain'] for r in rows],[r['observed_gain'] for r in rows],marker=marker,s=14,color=col,alpha=.65,label=label if rank==1 else None)
   rows=[r for r in nr if r['rank']==rank];obs=np.array([r['observed_ratio'] for r in rows])
   axs[1].errorbar([r['predicted_ratio'] for r in rows],obs,yerr=[obs-np.array([r['ci_low'] for r in rows]),np.array([r['ci_high'] for r in rows])-obs],fmt=marker,ms=3.5,color=col,capsize=2)
 for ax,title in zip(axs,['(a) Weak-signal gain','(b) Label-noise variance ratio']):
  lo=min(ax.get_xlim()[0],ax.get_ylim()[0]);hi=max(ax.get_xlim()[1],ax.get_ylim()[1]);ax.plot([lo,hi],[lo,hi],ls='--',lw=.8,color='.5');ax.set_xlabel('Predicted');ax.set_ylabel('Observed');ax.set_title(title);ax.grid(alpha=.15)
 axs[0].legend(fontsize=8,loc='upper left')
 compact.savefig(out/'deep_transfer_main.pdf');compact.savefig(out/'deep_transfer_main.png',dpi=240);compact.savefig(out/'deep_transfer_main.svg');plt.close(compact)
 for filename,rows in [('signal.csv',signal_rows),('noise.csv',noise_rows),('real.csv',real_rows)]:
  with (out/filename).open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 summary['counts']={'signal_fits':72,'noise_fits':800,'real_fits':720,'total_optimized_heads':1592,'independent_noise_label_vectors_per_model':100,'real_bootstrap_replicates_per_task_model':30}
 (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
 print('REAL ENDPOINTS')
 for row in real_rows:
  if row['radius'] in [0,1]:print(row)
if __name__=='__main__':main()
