from pathlib import Path
import json,csv
import numpy as np
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator,FixedLocator,FixedFormatter
R=Path(__file__).resolve().parent;OUT=R/'results';OUT.mkdir(exist_ok=True)
FIG=R/'figures'
FIG.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
def savecsv(name,rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
def main():
 rows=json.loads((R/'raw/population.json').read_text());savecsv('population.csv',rows)
 pop=[r for r in rows if r['grid']==1024];path=sorted([r for r in pop if r['epsilon']==.1],key=lambda r:r['delta']);sc=sorted([r for r in pop if r['delta']==.25],key=lambda r:r['epsilon'])
 slopes={k:float(np.polyfit(np.log([r['epsilon'] for r in sc]),np.log([r[k] for r in sc]),1)[0]) for k in ['absolute','relative']}
 quadrature=max(abs(r['c3']-next(t['c3'] for t in rows if t['grid']==512 and t['epsilon']==r['epsilon'] and t['delta']==r['delta'])) for r in pop)
 fig,axs=plt.subplots(1,3,figsize=(7.1,2.4));colors=['#315B83','#B44948']
 for field,label,col,marker in [('c1','Samplewise',colors[0],'o'),('matched_c1','Matched logistic',colors[1],'s')]:
  axs[0].plot([r['delta'] for r in path],[r[field]/path[0][field] for r in path],marker=marker,color=col,lw=1.1,ms=3,label=label)
 axs[0].set(xlabel='Radius',ylabel='Fundamental / radius-zero value',title='(a) Whole-radius signal');axs[0].legend(fontsize=7,loc='upper left')
 local=[r for r in path if r['delta']<=.25]
 for field,col,marker in [('c3',colors[0],'o'),('matched_c3',colors[1],'s')]:axs[1].plot([r['delta'] for r in local],[r[field]*1e6 for r in local],marker=marker,color=col,lw=1.1,ms=3)
 axs[1].set(xlabel='Radius',ylabel='Third coefficient ($10^{-6}$)',title='(b) Local structural separation')
 for k,col,marker in [('absolute',colors[0],'o'),('relative',colors[1],'s')]:axs[2].loglog([r['epsilon'] for r in sc],[r[k] for r in sc],marker=marker,color=col,ms=3,label=f'{k.capitalize()}: slope {slopes[k]:.2f}')
 axs[2].set(xlabel='Signal amplitude',ylabel='Prediction difference',title='(c) Matching error at radius .25');axs[2].legend(fontsize=7,loc='lower right')
 axs[2].xaxis.set_major_locator(FixedLocator([.025,.05,.1,.2]));axs[2].xaxis.set_major_formatter(FixedFormatter(['.025','.05','.1','.2']));axs[2].xaxis.set_minor_locator(NullLocator())
 for ax in axs:ax.grid(alpha=.15)
 fig.tight_layout(w_pad=1.0);fig.savefig(FIG/'matched_spectral.pdf');(OUT/'matched_spectral.pdf').write_bytes((FIG/'matched_spectral.pdf').read_bytes());fig.savefig(OUT/'matched_spectral.png',dpi=180);plt.close(fig)
 fig,ax=plt.subplots(figsize=(4.5,2.5))
 for field,label,col in [('c3','Samplewise',colors[0]),('matched_c3','Matched logistic',colors[1])]:ax.plot([r['delta'] for r in path],[r[field]*1e6 for r in path],'o-',label=label,color=col,ms=3)
 ax.set(xlabel='Radius',ylabel='Third coefficient ($10^{-6}$)');ax.legend();ax.grid(alpha=.15);fig.tight_layout();fig.savefig(FIG/'matched_full_path.pdf');(OUT/'matched_full_path.pdf').write_bytes((FIG/'matched_full_path.pdf').read_bytes());plt.close(fig)
 real=[];weak=[];rng=np.random.default_rng(9143200);ix=rng.integers(30,size=(2000,30))
 for name in ['resnet18','vit_b_16']:
  sg=np.load(R/'raw'/f'{name}_signal.npz');rel=np.linalg.norm(sg['matched']-sg['original'],axis=1)/np.linalg.norm(sg['original'],axis=1)
  for di,delta in enumerate(sg['radii']):
   for ei,e in enumerate(sg['epsilon']):
    for j,rank in enumerate([1,8,32]):weak.append(dict(model=name,radius=float(delta),epsilon=float(e),rank=rank,relative_parameter_difference=float(rel[di,ei*3+j])))
  for a,b in [(3,5),(1,9),(0,8)]:
   z=np.load(R/'raw'/f'{name}_real_{a}_{b}.npz');scores=z['scores'];probs=expit(scores);y=z['test_labels'];truth=(y+1)/2;methods=z['methods'];loss=np.logaddexp(0,-scores*y[None,:,None]).mean(axis=1)
   for mi,method in enumerate(methods):
    diff=loss[mi]-loss[1];ci=np.quantile(diff[ix].mean(axis=1),[.025,.975]);cfg=json.loads((R/'raw'/f'{name}_real_{a}_{b}.json').read_text())['records'][mi]
    real.append(dict(model=name,pair=f'{a}/{b}',method=str(method),tau=cfg['tau'],scale=cfg['scale'],log_loss=float(loss[mi].mean()),brier=float(((probs[mi]-truth[:,None])**2).mean()),accuracy=float(((scores[mi]>0)==(y[:,None]>0)).mean()),score_variance=float(scores[mi].var(axis=1,ddof=1).mean()),probability_variance=float(probs[mi].var(axis=1,ddof=1).mean()),probability_rms_difference=float(np.sqrt(((probs[mi]-probs[1])**2).mean())),disagreement=float(((scores[mi]>0)!=(scores[1]>0)).mean()),loss_difference=float(diff.mean()),difference_ci_low=float(ci[0]),difference_ci_high=float(ci[1])))
 savecsv('real.csv',real);savecsv('weak_matching.csv',weak)
 summary=dict(slopes=slopes,max_quadrature_c3_change=quadrature,periodic_epsilon_point_one=path,max_population_residual=max(r['residual'] for r in rows),deep_max_relative_difference={n:max(r['relative_parameter_difference'] for r in weak if r['model']==n and r['radius']>0) for n in ['resnet18','vit_b_16']},real=real)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 lines=[r'\begin{tabular}{llrrrrr}',r'\toprule',r'Features & Task & ERM & Samplewise & Matched & Tuned & Calibrated\\',r'\midrule']
 labels={'3/5':'Cat/dog','1/9':'Car/truck','0/8':'Plane/ship'}
 for n in ['resnet18','vit_b_16']:
  for pair in ['3/5','1/9','0/8']:
   subset=[r for r in real if r['model']==n and r['pair']==pair]
   lines.append(('ResNet-18' if n=='resnet18' else 'ViT-B/16')+' & '+labels[pair]+' & '+' & '.join(f"{r['log_loss']:.3f}" for r in subset)+r'\\')
 lines.extend([r'\bottomrule',r'\end{tabular}']);(OUT/'main_table.tex').write_text('\n'.join(lines))
 print(json.dumps({k:v for k,v in summary.items() if k not in ['real','periodic_epsilon_point_one']},indent=2));print('\n'.join(lines))
if __name__=='__main__':main()
