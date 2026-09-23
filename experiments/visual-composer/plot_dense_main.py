from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

R=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'results';OUT.mkdir(exist_ok=True)
PAPER=Path(__file__).resolve().parent/'figures'
PAPER.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
 'font.size':8,'axes.titlesize':8.5,'axes.labelsize':8,'legend.fontsize':6.5,
 'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';PURPLE='#CC79A7';GRAY='#666666'

def save(fig,name):
 fig.tight_layout(w_pad=0.8,h_pad=0.8)
 fig.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
 fig.savefig(OUT/f'{name}.png',dpi=200,bbox_inches='tight')
 (PAPER/f'{name}.pdf').write_bytes((OUT/f'{name}.pdf').read_bytes())
 plt.close(fig)

def figure2():
 rows=json.loads((R/'experiments/matching/raw/population.json').read_text())
 rows=[r for r in rows if r['grid']==1024]
 eps=sorted({r['epsilon'] for r in rows});ds=sorted({r['delta'] for r in rows})
 cmap=plt.get_cmap('viridis');cols=[cmap(x) for x in np.linspace(.12,.88,len(eps))]
 fig,ax=plt.subplots(2,2,figsize=(7.15,4.25))
 for e,col in zip(eps,cols):
  rr=sorted([r for r in rows if r['epsilon']==e],key=lambda z:z['delta'])
  ax[0,0].plot(ds,[r['c1'] for r in rr],'-o',color=col,ms=2.7,lw=1,label=fr'$\epsilon={e:g}$')
  ax[0,0].plot(ds,[r['matched_c1'] for r in rr],'--',color=col,lw=1)
  loc=[r for r in rr if r['delta']<=.25]
  ax[0,1].plot([r['delta'] for r in loc],[r['c3'] for r in loc],'-o',color=col,ms=2.5,lw=1)
  ax[0,1].plot([r['delta'] for r in loc],[r['matched_c3'] for r in loc],'--s',color=col,ms=2.2,lw=1)
 ax[0,0].plot([],[],'-',color='k',label='Samplewise');ax[0,0].plot([],[],'--',color='k',label='Matched')
 ax[0,0].legend(ncol=2,frameon=False,loc='upper left',columnspacing=.8,handlelength=1.5)
 ax[0,0].set(title='(a) Fundamental amplitude',xlabel='Radius',ylabel=r'$c_1$')
 ax[0,1].set_yscale('log')
 ax[0,1].set(title='(b) Local third-mode amplitude',xlabel='Radius',ylabel=r'$c_3$ (log scale)')
 # Collapse the four normalized curves into a mean path and an observed range.
 for key,col,label,marker in [('c3',BLUE,'Samplewise','o'),('matched_c3',ORANGE,'Matched','s')]:
  vals=np.array([[next(r[key] for r in rows if r['epsilon']==e and r['delta']==d)/e**3 for d in ds] for e in eps])
  ax[1,0].fill_between(ds,vals.min(0),vals.max(0),color=col,alpha=.16,lw=0)
  ax[1,0].plot(ds,vals.mean(0),color=col,marker=marker,ms=2.8,lw=1.2,label=label)
 ax[1,0].axhline(0,color='.55',ls=':',lw=.8);ax[1,0].set(title='(c) Cubic collapse and sign reversal',xlabel='Radius',ylabel=r'$c_3 / \epsilon^3$')
 ax[1,0].legend(frameon=False,loc='lower left')
 # Relative error (%) matrix, exclude delta zero where the exact value is zero.
 mat=np.array([[next(r['relative'] for r in rows if r['epsilon']==e and r['delta']==d)*100 for d in ds[1:]] for e in eps])
 im=ax[1,1].imshow(mat,aspect='auto',origin='lower',cmap='magma',norm=LogNorm(vmin=max(mat.min(),1e-6),vmax=mat.max()))
 ax[1,1].set(title='(d) Complete-function mismatch (%)',xlabel='Radius',ylabel='Signal amplitude',xticks=range(len(ds)-1),xticklabels=[f'{d:g}' for d in ds[1:]],yticks=range(len(eps)),yticklabels=[f'{e:g}' for e in eps])
 for i in range(len(eps)):
  for j in range(len(ds)-1):
   v=mat[i,j];ax[1,1].text(j,i,f'{v:.3g}',ha='center',va='center',fontsize=5.8,color='black' if v>np.sqrt(mat.min()*mat.max()) else 'white')
 cb=fig.colorbar(im,ax=ax[1,1],fraction=.047,pad=.02);cb.ax.set_ylabel('Relative $L^2$ difference (%)',fontsize=7);cb.ax.tick_params(labelsize=6)
 for a in ax.flat:a.grid(alpha=.14)
 ax[1,1].grid(False)
 save(fig,'matched_spectral_dense')

def read_csv(path):
 with path.open(newline='') as f:return list(csv.DictReader(f))

def figure3():
 sig=read_csv(R/'experiments/deep_features/results/signal.csv');noi=read_csv(R/'experiments/deep_features/results/noise.csv');real=read_csv(R/'experiments/matching/results/real.csv')
 fig,ax=plt.subplots(2,3,figsize=(7.15,4.75))
 rank_cols={1:BLUE,8:ORANGE,32:GREEN}
 methods=['ERM','Matched','Tuned','Calibrated'];mcols={'ERM':GRAY,'Matched':ORANGE,'Tuned':GREEN,'Calibrated':PURPLE};mmark={'ERM':'o','Matched':'D','Tuned':'^','Calibrated':'v'}
 tasks=['3/5','1/9','0/8'];task_labels=['Cat/dog','Car/truck','Plane/ship']
 ranks=[1,8,32];epsilons=[.05,.1,.2];radii=[.25,.5,1.]
 signal_images=[]
 for row,name in enumerate(['resnet18','vit_b_16']):
  ss=[r for r in sig if r['model']==name and float(r['radius'])>0]
  smat=np.abs(np.array([[100*(float(next(r['observed_gain'] for r in ss if int(r['rank'])==rank and float(r['epsilon'])==e and float(r['radius'])==d))/float(next(r['predicted_gain'] for r in ss if int(r['rank'])==rank and float(r['epsilon'])==e and float(r['radius'])==d))-1) for d in radii] for rank in ranks for e in epsilons]))
  sim=ax[row,0].imshow(smat,aspect='auto',cmap='magma',norm=LogNorm(vmin=1e-5,vmax=2))
  signal_images.append(sim)
  for i in range(smat.shape[0]):
   for j in range(smat.shape[1]):
    v=smat[i,j]; label=f'{v:.2f}' if v>=.01 else f'{v:.1e}'
    ax[row,0].text(j,i,label,ha='center',va='center',fontsize=5.3,color='black' if v>.08 else 'white')
  nn=[r for r in noi if r['model']==name and float(r['radius'])>0]
  for rank in ranks:
   rr=[next(r for r in nn if int(r['rank'])==rank and float(r['radius'])==d) for d in radii]
   pred=np.array([float(r['predicted_ratio']) for r in rr]);obs=np.array([float(r['observed_ratio']) for r in rr])
   lo=np.array([float(r['ci_low']) for r in rr]);hi=np.array([float(r['ci_high']) for r in rr])
   resid=100*(obs/pred-1);resid_lo=100*(lo/pred-1);resid_hi=100*(hi/pred-1)
   ax[row,1].errorbar(radii,resid,yerr=np.vstack([resid-resid_lo,resid_hi-resid]),fmt='-o',color=rank_cols[rank],lw=1.25,ms=3.8,capsize=2.8,elinewidth=1.15,label=f'Rank {rank}')
  rr=[r for r in real if r['model']==name]
  xs=np.arange(3)
  offsets=dict(zip(methods,[-.12,-.04,.04,.12]))
  for method in methods:
   sel=[next(r for r in rr if r['pair']==t and r['method']==method) for t in tasks]
   vals=np.array([float(r['loss_difference']) for r in sel]);lo=np.array([float(r['difference_ci_low']) for r in sel]);hi=np.array([float(r['difference_ci_high']) for r in sel])
   ax[row,2].errorbar(xs+offsets[method],vals,yerr=np.vstack([vals-lo,hi-vals]),fmt=mmark[method],color=mcols[method],ms=4,capsize=2.3,elinewidth=1.05,label=method)
  ax[row,2].axhline(0,color=BLUE,ls='--',lw=1,label='Samplewise')
  title='ResNet-18' if name=='resnet18' else 'ViT-B/16'
  ax[row,0].set_title(f'({chr(97+row*3)}) {title}: |signal error| (%)')
  ax[row,1].set_title(f'({chr(98+row*3)}) {title}: noise residual with 95% CI')
  ax[row,2].set_title(f'({chr(99+row*3)}) {title}: paired loss change')
  ax[row,0].set_xticks(range(3));ax[row,0].set_xticklabels([f'{d:g}' for d in radii]);ax[row,0].set_xlabel('Radius')
  ax[row,0].set_yticks(range(9),[fr'$r={rank},\ \epsilon={e:g}$' for rank in ranks for e in epsilons]);ax[row,0].tick_params(axis='y',labelsize=5.6)
  ax[row,1].axhline(0,color='.35',ls='--',lw=.9)
  ax[row,1].set_xticks(radii);ax[row,1].set_xticklabels([f'{d:g}' for d in radii]);ax[row,1].set_xlabel('Radius');ax[row,1].set_ylabel(r'$100(\mathrm{observed}/\mathrm{predicted}-1)$');ax[row,1].grid(alpha=.14)
  ax[row,2].set_xticks(xs);ax[row,2].set_xticklabels(task_labels,rotation=18,ha='right');ax[row,2].set_ylabel(r'$\Delta$ test log loss vs. samplewise');ax[row,2].grid(alpha=.14)
 cb=fig.colorbar(signal_images[0],ax=ax[:,0],fraction=.035,pad=.02);cb.ax.set_ylabel('Absolute relative error (%)',fontsize=6.5);cb.ax.tick_params(labelsize=5.7)
 rank_handles,rank_labels=ax[0,1].get_legend_handles_labels()
 method_handles,method_labels=ax[0,2].get_legend_handles_labels()
 fig.legend(rank_handles+method_handles,rank_labels+method_labels,
            loc='lower center',bbox_to_anchor=(.5,1.005),ncol=8,frameon=False,
            columnspacing=.65,handletextpad=.25,fontsize=5.7)
 save(fig,'deep_transfer_dense')

if __name__=='__main__':figure2();figure3()
