"""Full prespecified exact-head fits, float64 and no representation truncation."""
from pathlib import Path
import time,json,hashlib
import numpy as np
import torch
R=Path(__file__).resolve().parent
RADII=[0.,.25,.5,1.];TAU=.02;EPS=[.05,.1,.2];RANKS=[0,7,31]
torch.set_num_threads(4)
DT=torch.float64;DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
def tensor(x):return torch.as_tensor(x,dtype=DT,device=DEVICE)

def solve(x,m,delta,weights=None):
 """Columns are independently optimized tasks; m in [-1,1] is label mean."""
 n,d=x.shape;b=m.shape[1] if weights is None else weights.shape[1];w=torch.zeros((d,b),dtype=DT,device=DEVICE);z=w.clone()
 # For bootstrap tasks, max per-task row weight gives a valid shared Hessian bound.
 if weights is None:
  lip=TAU+.25*float(torch.linalg.eigvalsh(x.T@x/n)[-1])
 else:
  lip=TAU+.25*float(torch.linalg.eigvalsh(x.T@(x*weights.max(dim=1).values[:,None])/n)[-1])
 beta=(lip**.5-TAU**.5)/(lip**.5+TAU**.5)
 def grad(v):
  f=x@v
  g=-(1+m)*.5*torch.sigmoid(delta-f)+(1-m)*.5*torch.sigmoid(delta+f)
  if weights is not None:g=g*weights
  return x.T@g/n+TAU*v
 for it in range(2000):
  new=z-grad(z)/lip;z=new+beta*(new-w);w=new
  if it%10==0:
   norms=torch.linalg.vector_norm(grad(w),dim=0)
   if float(norms.max())<1e-10:break
 else:raise RuntimeError('Convergence failure '+str(float(norms.max())))
 g=grad(w)
 return w.detach(),{'iterations':it+1,'gradient_max':float(torch.linalg.vector_norm(g,dim=0).max()),'lipschitz':lip}

def main():
 (R/'raw').mkdir(exist_ok=True)
 indices=np.load(R/'raw/data_indices.npz');train_y=indices['train_labels'];test_y=indices['test_labels']
 for name in ['resnet18','vit_b_16']:
  data=np.load(R/'raw'/f'{name}_features.npz');x=tensor(data['train_raw']);xt=tensor(data['test_raw']);x=x/torch.linalg.vector_norm(x,dim=1,keepdim=True);xt=xt/torch.linalg.vector_norm(xt,dim=1,keepdim=True);n,d=x.shape
  ev,u=torch.linalg.eigh(x.T@x/n);ev=ev.flip(0);u=u.flip(1);probes=u[:,RANKS];q=x@probes;q=q/q.abs().max(dim=0).values
  means=torch.cat([e*q for e in EPS],dim=1)
  path=R/'raw'/f'{name}_signal.npz'
  if not path.exists():
   records=[];fits=[];refs=[]
   for delta in RADII:
    t=time.perf_counter();w,rec=solve(x,means,delta)
    s=1/(1+np.exp(-delta));p=s*(1-s)
    ref=u@((u.T@(x.T@means/n))*s/(TAU+p*ev)[:,None])
    fits.append(w.cpu().numpy());refs.append(ref.cpu().numpy());records.append(rec)
    print(name,'signal',delta,rec,'seconds',time.perf_counter()-t,flush=True)
   np.savez_compressed(path,fit=np.stack(fits),reference=np.stack(refs),probes=probes.cpu().numpy(),q=q.cpu().numpy(),eigenvalues=ev.cpu().numpy(),radii=RADII,epsilon=EPS)
   path.with_suffix('.json').write_text(json.dumps(records,indent=2))
  subset=np.random.default_rng(9142027).choice(n,2048,replace=False);xb=x[subset];nb=len(subset)
  evb,ub=torch.linalg.eigh(xb.T@xb/nb);evb=evb.flip(0);ub=ub.flip(1);pb=ub[:,RANKS]
  labels=np.random.default_rng(9142100).choice([-1.,1.],size=(nb,100));y=tensor(labels)
  path=R/'raw'/f'{name}_noise.npz'
  if not path.exists():
   fits=[];refs=[];cond=[];records=[]
   for delta in RADII:
    t=time.perf_counter();w,rec=solve(xb,y,delta);s=1/(1+np.exp(-delta));p=s*(1-s)
    ref=ub@((ub.T@(xb.T@y/nb))*s/(TAU+p*evb)[:,None])
    fits.append(w.cpu().numpy());refs.append(ref.cpu().numpy());cond.append((s*s*evb[RANKS]/(TAU+p*evb[RANKS])**2).cpu().numpy());records.append(rec)
    print(name,'noise',delta,rec,'seconds',time.perf_counter()-t,flush=True)
   np.savez_compressed(path,fit=np.stack(fits),reference=np.stack(refs),conditional=np.stack(cond),probes=pb.cpu().numpy(),eigenvalues=evb.cpu().numpy(),labels=labels,subset=subset,radii=RADII)
   path.with_suffix('.json').write_text(json.dumps(records,indent=2))
  for pi,(a,b) in enumerate([(3,5),(1,9),(0,8)]):
   path=R/'raw'/f'{name}_real_{a}_{b}.npz'
   if path.exists():continue
   ti=np.flatnonzero((train_y==a)|(train_y==b));vi=np.flatnonzero((test_y==a)|(test_y==b));xr=x[ti];xv=xt[vi];nr=len(ti)
   yreal=tensor(np.where(train_y[ti]==b,1.,-1.))[:,None];yt=np.where(test_y[vi]==b,1.,-1.)
   rng=np.random.default_rng(9142300+pi);counts=np.stack([np.bincount(rng.integers(nr,size=nr),minlength=nr) for _ in range(30)],axis=1);weights=tensor(counts)
   scores=[];fits=[];records=[]
   for delta in RADII:
    t=time.perf_counter();w,rec=solve(xr,yreal,delta,weights);scores.append((xv@w).cpu().numpy());fits.append(w.cpu().numpy());records.append(rec)
    print(name,'real',a,b,delta,rec,'seconds',time.perf_counter()-t,flush=True)
   np.savez_compressed(path,fit=np.stack(fits),scores=np.stack(scores),test_labels=yt,train_labels=yreal.cpu().numpy(),counts=counts,train_indices=ti,test_indices=vi,radii=RADII)
   path.with_suffix('.json').write_text(json.dumps(records,indent=2))
 print('ALL FITS COMPLETE',flush=True)
if __name__=='__main__':main()
