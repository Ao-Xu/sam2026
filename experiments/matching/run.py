"""Execute protocol.md with exact full-dimensional convex objectives."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import sys,json,time
import numpy as np
import torch
from scipy.special import expit
from scipy.optimize import minimize_scalar
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R.parent))
from run_experiments import kernel,Model
torch.set_num_threads(4)
DEV='cuda' if torch.cuda.is_available() else 'cpu'
def ten(x):return torch.as_tensor(x,dtype=torch.float64,device=DEV)
def solve(x,m,tau,delta=0.,weights=None,tol=1e-11):
 n,d=x.shape;b=m.shape[1] if weights is None else weights.shape[1]
 w=torch.zeros((d,b),dtype=torch.float64,device=DEV);z=w.clone()
 bound=x.T@x/n if weights is None else x.T@(x*weights.max(1).values[:,None])/n
 L=tau+.25*float(torch.linalg.eigvalsh(bound)[-1]);beta=(L**.5-tau**.5)/(L**.5+tau**.5)
 def grad(v):
  f=x@v;g=-(1+m)*.5*torch.sigmoid(delta-f)+(1-m)*.5*torch.sigmoid(delta+f)
  if weights is not None:g=g*weights
  return x.T@g/n+tau*v
 for it in range(12000):
  wn=z-grad(z)/L;z=wn+beta*(wn-w);w=wn
  if it%20==0 and float(torch.linalg.vector_norm(grad(w),dim=0).max())<tol:break
 else:raise RuntimeError('convergence')
 return w,dict(iterations=it+1,residual=float(torch.linalg.vector_norm(grad(w),dim=0).max()),tau=tau,delta=delta)
def match(delta,tau=.02):
 s=expit(delta);p=s*(1-s);return tau/(4*p),s/(2*p)
def population():
 eps=np.array([.025,.05,.1,.2]);deltas=[0,.005,.02,.1,.25,.5,1];rows=[];checks=[]
 for n in [512,1024]:
  x=np.arange(n)*2*np.pi/n;gram=ten(kernel(x,x));ev,u=torch.linalg.eigh(gram);phi=u*ev.clamp_min(0).sqrt();m=ten(np.cos(x)[:,None]*eps)
  fits=[];controls=[]
  for delta in deltas:
   lam,scale=match(delta);w,rec=solve(phi,m,.02,delta,tol=1e-13);v,rc=solve(phi,m,lam,tol=1e-13)
   f=(phi@w).cpu().numpy();g=(phi@v).cpu().numpy()*scale;fits.append(f);controls.append(g)
   for j,e in enumerate(eps):
    diff=np.sqrt(np.mean((f[:,j]-g[:,j])**2));rows.append(dict(grid=n,epsilon=float(e),delta=delta,c1=float(2*np.mean(f[:,j]*np.cos(x))),c3=float(2*np.mean(f[:,j]*np.cos(3*x))),matched_c1=float(2*np.mean(g[:,j]*np.cos(x))),matched_c3=float(2*np.mean(g[:,j]*np.cos(3*x))),absolute=diff,relative=diff/np.sqrt(np.mean(f[:,j]**2)),residual=max(rec['residual'],rc['residual']),min_gram_eigenvalue=float(ev[0])))
   print('population',n,delta,flush=True)
  np.savez_compressed(R/'raw'/f'periodic_{n}.npz',x=x,epsilon=eps,radii=deltas,fit=np.stack(fits),matched=np.stack(controls))
 # Independent original Newton implementation, full closed-kernel Gram.
 x=np.arange(256)*2*np.pi/256;ev,u=torch.linalg.eigh(ten(kernel(x,x)));phi=u*ev.clamp_min(0).sqrt();m=ten(.1*np.cos(x)[:,None]);model=Model(x,.02)
 for delta in [.005,.25]:
  lam,scale=match(delta)
  for tau,radius,factor in [(.02,delta,1.),(lam,0.,scale)]:
   model.tau=tau;wn,info=model.fit(radius,p=(1+.1*np.cos(x))/2,tol=1e-13);wt,rec=solve(phi,m,tau,radius,tol=1e-13)
   err=float(np.max(np.abs(factor*(model.phi@wn-(phi@wt).cpu().numpy()[:,0]))))
   checks.append(dict(delta=delta,tau=tau,maximum_prediction_difference=err,newton_residual=info['residual'],gradient_residual=rec['residual']))
 (R/'raw'/'population.json').write_text(json.dumps(rows,indent=2));(R/'raw'/'checks.json').write_text(json.dumps(checks,indent=2))
def deep():
 old=R.parent/'deep_features'/'raw';ind=np.load(old/'data_indices.npz');ytr=ind['train_labels'];yte=ind['test_labels']
 for name in ['resnet18','vit_b_16']:
  data=np.load(old/f'{name}_features.npz');x=ten(data['train_raw']);xt=ten(data['test_raw']);x/=torch.linalg.vector_norm(x,dim=1,keepdim=True);xt/=torch.linalg.vector_norm(xt,dim=1,keepdim=True)
  sig=np.load(old/f'{name}_signal.npz');means=ten(np.concatenate([e*sig['q'] for e in sig['epsilon']],axis=1));controls=[];records=[]
  for delta in sig['radii']:
   lam,scale=match(delta);w,rec=solve(x,means,lam);controls.append((w*scale).cpu().numpy());records.append(rec)
  np.savez_compressed(R/'raw'/f'{name}_signal.npz',matched=np.stack(controls),original=sig['fit'],epsilon=sig['epsilon'],radii=sig['radii']);(R/'raw'/f'{name}_signal.json').write_text(json.dumps(records,indent=2))
  for pi,(a,b) in enumerate([(3,5),(1,9),(0,8)]):
   rng=np.random.default_rng(9143000+pi);tr=[];va=[]
   for c in [a,b]:
    ix=rng.permutation(np.flatnonzero(ytr==c));tr.extend(ix[:240]);va.extend(ix[240:])
   tr=np.array(tr);va=np.array(va);te=np.flatnonzero((yte==a)|(yte==b));xr=x[tr];xv=x[va];xe=xt[te];y=ten(np.where(ytr[tr]==b,1.,-1.))[:,None];yv=np.where(ytr[va]==b,1.,-1.);yt=np.where(yte[te]==b,1.,-1.)
   candidates=[]
   for tau in [.0002,.001,.005,.02,.08,.32]:
    w,rec=solve(xr,y,tau);f=(xv@w).cpu().numpy()[:,0]
    loss=lambda logscale:float(np.logaddexp(0,-yv*f*np.exp(logscale)).mean())
    opt=minimize_scalar(loss,bounds=(np.log(.05),np.log(20)),method='bounded',options={'xatol':1e-10})
    candidates.append(dict(tau=tau,loss=loss(0),scale=float(np.exp(opt.x)),calibrated_loss=float(opt.fun),**{k:v for k,v in rec.items() if k not in ['tau']}))
   best=min(candidates,key=lambda v:v['loss']);cal=min(candidates,key=lambda v:v['calibrated_loss']);lam,scale=match(1.)
   settings=[('ERM',.02,0.,1.),('Samplewise',.02,1.,1.),('Matched',lam,0.,scale),('Tuned',best['tau'],0.,1.),('Calibrated',cal['tau'],0.,cal['scale'])]
   rng=np.random.default_rng(9143100+pi);n=len(tr);counts=np.stack([np.bincount(rng.integers(n,size=n),minlength=n) for _ in range(30)],axis=1);scores=[];fits=[];records=[]
   for method,tau,delta,scale in settings:
    w,rec=solve(xr,y,tau,delta,ten(counts));scores.append((xe@w*scale).cpu().numpy());fits.append((w*scale).cpu().numpy());records.append(dict(method=method,scale=scale,**rec));print(name,a,b,method,rec,flush=True)
   np.savez_compressed(R/'raw'/f'{name}_real_{a}_{b}.npz',scores=np.stack(scores),fit=np.stack(fits),test_labels=yt,train_indices=tr,validation_indices=va,test_indices=te,counts=counts,methods=[s[0] for s in settings])
   (R/'raw'/f'{name}_real_{a}_{b}.json').write_text(json.dumps(dict(candidates=candidates,records=records),indent=2))
if __name__=='__main__':
 (R/'raw').mkdir(parents=True,exist_ok=True)
 if '--population' in sys.argv:population()
 else:deep()
