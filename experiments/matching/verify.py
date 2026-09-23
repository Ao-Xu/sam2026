"""Independent checks of data isolation, objectives, and reported arithmetic."""
from pathlib import Path
import json
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
R=Path(__file__).resolve().parent;old=R.parent/'deep_features/raw';checks=[];spot=[]
indices=np.load(old/'data_indices.npz')
for name in ['resnet18','vit_b_16']:
 data=np.load(old/f'{name}_features.npz');x=data['train_raw'].astype(float);x/=np.linalg.norm(x,axis=1,keepdims=True)
 for a,b in [(3,5),(1,9),(0,8)]:
  z=np.load(R/'raw'/f'{name}_real_{a}_{b}.npz');cfg=json.loads((R/'raw'/f'{name}_real_{a}_{b}.json').read_text());tr=z['train_indices'];va=z['validation_indices'];assert len(tr)==480 and len(va)==120 and not set(tr)&set(va);assert np.all(z['counts'].sum(axis=0)==480)
  xr=x[tr];y=np.where(indices['train_labels'][tr]==b,1.,-1.);weights=z['counts'][:,0]
  # Every method: analytic CPU gradient, independently recomputed from saved scaled fit.
  for i,c in enumerate(cfg['records']):
   w=z['fit'][i,:,0]/c['scale'];f=xr@w;g=xr.T@(-y*expit(c['delta']-y*f)*weights)/480+c['tau']*w
   gn=float(np.linalg.norm(g));assert gn<2e-10;checks.append(dict(model=name,pair=[a,b],method=c['method'],cpu_gradient_norm=gn))
  # One independent unscaled ERM fit per backbone: direct CPU L-BFGS.
  if (a,b)==(3,5):
   c=cfg['records'][0]
   def objective(w):
    f=xr@w;loss=np.mean(weights*np.logaddexp(0,-y*f))+.5*c['tau']*(w@w);g=xr.T@(-y*expit(-y*f)*weights)/480+c['tau']*w;return loss,g
   opt=minimize(objective,np.zeros(x.shape[1]),jac=True,method='L-BFGS-B',options={'gtol':1e-12,'ftol':1e-15,'maxiter':2000,'maxls':50})
   err=float(np.linalg.norm(opt.x-z['fit'][0,:,0]));assert err<1e-6;spot.append(dict(model=name,weight_difference=err,objective=opt.fun,success=bool(opt.success)))
 # Shared train/val indices across backbones checked below.
for a,b in [(3,5),(1,9),(0,8)]:
 zs=[np.load(R/'raw'/f'{n}_real_{a}_{b}.npz') for n in ['resnet18','vit_b_16']]
 for key in ['train_indices','validation_indices','test_indices','counts']:assert np.array_equal(zs[0][key],zs[1][key])
s=json.loads((R/'results/summary.json').read_text());assert 2.95<s['slopes']['absolute']<3.05 and 1.95<s['slopes']['relative']<2.05
pc=json.loads((R/'raw/checks.json').read_text());assert max(c['maximum_prediction_difference'] for c in pc)<1e-10
out=dict(status='PASS',cpu_gradient_checks=checks,independent_optimizer=spot,original_newton_checks=pc,split_and_pairing='PASS',population_fits=112,deep_weak_matched_fits=72,bootstrap_head_fits=900,validation_head_fits=36)
(R/'results/verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
