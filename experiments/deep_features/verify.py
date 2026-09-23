"""Independent CPU objective/optimizer check of GPU fits and exact feature identities."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.special import expit
from scipy.optimize import minimize,check_grad
R=Path(__file__).resolve().parent;TAU=.02
def main():
 checks=[];rng=np.random.default_rng(9142500)
 for name in ['resnet18','vit_b_16']:
  f=np.load(R/'raw'/f'{name}_features.npz');x=f['train_raw'].astype(float);x/=np.linalg.norm(x,axis=1,keepdims=True)
  sig=np.load(R/'raw'/f'{name}_signal.npz');noise=np.load(R/'raw'/f'{name}_noise.npz');real=np.load(R/'raw'/f'{name}_real_3_5.npz')
  tasks=[('signal',x,.2*sig['q'][:,1],np.ones(len(x)),sig['fit'][3,:,7]),('noise',x[noise['subset']],noise['labels'][:,0],np.ones(len(noise['subset'])),noise['fit'][3,:,0]),('real',x[real['train_indices']],real['train_labels'][:,0],real['counts'][:,0],real['fit'][3,:,0])]
  for kind,z,m,a,reported in tasks:
   n=len(z)
   def fun(w):
    pred=z@w
    value=np.mean(a*((1+m)/2*np.logaddexp(0,1-pred)+(1-m)/2*np.logaddexp(0,1+pred)))+TAU/2*(w@w)
    grad=z.T@(a*(-(1+m)/2*expit(1-pred)+(1-m)/2*expit(1+pred)))/n+TAU*w
    return value,grad
   fit=minimize(fun,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',options={'gtol':1e-12,'ftol':1e-15,'maxiter':2000,'maxls':50})
   error=np.linalg.norm(fit.x-reported);residual=np.linalg.norm(fun(reported)[1]);assert residual<1e-8 and error<2e-5,(kind,error,residual)
   direction=rng.normal(size=z.shape[1]);direction/=np.linalg.norm(direction);w=rng.normal(scale=.01,size=z.shape[1]);h=1e-5
   fd=(fun(w+h*direction)[0]-fun(w-h*direction)[0])/(2*h);gd=fun(w)[1]@direction
   assert abs(fd-gd)<1e-8
   checks.append({'model':name,'kind':kind,'coefficient_difference':float(error),'reported_gradient_norm':float(residual),'directional_gradient_error':float(abs(fd-gd)),'independent_optimizer_success':bool(fit.success)})
  # Analytic covariance agrees with direct matrix solve, independent of stored eigendecomposition.
  z=x[noise['subset']];t=z.T@z/len(z);s=expit(1);p=s*(1-s);h=TAU*np.eye(z.shape[1])+p*t
  v=np.linalg.solve(h,noise['probes']);diag=s*s*np.sum(v*(t@v),axis=0)
  assert np.allclose(diag,noise['conditional'][3],rtol=1e-9,atol=1e-10)
  assert np.allclose(np.linalg.norm(x,axis=1),1,atol=1e-14)
 out={'status':'PASS','independent_checks':checks,'feature_and_conditional_covariance_checks':'PASS'}
 (R/'results/verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
