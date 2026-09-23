"""Independent changed-path checks: objective derivative, kernel prediction, sampling."""
import json
import numpy as np
from scipy.special import expit
from run import Model,rbf,quadrature,sample,ROOT
rng=np.random.RandomState(1843);checks=[]
for weighted in [False,True]:
    if weighted:x,weights=quadrature(8)
    else:x=sample(rng,64);weights=np.ones(64)/64
    K=rbf(x,x);m=Model(K,weights);tau=.02;delta=.5
    y=np.where(rng.rand(len(x))<.5,1.,-1.)
    p=(1+.15*(x[:,0]+.5*x[:,1])/1.5)/2
    w=rng.normal(size=m.phi.shape[1])*.02;v=rng.normal(size=len(w));v/=np.linalg.norm(v)
    def objective(w):
        f=m.phi.dot(w)
        losses=p*np.logaddexp(0,delta-f)+(1-p)*np.logaddexp(0,delta+f) if weighted else np.logaddexp(0,delta-y*f)
        return weights.dot(losses)+tau*w.dot(w)/2
    f=m.phi.dot(w)
    score=-p*expit(delta-f)+(1-p)*expit(delta+f) if weighted else -y*expit(delta-y*f)
    grad=m.phi.T.dot(weights*score)+tau*w
    h=1e-5;fd=(objective(w+h*v)-objective(w-h*v))/(2*h)
    # Independent alpha representation and direct Nystrom evaluation agree off-grid.
    xt=sample(rng,13);pred=rbf(xt,x).dot(m.alpha(w));feature=rbf(xt,x).dot(m.u)/np.sqrt(m.ev)
    fit,info=m.fit(delta,tau,p=p if weighted else None,y=None if weighted else y)
    ff=m.phi.dot(fit);sc=-p*expit(delta-ff)+(1-p)*expit(delta+ff) if weighted else -y*expit(delta-y*ff)
    # Stationarity via alpha-domain kernel representer, checked in H using K quadratic form.
    alpha=m.alpha(fit);r=weights*sc+tau*alpha;res=np.sqrt(max(r.dot(K.dot(r)),0.))
    checks.append(dict(weighted=weighted,finite_difference_absolute_error=float(abs(fd-grad.dot(v))),heldout_prediction_error=float(np.max(abs(pred-feature.dot(w)))),independent_rkhs_residual=float(res),stored_residual=info['residual'],omitted_gradient_bound=m.omitted))
    assert abs(fd-grad.dot(v))<1e-8
    assert np.max(abs(pred-feature.dot(w)))<1e-10
    assert res<1e-7
x=sample(rng,200000);means=x.mean(axis=0);expected=np.array([.6,.4])/3
se=np.sqrt((1/3-expected**2)/len(x));z=(means-expected)/se
assert max(abs(z))<5
raw=[json.loads(line) for name in ['e1','e2','weak'] for line in (ROOT/'raw'/(name+'.jsonl')).read_text().splitlines()]
out=dict(checks=checks,sampler_means=means.tolist(),expected_means=expected.tolist(),mean_zscores=z.tolist(),max_omitted_gradient_bound=max(v['full_residual_bound']-v['residual'] for v in raw),max_full_residual_bound=max(v['full_residual_bound'] for v in raw),max_retained_residual=max(v['residual'] for v in raw),max_iterations=max(v['iterations'] for v in raw),nonconverged=sum(v['residual']>1e-9 for v in raw))
(ROOT/'results/verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
