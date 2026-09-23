import run as base
import numpy as np
from scipy.linalg import eigh
from scipy.special import expit
def kernel(x,z,b):
    d=np.maximum(np.sum(x*x,axis=1)[:,None]+np.sum(z*z,axis=1)[None,:]-2*x.dot(z.T),0.)
    return np.exp(-d/(2*b*b))
with (base.ROOT/'raw/bandwidth_reference.jsonl').open('w') as out:
    for b in [.4,1.2]:
        for order in [24,32]:
            x,wt=base.quadrature(order);sw=np.sqrt(wt);K=kernel(x,x,b);ev,U=eigh(sw[:,None]*K*sw[None,:]);keep=ev>1e-13;ev=ev[keep];U=U[:,keep]
            probes=np.column_stack([np.ones(len(x)),x]);hc=U.T.dot(sw[:,None]*probes);s=expit(1.);p=s*(1-s);gain=s*ev/(.02+p*ev)
            base.write(out,dict(bandwidth=b,order=order,covariance=((hc.T*gain**2).dot(hc)).tolist()))
with (base.ROOT/'raw/bandwidth.jsonl').open('w') as out:
    xt,wt=base.quadrature(32);probes=np.column_stack([np.ones(len(xt)),xt])
    for b in [.4,1.2]:
        for n in [256,1024]:
            for rep in range(100):
                rng=np.random.RandomState(4930000+n*1000+rep);x=base.sample(rng,n);y=np.where(rng.rand(n)<.5,1.,-1.)
                m=base.Model(kernel(x,x,b));B=((wt[:,None]*probes).T.dot(kernel(xt,x,b))).dot(m.u)/np.sqrt(m.ev)
                s=expit(1.);p=s*(1-s);h=.02+p*m.ev/n;linear=s*m.phi.T.dot(y)/n/h;cond=(B*(s*s*m.ev/n/h**2)).dot(B.T)
                w,info=m.fit(1.,.02,y=y);row=dict(bandwidth=b,n=n,rep=rep,tau=.02,delta=1.,true=(np.sqrt(n)*B.dot(w)).tolist(),linear=(np.sqrt(n)*B.dot(linear)).tolist(),conditional=cond.tolist(),rkhs_remainder=float(np.linalg.norm(w-linear)));row.update(info);base.write(out,row)
            print('bandwidth={} n={} complete'.format(b,n),flush=True)
