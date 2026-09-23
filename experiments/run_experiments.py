"""Independent exact empirical optimization for the revised paper.

Run with OMP_NUM_THREADS=MKL_NUM_THREADS=1, Python with NumPy/SciPy.
No robust estimator is synthesized from an asymptotic formula.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh, solve
from scipy.special import expit
from scipy.optimize import minimize_scalar

ROOT=Path(__file__).resolve().parent
DEN=1+np.pi**4/45

def kernel(x,z):
    t=np.mod(np.asarray(x)[:,None]-np.asarray(z)[None,:],2*np.pi)
    return (1+2*(np.pi**4/90-np.pi**2*t*t/12+np.pi*t**3/12-t**4/48))/DEN

def mu(j):return 1/DEN if j==0 else 1/(DEN*j**4)

class Model:
    def __init__(self,x,tau=.02):
        self.x=np.asarray(x);self.tau=tau
        ev,u=eigh(kernel(x,x),check_finite=False)
        self.min_eigenvalue=float(ev[0])
        keep=ev>max(ev[-1]*1e-13,1e-13)
        self.ev=ev[keep];self.u=u[:,keep]
        self.phi=self.u*np.sqrt(self.ev)
        self.discarded=int((~keep).sum())
        self.omitted_eigenvalue_bound=max(float(ev[-1])*1e-13,1e-13) if self.discarded else 0.

    def fit(self,delta,y=None,p=None,w0=None,tol=1e-11):
        phi=self.phi;n=len(self.x);tau=self.tau
        w=np.zeros(phi.shape[1]) if w0 is None else w0.copy()
        def evaluate(w,hessian=False):
            f=phi.dot(w)
            if p is None:
                a=delta-y*f;s=expit(a)
                val=np.logaddexp(0,a).mean()+tau*w.dot(w)/2
                score=-y*s;curv=s*(1-s)
            else:
                sp=expit(delta-f);sm=expit(delta+f)
                val=np.mean(p*np.logaddexp(0,delta-f)+(1-p)*np.logaddexp(0,delta+f))+tau*w.dot(w)/2
                score=-p*sp+(1-p)*sm
                curv=p*sp*(1-sp)+(1-p)*sm*(1-sm)
            g=phi.T.dot(score)/n+tau*w
            h=(phi.T*curv).dot(phi)/n+tau*np.eye(len(w)) if hessian else None
            return val,g,h
        for it in range(80):
            val,g,h=evaluate(w,True)
            gn=np.linalg.norm(g)
            if gn<=tol:break
            step=solve(h,g,assume_a='pos',check_finite=False)
            rate=1.
            while rate>2**-30:
                candidate=w-rate*step
                nv,_,_=evaluate(candidate)
                if nv<=val-1e-4*rate*g.dot(step)+1e-15:break
                rate*=.5
            w=candidate
        val,g,h=evaluate(w,True)
        # |score_i|<=1 gives omitted gradient norm <=sqrt(cutoff/n).
        # This conservative term distinguishes numerical rank filtering from exact arithmetic.
        residual_bound=float(np.linalg.norm(g))+np.sqrt(self.omitted_eigenvalue_bound/len(self.x))
        return w,{'residual':float(np.linalg.norm(g)), 'full_residual_bound':residual_bound,
            'distance_bound':residual_bound/tau,'iterations':it+1,'objective':float(val),
            'discarded':self.discarded,'min_gram_eigenvalue':self.min_eigenvalue}

    def drift(self,w,y=None,p=None,delta=0.):
        f=self.phi.dot(w)
        if p is None:
            s=expit(delta-y*f);curv=s*(1-s);r=-y*curv
        else:
            sp=expit(delta-f);sm=expit(delta+f)
            curv=p*sp*(1-sp)+(1-p)*sm*(1-sm)
            r=-p*sp*(1-sp)+(1-p)*sm*(1-sm)
        h=(self.phi.T*curv).dot(self.phi)/len(f)+self.tau*np.eye(len(w))
        return solve(h,self.phi.T.dot(r)/len(f),assume_a='pos',check_finite=False)

    def coefficient(self,w,k):
        # Exact RKHS eigen-coordinate: <f,e_k>_H = sqrt(2 mu_k) sum alpha_i cos(k x_i).
        alpha=self.u.dot(w/np.sqrt(self.ev))
        if k==0:return np.sqrt(mu(0))*alpha.sum()
        return np.sqrt(2*mu(k))*alpha.dot(np.cos(k*self.x))

def append(handle,row):
    handle.write(json.dumps(row,sort_keys=True)+'\n');handle.flush()

def asymptotic(reps,sizes):
    gammas=[.25,.5,.75];etas=[.3,1.]
    with (ROOT/'raw'/'asymptotic.jsonl').open('w') as out:
        for n in sizes:
            for rep in range(reps):
                rng=np.random.RandomState(410000+n*1000+rep)
                x=rng.uniform(0,2*np.pi,n);p=(1+.5*np.cos(x))/2
                y=np.where(rng.rand(n)<p,1.,-1.)
                m=Model(x);w0,info0=m.fit(0,y=y);d=m.drift(w0,y=y)
                for eta in etas:
                    for gamma in gammas:
                        delta=eta*n**(-gamma)
                        w,info=m.fit(delta,y=y,w0=w0)
                        diff=w-w0;norm=np.linalg.norm(diff)
                        append(out,dict(n=n,rep=rep,gamma=gamma,eta=eta,delta=delta,
                            distance=float(norm),scaled_distance=float(np.sqrt(n)*norm),
                            normalized_distance=float(norm/delta),
                            prediction_relative_error=float(np.linalg.norm(diff+delta*d)/max(norm,1e-16)),
                            prediction_cosine=float(-diff.dot(d)/max(norm*np.linalg.norm(d),1e-16)),
                            residual=max(info['residual'],info0['residual']),discarded=m.discarded))
            print('asymptotic n={} complete'.format(n),flush=True)

def balanced(reps,sizes):
    radii=[0.,.25,.5,1.]
    with (ROOT/'raw'/'balanced.jsonl').open('w') as out:
        for n in sizes:
            for rep in range(reps):
                rng=np.random.RandomState(870000+n*1000+rep)
                x=rng.uniform(0,2*np.pi,n);y=np.where(rng.rand(n)<.5,1.,-1.)
                m=Model(x);w0,info0=m.fit(0,y=y)
                for delta in radii+[n**(-.25)]:
                    w,info=m.fit(delta,y=y,w0=w0)
                    append(out,dict(n=n,rep=rep,delta=delta,shrinking=delta not in radii,
                         rkhs_norm=float(np.linalg.norm(w)),scaled_norm=float(np.sqrt(n)*np.linalg.norm(w)),
                         paired_scaled=float(np.sqrt(n)*np.linalg.norm(w-w0)),
                         coords=[float(np.sqrt(n)*m.coefficient(w,k)) for k in [0,1,2]],
                         residual=max(info['residual'],info0['residual'])))
            print('balanced n={} complete'.format(n),flush=True)

def spectral():
    with (ROOT/'raw'/'spectral.jsonl').open('w') as out:
        for grid in [256,512,1024]:
            x=np.arange(grid)*2*np.pi/grid
            for tau in [.005,.02,.08]:
                m=Model(x,tau)
                for k in [1,3]:
                    for eps in [.025,.05,.1,.2,.4]:
                        p=(1+eps*np.cos(k*x))/2
                        for delta in [0.,.005,.02,.1,.25,.5,1.]:
                            w,info=m.fit(delta,p=p)
                            f=m.phi.dot(w)
                            s=expit(delta);sp=s*(1-s);spp=sp*(1-2*s);sppp=sp*(1-6*s+6*s*s)
                            a=mu(k)*s/(tau+mu(k)*sp)
                            b3=mu(3*k)/(tau+mu(3*k)*sp)*(spp*a*a/8-sppp*a**3/24)
                            b1=3*mu(k)/(tau+mu(k)*sp)*(spp*a*a/8-sppp*a**3/24)
                            lead=eps*a*np.cos(k*x)
                            cubic=lead+eps**3*(b1*np.cos(k*x)+b3*np.cos(3*k*x))
                            # quadrature coefficients match the population Fourier convention
                            append(out,dict(grid=grid,tau=tau,k=k,epsilon=eps,delta=delta,
                                fundamental=float(2*np.mean(f*np.cos(k*x))),third=float(2*np.mean(f*np.cos(3*k*x))),
                                predicted_fundamental=float(eps*a+eps**3*b1),predicted_third=float(eps**3*b3),
                                leading_l2_error=float(np.sqrt(np.mean((f-lead)**2))),
                                cubic_l2_error=float(np.sqrt(np.mean((f-cubic)**2))),
                                rkhs_norm=float(np.linalg.norm(w)),residual=info['residual'],discarded=m.discarded))
            print('spectral grid={} complete'.format(grid),flush=True)

def scalar():
    rows=[]
    for p in [.5,.65,.8]:
        for delta in [0.,.1,.25,.5,1.]:
            tau=.02
            risk=lambda t:p*np.logaddexp(0,-t)+(1-p)*np.logaddexp(0,t)
            objs={'ordinary':lambda t:risk(t)+tau*t*t/2,
                  'samplewise':lambda t:p*np.logaddexp(0,delta-t)+(1-p)*np.logaddexp(0,delta+t)+tau*t*t/2,
                  'shared':lambda t:max(risk(t-delta),risk(t+delta))+tau*t*t/2}
            for name,obj in objs.items():
                res=minimize_scalar(obj,bounds=(-10,10),method='bounded',options={'xatol':1e-12})
                rows.append(dict(p=p,delta=delta,tau=tau,method=name,optimum=float(res.x),objective=float(res.fun)))
    (ROOT/'raw'/'scalar.json').write_text(json.dumps(rows,indent=2))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--family',choices=['all','asymptotic','balanced','spectral','scalar'],default='all')
    parser.add_argument('--replicates',type=int,default=100);parser.add_argument('--sizes',default='64,128,256,512,1024')
    args=parser.parse_args();(ROOT/'raw').mkdir(exist_ok=True)
    config=dict(vars(args));config.update(kernel='periodic_sobolev_order2_closed_form',tau=.02,gradient_tolerance=1e-11,
        numpy=np.__version__,seed_scheme='family offset + n*1000 + replicate',kernel_diagonal=1.)
    (ROOT/'raw'/('config_'+args.family+'.json')).write_text(json.dumps(config,indent=2))
    started=time.time();sizes=[int(v) for v in args.sizes.split(',')]
    if args.family in ['all','asymptotic']:asymptotic(args.replicates,sizes)
    if args.family in ['all','balanced']:balanced(args.replicates,sizes)
    if args.family in ['all','spectral']:spectral()
    if args.family in ['all','scalar']:scalar()
    print('Completed {} in {:.1f}s'.format(args.family,time.time()-started),flush=True)

if __name__=='__main__':main()
