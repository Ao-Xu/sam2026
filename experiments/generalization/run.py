import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name]='1'
import sys,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh,solve
from scipy.special import expit
from numpy.polynomial.legendre import leggauss
ROOT=Path(__file__).resolve().parent
for name in ['raw','results','figures']:(ROOT/name).mkdir(exist_ok=True)
DEN=1+np.pi**4/45
def periodic(x,z):
    t=np.mod(x[:,None]-z[None,:],2*np.pi)
    return (1+2*(np.pi**4/90-np.pi**2*t*t/12+np.pi*t**3/12-t**4/48))/DEN
def rbf(x,z):
    d=np.maximum(np.sum(x*x,axis=1)[:,None]+np.sum(z*z,axis=1)[None,:]-2*x.dot(z.T),0.)
    return np.exp(-d/(2*.7**2))
class Model:
    def __init__(self,K,weights=None):
        n=len(K);self.weights=np.ones(n)/n if weights is None else weights
        ev,u=eigh(K,check_finite=False);cut=max(ev[-1]*1e-13,1e-13);keep=ev>cut
        self.ev=ev[keep];self.u=u[:,keep];self.phi=self.u*np.sqrt(self.ev)
        self.omitted=float(np.sqrt(cut)*np.linalg.norm(self.weights)) if (~keep).any() else 0.
        self.discarded=int((~keep).sum());self.mineig=float(ev[0])
    def fit(self,delta,tau,y=None,p=None,w0=None):
        P=self.phi;wt=self.weights;w=np.zeros(P.shape[1]) if w0 is None else w0.copy()
        def evaluate(w,hess=False):
            f=P.dot(w)
            if p is None:
                a=delta-y*f;s=expit(a);v=wt.dot(np.logaddexp(0,a));g=-y*s;c=s*(1-s)
            else:
                sp=expit(delta-f);sm=expit(delta+f)
                v=wt.dot(p*np.logaddexp(0,delta-f)+(1-p)*np.logaddexp(0,delta+f))
                g=-p*sp+(1-p)*sm;c=p*sp*(1-sp)+(1-p)*sm*(1-sm)
            return v+tau*w.dot(w)/2,P.T.dot(wt*g)+tau*w,((P.T*(wt*c)).dot(P)+tau*np.eye(len(w))) if hess else None
        for it in range(80):
            v,g,H=evaluate(w,True)
            if np.linalg.norm(g)<1e-10:break
            d=solve(H,g,assume_a='pos',check_finite=False);rate=1.
            while rate>2**-30:
                wn=w-rate*d;vn,_,_=evaluate(wn)
                if vn<=v-1e-4*rate*g.dot(d)+1e-15:break
                rate*=.5
            w=wn
        v,g,_=evaluate(w)
        return w,dict(residual=float(np.linalg.norm(g)),full_residual_bound=float(np.linalg.norm(g)+self.omitted),distance_bound=float((np.linalg.norm(g)+self.omitted)/tau),iterations=it+1,discarded=self.discarded,min_eigenvalue=self.mineig,objective=float(v))
    def alpha(self,w):return self.u.dot(w/np.sqrt(self.ev))
    def linear(self,delta,tau,q):
        s=expit(delta);p=s*(1-s);P=self.phi
        return solve(tau*np.eye(P.shape[1])+p*(P.T*self.weights).dot(P),s*P.T.dot(self.weights*q),assume_a='pos',check_finite=False)
def write(f,row):f.write(json.dumps(row,sort_keys=True)+'\n');f.flush()
def quadrature(order):
    v,w=leggauss(order);xx,yy=np.meshgrid(v,v,indexing='ij');wx,wy=np.meshgrid(w,w,indexing='ij')
    x=np.column_stack([xx.ravel(),yy.ravel()]);wt=(wx*wy).ravel()*(1+.6*x[:,0])*(1+.4*x[:,1])/4
    return x,wt
def e1():
    with (ROOT/'raw/e1.jsonl').open('w') as out:
        for grid in [256,512]:
            x=np.arange(grid)*2*np.pi/grid;K=periodic(x,x)
            for a in [0.,.3,.6]:
                m=Model(K,(1+a*np.cos(x))/grid)
                for b in [0.,.25,.5]:
                    q=np.cos(x)+b*np.cos(2*x);p=(1+.15*q)/2
                    for tau in [.02,.08]:
                        for delta in [0.,.005,.02,.1,.25,.5,1.]:
                            w,info=m.fit(delta,tau,p=p);f=m.phi.dot(w);lin=.15*m.phi.dot(m.linear(delta,tau,q))
                            coeff=[float(np.mean(f))]+[float(2*np.mean(f*np.cos(j*x))) for j in range(1,7)]
                            lc=[float(np.mean(lin))]+[float(2*np.mean(lin*np.cos(j*x))) for j in range(1,7)]
                            row=dict(grid=grid,a=a,b=b,epsilon=.15,tau=tau,delta=delta,coefficients=coeff,linear_coefficients=lc,rkhs_norm=float(np.linalg.norm(w)),linear_l2_error=float(np.sqrt(m.weights.dot((f-lin)**2))))
                            row.update(info);write(out,row)
                print('E1 grid={} a={} complete'.format(grid,a),flush=True)
def reference():
    with (ROOT/'raw/reference.jsonl').open('w') as out,(ROOT/'raw/weak.jsonl').open('w') as weak:
        for order in [24,32]:
            x,wt=quadrature(order);K=rbf(x,x);sw=np.sqrt(wt);B=(sw[:,None]*K)*sw[None,:]
            ev,U=eigh(B,check_finite=False);keep=ev>1e-13;ev=ev[keep];U=U[:,keep]
            probes=np.column_stack([np.ones(len(x)),x]);hc=U.T.dot(sw[:,None]*probes)
            m=Model(K,wt);q=(x[:,0]+.5*x[:,1])/1.5
            for tau in [.02,.08]:
                for delta in [0.,.5,1.]:
                    s=expit(delta);p=s*(1-s);gain=s*ev/(tau+p*ev)
                    cov=(hc.T*gain**2).dot(hc)
                    write(out,dict(order=order,tau=tau,delta=delta,covariance=cov.tolist(),trace_prediction=float(np.sum(gain**2)),quadrature_weight_sum=float(wt.sum())))
                    wl=m.linear(delta,tau,q);fl=m.phi.dot(wl)
                    for eps in [.05,.1,.2]:
                        w,info=m.fit(delta,tau,p=(1+eps*q)/2);f=m.phi.dot(w)
                        row=dict(order=order,tau=tau,delta=delta,epsilon=eps,l2_error=float(np.sqrt(wt.dot((f-eps*fl)**2))),relative_l2_error=float(np.sqrt(wt.dot((f-eps*fl)**2)/wt.dot(f*f))),projections=(probes.T.dot(wt*f)).tolist(),linear_projections=(probes.T.dot(wt*eps*fl)).tolist())
                        row.update(info);write(weak,row)
            print('E2 reference order={} complete'.format(order),flush=True)
def sample(rng,n):
    u=rng.rand(n,2);a=np.array([.6,.4])
    return (-1+np.sqrt((1-a)**2+4*a*u))/a
def e2():
    xt,wt=quadrature(32);probes=np.column_stack([np.ones(len(xt)),xt])
    with (ROOT/'raw/e2.jsonl').open('w') as out:
        for n in [128,256,512]:
            for rep in range(100):
                rng=np.random.RandomState(1930000+n*1000+rep);x=sample(rng,n);y=np.where(rng.rand(n)<.5,1.,-1.)
                m=Model(rbf(x,x));cross=rbf(xt,x);proj=(wt[:,None]*probes).T.dot(cross)
                for tau in [.02,.08]:
                    for delta in [0.,.5,1.]:
                        w,info=m.fit(delta,tau,y=y);alpha=m.alpha(w);ft=cross.dot(alpha)
                        row=dict(n=n,rep=rep,tau=tau,delta=delta,scaled_projections=(np.sqrt(n)*proj.dot(alpha)).tolist(),test_logloss=float(wt.dot((np.logaddexp(0,ft)+np.logaddexp(0,-ft))/2)),rkhs_norm=float(np.linalg.norm(w)))
                        row.update(info);write(out,row)
                if (rep+1)%25==0:print('E2 n={} rep={} complete'.format(n,rep+1),flush=True)
def main():
    started=time.time();family=sys.argv[1] if len(sys.argv)>1 else 'all'
    for name,fun in [('e1',e1),('reference',reference),('e2',e2)]:
        if family in ['all',name]:fun()
    (ROOT/'results'/('run_'+family+'.json')).write_text(json.dumps(dict(seconds=time.time()-started,numpy=np.__version__,python=sys.version,blas_threads=1),indent=2))
if __name__=='__main__':main()
