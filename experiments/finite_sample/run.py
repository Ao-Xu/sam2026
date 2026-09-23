import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
import sys,json,time
from pathlib import Path
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parent
import importlib.util
spec=importlib.util.spec_from_file_location('generalization_solver',str(ROOT.parent/'generalization/run.py'))
solver=importlib.util.module_from_spec(spec);spec.loader.exec_module(solver)
Model,rbf,quadrature,sample,write=[getattr(solver,k) for k in ['Model','rbf','quadrature','sample','write']]
for p in ['raw','results','figures']:(ROOT/p).mkdir(exist_ok=True)

def rbf_run():
    xt,wt=quadrature(32);probes=np.column_stack([np.ones(len(xt)),xt])
    with (ROOT/'raw/rbf.jsonl').open('w') as out:
        for n in [128,256,512,1024,2048]:
            for rep in range(100):
                rng=np.random.RandomState(4930000+n*1000+rep);x=sample(rng,n);y=np.where(rng.rand(n)<.5,1.,-1.)
                m=Model(rbf(x,x));project=(wt[:,None]*probes).T.dot(rbf(xt,x));B=project.dot(m.u)/np.sqrt(m.ev)
                z=m.phi.T.dot(y)/n
                for tau in [.02,.08]:
                    for delta in [0.,.5,1.]:
                        s=expit(delta);p=s*(1-s);h=tau+p*m.ev/n
                        linear=s*z/h;cond=(B*(s*s*m.ev/n/h**2)).dot(B.T)
                        w,info=m.fit(delta,tau,y=y)
                        row=dict(n=n,rep=rep,tau=tau,delta=delta,true=(np.sqrt(n)*B.dot(w)).tolist(),linear=(np.sqrt(n)*B.dot(linear)).tolist(),conditional=cond.tolist(),rkhs_remainder=float(np.linalg.norm(w-linear)))
                        row.update(info);write(out,row)
                if (rep+1)%10==0:print('RBF n={} rep={}'.format(n,rep+1),flush=True)

def real_run():
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from scipy.linalg import solve
    data=load_breast_cancer();itr,ite=train_test_split(np.arange(len(data.target)),test_size=.3,stratify=data.target,random_state=927)
    scaler=StandardScaler().fit(data.data[itr]);P=np.column_stack([np.ones(len(data.target)),scaler.transform(data.data)]);P/=np.linalg.norm(P,axis=1)[:,None]
    train,test=P[itr],P[ite];y0=2*data.target[itr]-1;yt=2*data.target[ite]-1;n=len(train)
    np.savez_compressed(str(ROOT/'raw/real_data_split.npz'),train_indices=itr,test_indices=ite,train_features=train,test_features=test,train_labels=y0,test_labels=yt,mean=scaler.mean_,scale=scaler.scale_)
    with (ROOT/'raw/real.jsonl').open('w') as out:
        for rep in range(100):
            rng=np.random.RandomState(882000+rep);ix=rng.randint(n,size=n);F=train[ix];T=F.T.dot(F)/n
            for mode,y in [('true',y0[ix]),('randomized',np.where(rng.rand(n)<.5,1.,-1.))]:
                # Model solver with the exact finite-feature factor, no Gram truncation.
                m=Model.__new__(Model);m.phi=F;m.weights=np.ones(n)/n;m.omitted=0.;m.discarded=0;m.mineig=0.
                for delta in [0.,.5,1.]:
                    tau=.02;w,info=m.fit(delta,tau,y=y);scores=test.dot(w);s=expit(delta);p=s*(1-s);H=tau*np.eye(F.shape[1])+p*T
                    linear=solve(H,s*F.T.dot(y)/n,assume_a='pos');G=solve(H,test.T,assume_a='pos').T
                    cond=np.sum((G.dot(T))*G,axis=1)*s*s
                    row=dict(rep=rep,mode=mode,n=n,delta=delta,tau=tau,scores=scores.tolist(),linear_scores=test.dot(linear).tolist(),conditional_diagonal=cond.tolist(),accuracy=float(np.mean((scores>=0)==(yt>0))),logloss=float(np.mean(np.logaddexp(0,-yt*scores))),brier=float(np.mean((expit(scores)-(yt>0))**2)))
                    row.update(info);write(out,row)
            if (rep+1)%20==0:print('REAL rep={}'.format(rep+1),flush=True)
    (ROOT/'results/real_source.txt').write_text(data.DESCR,encoding='utf8')

if __name__=='__main__':
    t=time.time();family=sys.argv[1] if len(sys.argv)>1 else 'rbf'
    (rbf_run if family=='rbf' else real_run)()
    (ROOT/'results'/('run_'+family+'.json')).write_text(json.dumps(dict(seconds=time.time()-t,python=sys.version,numpy=np.__version__,blas_threads=1),indent=2))
