import os
for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.special import expit
from scipy.linalg import solve
from scipy.optimize import minimize
from numpy.polynomial.legendre import leggauss
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parent
read=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
data=load_breast_cancer();sp=np.load(str(ROOT/'raw/real_data_split.npz'));itr,ite=train_test_split(np.arange(569),test_size=.3,stratify=data.target,random_state=927)
assert set(itr).isdisjoint(ite) and len(itr)==398 and len(ite)==171
np.testing.assert_array_equal(itr,sp['train_indices']);np.testing.assert_array_equal(ite,sp['test_indices'])
scaler=StandardScaler().fit(data.data[itr]);np.testing.assert_allclose(scaler.mean_,sp['mean']);np.testing.assert_allclose(scaler.scale_,sp['scale'])
P=np.column_stack([np.ones(569),scaler.transform(data.data)]);P/=np.linalg.norm(P,axis=1)[:,None]
np.testing.assert_allclose(P[itr],sp['train_features']);np.testing.assert_allclose(P[ite],sp['test_features'])
rr=read(ROOT/'raw/real.jsonl');r=next(r for r in rr if r['rep']==0 and r['mode']=='true' and r['delta']==1.)
rng=np.random.RandomState(882000);ix=rng.randint(398,size=398);F=P[itr][ix];y=(2*data.target[itr]-1)[ix]
def fun(w):
    a=1-y*F.dot(w)
    return np.mean(np.logaddexp(0,a))+.01*w.dot(w),-F.T.dot(y*expit(a))/398+.02*w
sol=minimize(fun,np.zeros(31),jac=True,method='L-BFGS-B',options=dict(gtol=1e-12,ftol=1e-15,maxiter=2000))
independent_fit_error=float(np.max(abs(P[ite].dot(sol.x)-r['scores'])));assert independent_fit_error<1e-6
# Independent full Gram solve for the RBF conditional covariance, with no spectral truncation.
r=next(r for r in read(ROOT/'raw/rbf.jsonl') if (r['n'],r['rep'],r['tau'],r['delta'])==(128,0,.02,1.))
rng=np.random.RandomState(4930000+128000);a=np.array([.6,.4]);x=(-1+np.sqrt((1-a)**2+4*a*rng.rand(128,2)))/a
v,w=leggauss(32);xx,yy=np.meshgrid(v,v,indexing='ij');wx,wy=np.meshgrid(w,w,indexing='ij');xt=np.column_stack([xx.ravel(),yy.ravel()]);wt=(wx*wy).ravel()*(1+.6*xt[:,0])*(1+.4*xt[:,1])/4
def kernel(x,z):return np.exp(-np.sum((x[:,None,:]-z[None,:,:])**2,axis=2)/(2*.7**2))
Q=(wt[:,None]*np.column_stack([np.ones(len(xt)),xt])).T.dot(kernel(xt,x));s=expit(1.);p=s*(1-s)
G=solve(.02*np.eye(128)+p*kernel(x,x)/128,Q.T,assume_a='pos');C=s*s*G.T.dot(G)/128
conditional_error=float(np.max(abs(C-np.array(r['conditional']))));assert conditional_error<1e-9
records=read(ROOT/'raw/rbf.jsonl')+read(ROOT/'raw/bandwidth.jsonl')+rr
assert len(records)==4000 and max(r['residual'] for r in records)<1e-9
for row in json.loads((ROOT/'results/decomposition.json').read_text()):
    assert abs(row['true_second_moment']-row['population']-row['nonlinear_bias']-row['design_bias']-row['label_mc'])<1e-10
provenance=dict(dataset='UCI Wisconsin Diagnostic Breast Cancer, bundled sklearn copy',official_urls=['https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html','https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic'],sklearn_version=sklearn.__version__,features_sha256=hashlib.sha256(np.ascontiguousarray(data.data).tobytes()).hexdigest(),labels_sha256=hashlib.sha256(np.ascontiguousarray(data.target).tobytes()).hexdigest(),hash_encoding='NumPy native float64 features and native target integer array, C-order; local runtime little endian',feature_shape=list(data.data.shape),target_dtype=str(data.target.dtype),split_seed=927,train_size=398,test_size=171,bootstrap_seeds='882000+rep',standardization_fit='fixed training pool only, frozen across bootstrap fits',test_access='evaluation only; no hyperparameter selection',selection_reason='packaged real binary-classification features directly match the scalar logistic model; digits was a tentative suggestion, not a selected protocol dataset')
(ROOT/'results/data_provenance.json').write_text(json.dumps(provenance,indent=2))
result=dict(status='PASS',independent_lbfgs_score_error=independent_fit_error,independent_full_gram_conditional_error=conditional_error,records=4000,split_disjoint=True,preprocessing_train_only=True)
(ROOT/'results/independent_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
