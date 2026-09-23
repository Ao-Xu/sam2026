"""Outward interval automatic differentiation for the cubic coefficient.
This certifies a signal-independent radius, not a numeric epsilon threshold.
"""
import mpmath as mp
import json
from pathlib import Path
mp.iv.dps=40
iv=mp.iv
class J:
    def __init__(self,v,d=0,dd=0): self.v,self.d,self.dd=iv.mpf(v),iv.mpf(d),iv.mpf(dd)
    def __add__(a,b):
        b=b if isinstance(b,J) else J(b)
        return J(a.v+b.v,a.d+b.d,a.dd+b.dd)
    __radd__=__add__
    def __neg__(a): return J(-a.v,-a.d,-a.dd)
    def __sub__(a,b): return a+-asj(b)
    def __rsub__(a,b): return asj(b)+-a
    def __mul__(a,b):
        b=asj(b)
        return J(a.v*b.v,a.d*b.v+a.v*b.d,a.dd*b.v+2*a.d*b.d+a.v*b.dd)
    __rmul__=__mul__
    def inv(a): return J(1/a.v,-a.d/a.v**2,2*a.d**2/a.v**3-a.dd/a.v**2)
    def __truediv__(a,b): return a*asj(b).inv()
    def __rtruediv__(a,b): return asj(b)*a.inv()
    def exp(a):
        e=iv.exp(a.v)
        return J(e,e*a.d,e*(a.d**2+a.dd))
def asj(x): return x if isinstance(x,J) else J(x)
def coeff(delta,tau,k):
    mu=J(1/(1+iv.pi**4/45)/(k**4)); mu3=mu/81
    tau=J(tau)
    s=1/(1+(-delta).exp()); p=s*(1-s); a=mu*s/(tau+mu*p)
    return mu3*(p*(1-2*s)*a*a/8-p*(1-6*p)*a*a*a/24)/(tau+mu3*p)
rows=[]
for tau_s in ['.005','.02','.08']:
    for k in [1,2]:
        tau=iv.mpf(tau_s); D=iv.mpf('.1')
        b0=coeff(J(0,1),tau,k); bound=coeff(J([0,'.1'],1),tau,k)
        L0=abs(bound.d); L1=abs(bound.dd)
        # Lower endpoints for positive margins, upper endpoints for derivative bounds.
        r=min(D.a,(b0.v.a/(2*L0.b)).a,(-b0.d.b/(2*L1.b)).a)
        assert r>0
        rows.append(dict(tau=tau_s,k=k,D='.1',b0=str(b0.v),minus_b0_prime=str(-b0.d),L0=str(L0.b),L1=str(L1.b),certified_radius_lower_bound=str(r),epsilon_threshold='requires full RKHS C and epsilon_0'))
out=Path(__file__).with_name('radius_certificate.json')
out.write_text(json.dumps(rows,indent=2),encoding='utf-8')
print(json.dumps(rows,indent=2))
