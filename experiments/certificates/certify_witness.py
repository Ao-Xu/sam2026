"""Interval certificate for a fixed rational-feature covariance derivative."""
import mpmath as mp
import json
from pathlib import Path
mp.iv.dps=60
iv=mp.iv
z=iv.mpf(0); one=iv.mpf(1)
def add(A,B): return [[A[i][j]+B[i][j] for j in range(2)] for i in range(2)]
def neg(A): return [[-a for a in r] for r in A]
def mul(A,B): return [[sum((A[i][k]*B[k][j] for k in range(2)),z) for j in range(2)] for i in range(2)]
def inv(A):
    d=A[0][0]*A[1][1]-A[0][1]*A[1][0]
    return [[A[1][1]/d,-A[0][1]/d],[-A[1][0]/d,A[0][0]/d]]
def weighted(weights):
    return [[sum((w*x[i]*x[j]/3 for w,x in zip(weights,features)),z) for j in range(2)] for i in range(2)]
features=[[-5,6],[-1,4],[-6,0]]
k=[iv.sqrt(sum(a*a for a in x)) for x in features]
delta=iv.mpf(2); tau=one/250
s=[one/(one+iv.exp(-delta*a)) for a in k]; p=[a*(one-a) for a in s]
H=add([[tau,z],[z,tau]],weighted(p)); R=inv(H)
Hp=weighted([a*b*(one-2*c) for a,b,c in zip(k,p,s)])
B=weighted([a*a for a in s]); Bp=weighted([2*a*b*c for a,b,c in zip(k,s,p)])
C=mul(mul(R,B),R)
Cp=add(add(mul(mul(R,Bp),R),neg(mul(mul(R,Hp),C))),neg(mul(mul(C,Hp),R)))
v=[-5,-3]
q=sum((v[i]*Cp[i][j]*v[j] for i in range(2) for j in range(2)),z)
assert q.b<0
record={'features':features,'probabilities':'1/3 each','tau':'1/250','delta':'2','direction':v,'precision_digits':60,'quadratic_derivative_interval':str(q),'matrix_derivative_intervals':[[str(a) for a in r] for r in Cp]}
print(json.dumps(record,indent=2))
Path(__file__).with_name('witness_certificate.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
