import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
from scipy.special import expit
from run_experiments import mu,ROOT

COLORS=['#0072B2','#D55E00','#009E73']
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})

def read(name):return [json.loads(s) for s in (ROOT/'raw'/name).read_text().splitlines() if s.strip()]
def arr(rows,key):return np.asarray([r[key] for r in rows])
def select(rows,**kw):return [r for r in rows if all(r.get(k)==v for k,v in kw.items())]
def save(fig,name):
    fig.tight_layout(pad=.8,rect=(0,.12,1,1) if fig.legends else (0,0,1,1))
    for ext in ['pdf','svg','png']:fig.savefig(str(ROOT/'figures'/(name+'.'+ext)),dpi=180,bbox_inches='tight',bbox_extra_artists=fig.legends)
    plt.close(fig)

def bootstrap_ratio(x,y,rng):
    b=rng.randint(0,len(x),(2000,len(x)))
    vals=np.var(x[b],axis=1,ddof=1)/np.var(y[b],axis=1,ddof=1)
    return [float(v) for v in np.percentile(vals,[2.5,97.5])]

def main():
    (ROOT/'figures').mkdir(exist_ok=True);(ROOT/'results').mkdir(exist_ok=True)
    spec=read('spectral.jsonl');asym=read('asymptotic.jsonl');bal=read('balanced.jsonl')
    assert len(spec)==630 and len(asym)==3000 and len(bal)==3000, 'Incomplete planned runs'
    assert len({(r['n'],r['rep'],r['gamma'],r['eta']) for r in asym})==3000
    assert len({(r['n'],r['rep'],r['delta']) for r in bal})==3000
    for n in [128,512,1024]:
        for d in [0.,.25,.5,1.]:
            assert sorted(r['rep'] for r in select(bal,n=n,delta=d))==list(range(200))
    rng=np.random.RandomState(20260912);summary={}
    # Structural response from actual population-integral quadrature.
    fig,ax=plt.subplots(1,3,figsize=(6.9,2.65))
    rows=sorted(select(spec,grid=1024,tau=.02,k=1,epsilon=.1),key=lambda z:z['delta'])
    ds=arr(rows,'delta');a=arr(rows,'fundamental');b=arr(rows,'third')
    ax[0].plot(ds,a/a[0],'o',color=COLORS[0],label='Optimized')
    ax[0].plot(ds,arr(rows,'predicted_fundamental')/rows[0]['predicted_fundamental'],'--',color='black',label='Cubic expansion')
    ax[0].set(xlabel=r'Radius $\delta$',ylabel='Fundamental amplitude ratio',title='(a) Existing signal')
    # Reserve an empty band above the data so the legend cannot cross the fit.
    ax[0].set_ylim(top=2.15)
    ax[0].legend(frameon=False,loc='upper left')
    ax[1].plot(ds,b/.1**3,'s',color=COLORS[1],label='Optimized')
    ax[1].plot(ds,arr(rows,'predicted_third')/.1**3,'--',color='black',label='Cubic expansion')
    ax[1].axhline(0,color='.75',lw=.7)
    ax[1].set(xlabel=r'Radius $\delta$',ylabel=r'Third harmonic / $\epsilon^3$',title='(b) Generated harmonic')
    rr=sorted(select(spec,grid=1024,tau=.02,k=1,delta=.25),key=lambda z:z['epsilon'])
    ee=arr(rr,'epsilon');err1=arr(rr,'leading_l2_error');err3=arr(rr,'cubic_l2_error')
    ax[2].loglog(ee,err1,'o-',color=COLORS[0],label='Linear approximation')
    ax[2].loglog(ee,err3,'s-',color=COLORS[1],label='Cubic approximation')
    ax[2].set(xlabel=r'Signal amplitude $\epsilon$',ylabel=r'$L^2$ error',title='(c) Expansion error')
    handles,labels=ax[2].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,loc='lower center',bbox_to_anchor=(.5,.005),ncol=2)
    save(fig,'spectral_response')
    summary['spectral']={'fundamental_ratio_delta1':float(a[-1]/a[0]),'third_values':list(zip(ds.tolist(),b.tolist())),
        'linear_error_slope':float(np.polyfit(np.log(ee[-3:]),np.log(err1[-3:]),1)[0]),
        'cubic_error_slope':float(np.polyfit(np.log(ee[-3:]),np.log(err3[-3:]),1)[0])}
    differences=[]
    for r in select(spec,grid=1024):
        old=select(spec,grid=512,tau=r['tau'],k=r['k'],epsilon=r['epsilon'],delta=r['delta'])[0]
        differences.extend([abs(r['fundamental']-old['fundamental']),abs(r['third']-old['third'])])
    summary['spectral']['max_512_1024_coefficient_difference']=float(max(differences))
    # Statistical response.
    fig,ax=plt.subplots(1,3,figsize=(6.9,2.75));slopes=[]
    ns=sorted(set(r['n'] for r in asym))
    for color,gamma in zip(COLORS,[.25,.5,.75]):
        means=[];ses=[];errors=[];samples=[]
        for n in ns:
            rr=select(asym,n=n,gamma=gamma,eta=.3);v=arr(rr,'scaled_distance');samples.append(v)
            means.append(v.mean());ses.append(v.std(ddof=1)/np.sqrt(len(v)));errors.append(arr(rr,'prediction_relative_error').mean())
        means=np.array(means);ses=np.array(ses)
        ax[0].loglog(ns,means,'o-',color=color,label=r'$\gamma={}$'.format(gamma))
        ax[0].fill_between(ns,means-ses,means+ses,color=color,alpha=.15)
        ax[1].loglog(ns,errors,'o-',color=color)
        slope=np.polyfit(np.log(ns),np.log(means),1)[0]
        bs=[]
        for _ in range(1000):
            mm=[rng.choice(v,len(v),replace=True).mean() for v in samples]
            bs.append(np.polyfit(np.log(ns),np.log(mm),1)[0])
        slopes.append({'gamma':gamma,'slope':float(slope),'theory':.5-gamma,'ci95':np.percentile(bs,[2.5,97.5]).tolist(),
                       'relative_error_nmax':float(errors[-1])})
    ax[0].set(xlabel=r'Sample size $n$',ylabel=r'$\sqrt{n}\|\hat f_\delta-\hat f_0\|_{\mathcal{H}}$',title='(a) Paired displacement')
    handles,labels=ax[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,loc='lower center',bbox_to_anchor=(.5,.005),ncol=3)
    ax[1].set(xlabel=r'Sample size $n$',ylabel='Relative first-order error',title='(b) Drift approximation')
    ratios=[];nmax=max(r['n'] for r in bal);ds=[.25,.5,1.]
    for j,color in zip([0,1,2],COLORS):
        base=sorted(select(bal,n=nmax,delta=0.,shrinking=False),key=lambda r:r['rep'])
        y=np.array([r['coords'][j] for r in base]);values=[];low=[];high=[]
        for delta in ds:
            rr=sorted(select(bal,n=nmax,delta=delta,shrinking=False),key=lambda r:r['rep'])
            x=np.array([r['coords'][j] for r in rr]);v=np.var(x,ddof=1)/np.var(y,ddof=1);ci=bootstrap_ratio(x,y,rng)
            s=expit(delta);th=(s/(.02+mu(j)*s*(1-s))/(.5/(.02+mu(j)/4)))**2
            values.append(v);low.append(v-ci[0]);high.append(ci[1]-v)
            ratios.append(dict(mode=j,delta=delta,ratio=float(v),theory=float(th),ci95=ci,n=nmax))
        offset=(j-1)*.016
        if j!=1:ax[2].errorbar(np.array(ds)+offset,values,yerr=[low,high],fmt='o',color=color,ms=3,capsize=2,label='Mode {}'.format(j))
        dd=np.linspace(0,1,100);ss=expit(dd)
        vv=(ss/(.02+mu(j)*ss*(1-ss))/(.5/(.02+mu(j)/4)))**2
        if j!=1:ax[2].plot(dd,vv,'--',color=color,lw=1)
    ax[2].set(xlabel=r'Fixed radius $\delta$',ylabel='Variance / zero-radius variance',title='(c) Balanced-label noise')
    ax[2].legend(frameon=False,loc='upper left')
    save(fig,'statistical_response')
    summary['slopes']=slopes;summary['variance_ratios']=ratios
    zero=[]
    for n in sorted(set(r['n'] for r in bal)):
        rr=select(bal,n=n,shrinking=True)
        zero.append(dict(n=n,scaled_norm_mean=float(arr(rr,'scaled_norm').mean()),paired_scaled_mean=float(arr(rr,'paired_scaled').mean())))
    summary['balanced_shrinking']=zero
    scalar=json.loads((ROOT/'raw'/'scalar.json').read_text())
    fig,ax=plt.subplots(1,2,figsize=(6.6,2.3))
    for name,color,mark in zip(['ordinary','samplewise','shared'],COLORS,['o','s','^']):
        rr=sorted(select(scalar,p=.8,method=name),key=lambda r:r['delta'])
        ax[0].plot(arr(rr,'delta'),arr(rr,'optimum'),mark+'-',color=color,label=name.title())
    ax[0].set(xlabel=r'Radius $\delta$',ylabel='Scalar optimum',title='(a) Different objectives')
    ax[0].legend(frameon=False)
    ax[1].plot([r['n'] for r in zero],[r['scaled_norm_mean'] for r in zero],'o-',color=COLORS[0],label='Estimator')
    ax[1].plot([r['n'] for r in zero],[r['paired_scaled_mean'] for r in zero],'s-',color=COLORS[1],label='Paired difference')
    ax[1].set(xlabel=r'Sample size $n$',ylabel=r'Mean root-$n$ norm',title=r'(b) Zero drift, $\gamma=1/4$')
    ax[1].set_xscale('log');ax[1].set_xticks([128,512,1024]);ax[1].set_xticklabels(['128','512','1024']);ax[1].xaxis.set_minor_locator(NullLocator());ax[1].legend(frameon=False)
    save(fig,'objective_and_zero_drift')
    summary['counts']={'asymptotic':len(asym),'balanced':len(bal),'spectral':len(spec),'scalar':len(scalar)}
    summary['max_projected_gradient_residual']=max(r['residual'] for r in asym+bal+spec)
    summary['max_rank_discarded']=max(r.get('discarded',0) for r in asym+spec)
    summary['conservative_omitted_gradient_bound']=float(np.sqrt(1e-13))
    (ROOT/'results'/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['% Generated from raw independent optimization results.','\\begin{tabular}{cccc}','\\toprule',
           r'$\gamma$ & Predicted slope & Measured slope & Bootstrap 95\% interval \\',r'\midrule']
    for z in slopes:
        lines.append('{:.2f} & {:+.2f} & {:+.3f} & [{:+.3f}, {:+.3f}] \\\\'.format(z['gamma'],z['theory'],z['slope'],*z['ci95']))
    lines+=['\\bottomrule','\\end{tabular}']
    (ROOT/'results'/'slopes_table.tex').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()

