import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def savefig(fig,out):
    fig.tight_layout()
    fig.savefig(out/'preview.png',dpi=160,bbox_inches='tight')
    plt.close(fig)

from scipy.optimize import curve_fit
def response(t,t0,rise,tau):return t0+rise*(1-np.exp(-np.asarray(t)/tau))
def fit_response(t,y):
    t,y=np.asarray(t,float),np.asarray(y,float)
    if t.ndim!=1 or t.shape!=y.shape or len(t)<10 or not np.all(np.isfinite([t,y])) or np.any(np.diff(t)<=0) or t[0]<0 or np.ptp(y)<1e-8:raise ValueError('Need finite increasing nonnegative times and a nonconstant temperature trace.')
    p,_=curve_fit(response,t,y,p0=[y[0],max(y[-1]-y[0],.1),max(np.ptp(t)/3,.1)],bounds=([-100,0,1e-4],[1000,2000,1e6]),maxfev=20000);return p
def example(out,seed):
    t=np.linspace(0,180,361);truth=[22,15,35];y=response(t,*truth)+np.random.default_rng(seed).normal(0,.15,len(t));cut=t<=110;p=fit_response(t[cut],y[cut]);pred=response(t,*p)
    fig,ax=plt.subplots(2,1,figsize=(9,6),sharex=True,gridspec_kw={'height_ratios':[3,1]});ax[0].plot(t,y,'.',ms=2,label='Synthetic measurements');ax[0].plot(t,pred,label='Fitted first-order model');ax[0].axvline(110,color='k',ls='--',label='Training cutoff');ax[0].set_ylabel('Temperature (°C)');ax[0].legend()
    ax[1].plot(t,y-pred);ax[1].axhline(0,color='k',lw=.5);ax[1].set(xlabel='Time (s)',ylabel='Residual (°C)');savefig(fig,out)
    np.savetxt(out/'temperature.csv',np.column_stack([t,y,pred,cut.astype(int)]),delimiter=',',header='time_s,temperature_c,predicted_c,training',comments='')
    return {'baseline_c':float(p[0]),'temperature_rise_c':float(p[1]),'time_constant_s':float(p[2]),'settling_95_s':float(-p[2]*np.log(.05)),'heldout_rmse_c':float(np.sqrt(np.mean((y[~cut]-pred[~cut])**2)))}

def run(out='results',seed=42):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    metrics=example(out,seed);metrics.update(seed=seed,data='synthetic')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2));return metrics
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results');p.add_argument('--seed',type=int,default=42);a=p.parse_args();print(json.dumps(run(a.out,a.seed),indent=2))
