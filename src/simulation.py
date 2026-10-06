"""Tasks 1/2: paired fixed-grid finite-sample experiments, with oracle reference."""
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import lasso_path
from sklearn.exceptions import ConvergenceWarning
from .core import metrics, ridge_path


def run_simulations(cfg, output, quick=False):
    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    records=[]; convergence=[]; started=time.time()
    n=cfg['train_size']; p=n*cfg['true_complexity']; nt=cfg['test_size']
    strengths=cfg['strength_runs'] if not quick else {'1.0':2}
    for strength,nruns in strengths.items():
        strength=float(strength)
        for run in range(nruns):
            rng=np.random.default_rng(cfg['seed']+run+round(strength*10000))
            x=rng.standard_normal((n+nt,p))
            beta=rng.standard_normal(p); beta*=np.sqrt(strength)/np.linalg.norm(beta)
            y=x@beta+rng.standard_normal(n+nt)
            perm=rng.permutation(p)
            for cq in cfg['observed_complexities']:
                p1=int(round(cq*n)); q=p1/p
                xr=x[:n,perm[:p1]]; xt=x[n:,perm[:p1]]
                zstar=cfg['true_complexity']*(1+strength*(1-q))/strength
                zs=cfg['ridge_z']+[zstar]
                betas=ridge_path(xr,y[:n],zs)
                predictions=xt@betas
                for j,z in enumerate(zs):
                    model='ridge_oracle' if j==len(zs)-1 else 'ridge'
                    result=metrics(y[n:],predictions[:,j],betas[:,j])
                    records.append(dict(strength=strength,run=run,cq=p1/n,q=q,model=model,penalty=float(z),**result))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always',ConvergenceWarning)
                    alphas, coeffs, gaps=lasso_path(np.asfortranarray(xr),y[:n],alphas=cfg['lasso_alpha'],tol=1e-6,max_iter=15000)
                convergence.append(dict(strength=strength,run=run,cq=cq,max_dual_gap=float(max(gaps)),warnings=len(caught)))
                predictions=xt@coeffs
                for j,alpha in enumerate(alphas):
                    result=metrics(y[n:],predictions[:,j],coeffs[:,j])
                    records.append(dict(strength=strength,run=run,cq=p1/n,q=q,model='lasso',penalty=float(alpha),**result))
            print(f'Simulation b={strength:g}, run {run+1}/{nruns}, elapsed {time.time()-started:.1f}s',flush=True)
            pd.DataFrame(records).to_csv(output/'simulation_raw.csv',index=False)
    frame=pd.DataFrame(records)
    columns=['r2','timing_mean','sharpe','sharpe_var','norm2','mse']
    # Oracle penalty changes with cq: label by model instead of blending fixed z curves.
    frame['curve']=np.where(frame.model=='ridge_oracle','oracle',frame.model+'_'+frame.penalty.astype(str))
    summary=frame.groupby(['strength','model','curve','cq'])[columns].agg(['mean','std','count'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(output/'simulation_summary.csv')
    pd.DataFrame(convergence).to_csv(output/'lasso_convergence.csv',index=False)
    return frame
