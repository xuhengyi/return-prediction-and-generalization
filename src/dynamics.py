"""Task 4: finite-network illustration, not a replication of DMFT asymptotics."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .core import metrics


def run_dynamics(cfg, output, quick=False):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    records=[]; selected=[]
    d=cfg['dimension'];n=cfg['train_size'];nv=cfg['validation_size'];nt=cfg['test_size']
    steps=cfg['steps'] if not quick else 160
    for run in range(cfg['runs'] if not quick else 1):
        rng=np.random.default_rng(cfg['seed']+run)
        u=rng.normal(size=d);u/=np.linalg.norm(u)
        v=rng.normal(size=d);v-=u*(u@v);v/=np.linalg.norm(v)
        shifted=.5*u+np.sqrt(.75)*v
        x=rng.normal(size=(n+nv+nt,d)).astype('float32')
        noise=rng.normal(size=len(x))*cfg['noise_sd']
        y=(1.5*np.tanh(x@u)+noise).astype('float32')
        drift=(1.5*np.tanh(x[n+nv:]@shifted)+noise[n+nv:]).astype('float32')
        tx=torch.from_numpy(x);ty=torch.from_numpy(y)
        for width in cfg['widths']:
            torch.manual_seed(cfg['seed']+1000*run+width)
            w=torch.randn(width,d);w=w/w.norm(dim=1,keepdim=True);w.requires_grad_()
            a=torch.ones(width,requires_grad=True)
            trajectories=[]
            for step in range(steps+1):
                if step%cfg['record_every']==0 or step==steps:
                    with torch.no_grad():
                        pred=((torch.tanh(tx@w.T)@a)/width).numpy()
                        alignment=float(torch.mean(torch.abs(w@torch.tensor(u,dtype=torch.float32))))
                        norm=float(a.abs().mean())
                    for split,truth,predicted in [('train',y[:n],pred[:n]),('validation',y[n:n+nv],pred[n:n+nv]),
                                                 ('stationary',y[n+nv:],pred[n+nv:]),('shifted',drift,pred[n+nv:])]:
                        item=dict(run=run,width=width,step=step,time=step*cfg['step_size'],split=split,
                                  output_l1=norm,alignment=alignment,**metrics(truth,predicted))
                        records.append(item);trajectories.append(item)
                if step==steps:break
                fitted=torch.tanh(tx[:n]@w.T)@a/width
                loss=.5*torch.mean((fitted-ty[:n])**2);loss.backward()
                with torch.no_grad():
                    # Euler approximation to dtheta/dt_hat=-m P grad empirical risk.
                    projected=w.grad-(w.grad*w).sum(dim=1,keepdim=True)*w
                    w-=cfg['step_size']*width*projected
                    w/=w.norm(dim=1,keepdim=True)
                    a-=cfg['step_size']*width*a.grad
                    w.grad.zero_();a.grad.zero_()
            best=min((r for r in trajectories if r['split']=='validation'),key=lambda r:r['mse'])['step']
            for policy,checkpoint in [('validation_stop',best),('last',steps)]:
                for r in trajectories:
                    if r['step']==checkpoint and r['split'] in ('stationary','shifted'):
                        selected.append(dict(policy=policy,**r))
            print(f'Neural illustration run={run+1}, width={width}, validation checkpoint={best}/{steps}',flush=True)
            Path(output).mkdir(parents=True,exist_ok=True)
            pd.DataFrame(records).to_csv(Path(output)/'dynamics_raw.csv',index=False)
            pd.DataFrame(selected).to_csv(Path(output)/'dynamics_checkpoints.csv',index=False)
    return pd.DataFrame(records)
