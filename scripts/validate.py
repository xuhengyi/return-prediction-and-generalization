"""Validate deliverables and compare a clean execution numerically."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from pypdf import PdfReader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import read_pair,load_config,save_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--compare');parser.add_argument('--output',default='outputs');args=parser.parse_args()
    cfg=load_config();out=Path(args.output);student=cfg['student']['id'];checks={}
    for pair in 'ABC':
        _,_,_,t,_=read_pair('Project1_export',pair)
        p=pd.read_csv(out/f'{student}_predictions_{pair}.csv')
        assert list(p)==['t','yhat']
        np.testing.assert_array_equal(p.t,t)
        assert np.isfinite(p.yhat).all()
        checks[f'pair_{pair}']=dict(rows=len(p),finite=True,exact_schema=True,order_matches=True)
    reader=PdfReader(out/f'{student}_report.pdf')
    assert 5<=len(reader.pages)<=15,f'Report length {len(reader.pages)}'
    for i,page in enumerate(reader.pages):
        assert len(page.extract_text())>180,f'Unexpected near-blank page {i+1}'
    checks['report']=dict(pages=len(reader.pages),nonempty=True)
    sim=pd.read_csv(out/'simulation_raw.csv')
    expected=sum(cfg['simulation']['strength_runs'].values())*len(cfg['simulation']['observed_complexities'])*(len(cfg['simulation']['ridge_z'])+1+len(cfg['simulation']['lasso_alpha']))
    assert len(sim)==expected
    assert np.isfinite(sim[['r2','sharpe','timing_mean','norm2']]).all().all()
    conv=pd.read_csv(out/'lasso_convergence.csv')
    assert conv.warnings.sum()==0
    checks['simulation']=dict(rows=len(sim),expected=expected,convergence_warnings=0)
    dyn=pd.read_csv(out/'dynamics_raw.csv')
    train=dyn[dyn.split=='train']
    increases=train.groupby(['run','width']).mse.apply(lambda s:float(np.diff(s).max()))
    assert increases.max()<1e-5,f'Unstable training dynamics: {increases.max()}'
    checks['dynamics']=dict(rows=len(dyn),max_recorded_train_mse_increase=float(increases.max()))
    if args.compare:
        other=Path(args.compare)
        comparisons={}
        for name in ['simulation_raw.csv','dynamics_raw.csv','dynamics_checkpoints.csv','validation_ranking.csv']+[f'{student}_predictions_{p}.csv' for p in 'ABC']:
            a,b=pd.read_csv(out/name),pd.read_csv(other/name)
            assert a.shape==b.shape,(name,a.shape,b.shape)
            numeric=a.select_dtypes(include='number').columns
            # CPU builds may differ in BLAS/kernel rounding. Identical random
            # datasets and model choices are required; allow float32 dynamics noise.
            tol=2e-4 if name.startswith('dynamics') else 1e-8
            np.testing.assert_allclose(a[numeric],b[numeric],rtol=tol,atol=tol,equal_nan=True,err_msg=name)
            for col in a.columns.difference(numeric):
                assert a[col].equals(b[col]),(name,col)
            dif=np.abs(a[numeric].to_numpy()-b[numeric].to_numpy())
            comparisons[name]=float(np.nanmax(dif))
        first=json.loads((out/'model_decisions.json').read_text());second=json.loads((other/'model_decisions.json').read_text())
        assert all(first[p]['final_selected']==second[p]['final_selected'] for p in 'ABC')
        checks['clean_reproduction_max_absolute_differences']=comparisons
    save_json(out/'validation_report.json',checks)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
