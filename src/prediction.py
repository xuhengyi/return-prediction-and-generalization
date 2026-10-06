"""Task 3: chronological nested validation followed by full-data refitting."""
from pathlib import Path
import hashlib
import warnings
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.exceptions import ConvergenceWarning
from .core import metrics, ridge_path, read_pair, save_json


def candidate_predictions(x,y,xt,cfg):
    """All transformations depend on the fitting subset. No centering/intercept.

    Column RMS scaling equalizes penalties while preserving the zero intercept.
    Kernel normalization K/n means z shares the ridge objective convention.
    """
    scale=np.sqrt(np.mean(x*x,axis=0)); scale=np.where(scale>1e-12,scale,1)
    x=x/scale; xt=xt/scale
    result={'zero':np.zeros(len(xt)), 'historical_mean':np.full(len(xt),y.mean())}
    betas=ridge_path(x,y,cfg['ridge_z'])
    for j,z in enumerate(cfg['ridge_z']):
        result[f'ridge_z={z:g}']=xt@betas[:,j]
    for alpha in cfg['lasso_alpha']:
        model=Lasso(alpha=alpha,fit_intercept=False,max_iter=30000,tol=1e-6).fit(x,y)
        result[f'lasso_a={alpha:g}']=model.predict(xt)
    for alpha in cfg['elastic_alpha']:
        model=ElasticNet(alpha=alpha,l1_ratio=.2,fit_intercept=False,max_iter=30000,tol=1e-6).fit(x,y)
        result[f'elastic_a={alpha:g}']=model.predict(xt)
    for factor in cfg['rbf_gamma_factor']:
        k=rbf_kernel(x,gamma=factor/x.shape[1]); kt=rbf_kernel(xt,x,gamma=factor/x.shape[1])
        vals,vec=eigh(k,check_finite=False); vals=np.maximum(vals,0)
        for z in cfg['rbf_z']:
            dual=vec@((vec.T@y)/(vals+len(y)*z))
            result[f'rbf_g={factor:g}_z={z:g}']=kt@dual
    return result


def rolling_select(x,y,cfg):
    boundaries=[int(len(y)*fraction) for fraction in (.4,.6,.8,1)]
    records=[]; pooled={}; actual=[]
    for fold,(end,stop) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        predictions=candidate_predictions(x[:end],y[:end],x[end:stop],cfg)
        actual.append(y[end:stop])
        for name,pred in predictions.items():
            records.append(dict(model=name,fold=fold,train_n=end,validation_n=stop-end,**metrics(y[end:stop],pred)))
            pooled.setdefault(name,[]).append(pred)
    yval=np.concatenate(actual)
    ranking=[]
    for name,preds in pooled.items():
        ranking.append(dict(model=name,**metrics(yval,np.concatenate(preds))))
    ranking=pd.DataFrame(ranking).sort_values(['mse','model']).reset_index(drop=True)
    # Constants are diagnostic baselines, excluded from model selection.
    # The chosen estimator sets exposure, with no post-selection scaling.
    selected=ranking.loc[~ranking.model.isin(['zero','historical_mean'])].iloc[0]['model']
    exposure=1.
    return selected,exposure,ranking,pd.DataFrame(records)


def run_predictions(cfg,data_dir,output,student_id):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    decisions={}; allrank=[]; allfold=[]; audit=[]
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        for pair in 'ABC':
            x,y,xt,t,features=read_pair(data_dir,pair)
            n=len(y); outer_end=int(.8*n)
            name,exposure,ranking,folds=rolling_select(x[:outer_end],y[:outer_end],cfg)
            outer_preds=candidate_predictions(x[:outer_end],y[:outer_end],x[outer_end:],cfg)
            outer=metrics(y[outer_end:],exposure*outer_preds[name])
            outer.pop('norm2')
            # Unbiased outer score belongs to the selection pipeline. Final
            # re-selection is allowed on all public training labels; its score
            # is explicitly labelled selection CV, not an unbiased test score.
            final_name,final_exposure,final_rank,final_folds=rolling_select(x,y,cfg)
            final=candidate_predictions(x,y,xt,cfg)[final_name]*final_exposure
            pd.DataFrame({'t':t,'yhat':final}).to_csv(output/f'{student_id}_predictions_{pair}.csv',index=False,float_format='%.17g')
            decisions[pair]=dict(outer_selected=name,outer_exposure=exposure,outer_metrics=outer,
                                 final_selected=final_name,final_exposure=final_exposure,
                                 selection_cv_r2=float(final_rank.loc[final_rank.model==final_name,'r2'].iloc[0]),
                                 n_train=n,n_test=len(t),features=len(features))
            for frame,stage in [(ranking,'inner_80pct'),(final_rank,'final_all_train')]:
                frame=frame.copy();frame['pair']=pair;frame['stage']=stage;allrank.append(frame)
            for frame,stage in [(folds,'inner_80pct'),(final_folds,'final_all_train')]:
                frame=frame.copy();frame['pair']=pair;frame['stage']=stage;allfold.append(frame)
            audit.append(dict(pair=pair,train_rows=n,test_rows=len(t),features=len(features),
                              train_y_mean=float(y.mean()),train_y_sd=float(y.std()),
                              train_x_mean=float(x.mean()),train_x_sd=float(x.std()),
                              prediction_sd=float(final.std()),prediction_mean=float(final.mean()),
                              test_t_first=int(t[0]),test_t_last=int(t[-1])))
            print(f'Pair {pair}: inner selected {name}, outer R2={outer["r2"]:.5f}; final {final_name}, exposure={final_exposure}',flush=True)
    pd.concat(allrank).to_csv(output/'validation_ranking.csv',index=False)
    pd.concat(allfold).to_csv(output/'validation_folds.csv',index=False)
    pd.DataFrame(audit).to_csv(output/'data_audit.csv',index=False)
    save_json(output/'model_decisions.json',decisions)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(data_dir).glob('*.csv')}
    save_json(output/'input_sha256.json',hashes)
    return decisions
