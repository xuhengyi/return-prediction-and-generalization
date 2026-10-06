"""Metric conventions, numerically stable ridge, and strict CSV validation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import svd


def load_config(path='config.json'):
    return json.loads(Path(path).read_text())


def metrics(y, yhat, beta=None):
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    if y.ndim != 1 or y.shape != yhat.shape or not np.isfinite(y).all() or not np.isfinite(yhat).all():
        raise ValueError('Targets and predictions must be equal-length finite vectors')
    pnl = y * yhat
    second = np.mean(pnl ** 2)
    variance = np.var(pnl)
    yy = np.mean(y ** 2)
    return dict(r2=1-np.mean((y-yhat)**2)/yy if yy else np.nan,
                timing_mean=float(pnl.mean()),
                sharpe=float(pnl.mean()/np.sqrt(second)) if second else 0.,
                sharpe_var=float(pnl.mean()/np.sqrt(variance)) if variance else 0.,
                mse=float(np.mean((y-yhat)**2)),
                norm2=float(np.dot(beta,beta)) if beta is not None else np.nan)


def ridge_path(x, y, zs):
    """Minimize ||y-Xb||²/n + z||b||²; no intercept, SVD minimum norm at z=0."""
    u, s, vt = svd(x, full_matrices=False, check_finite=False)
    uy = u.T @ y
    cutoff = np.finfo(float).eps * max(x.shape) * s[0]
    result = []
    for z in zs:
        if z < 0:
            raise ValueError('z must be nonnegative')
        if z == 0:
            inverse = np.divide(1., s, out=np.zeros_like(s), where=s > cutoff)
        else:
            inverse = s / (s*s + len(y)*z)
        result.append(vt.T @ (inverse*uy))
    return np.column_stack(result)


def read_pair(data_dir, pair):
    path = Path(data_dir)
    train_path, test_path = path/f'pair{pair}_train.csv', path/f'pair{pair}_test_features.csv'
    # Check raw headers as pandas otherwise silently renames duplicate columns.
    for p in (train_path,test_path):
        with p.open(encoding='utf-8') as handle:
            headers = handle.readline().strip().split(',')
        if len(headers) != len(set(headers)):
            raise ValueError(f'Duplicate headers in {p.name}')
    train, test = pd.read_csv(train_path), pd.read_csv(test_path)
    features = [c for c in train.columns if c.startswith('feature')]
    if not features or list(train.columns) != ['t']+features+['return']:
        raise ValueError('Unexpected training schema')
    if set(test.columns) != set(['t']+features):
        raise ValueError('Missing or extra test columns')
    for frame in (train,test):
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError('Nonfinite data')
        t = frame.t.to_numpy()
        if not np.equal(t,np.floor(t)).all() or not (np.diff(t)>0).all():
            raise ValueError('t must contain strictly increasing integers')
    return train[features].to_numpy(), train['return'].to_numpy(), test[features].to_numpy(), test.t.to_numpy(dtype=np.int64), features


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False, default=lambda x: x.item()), encoding='utf-8')
