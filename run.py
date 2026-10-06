"""Run from the project root: python run.py --task all."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse
from pathlib import Path
from src.core import load_config


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',choices=['simulation','prediction','dynamics','figures','report','all'],default='all')
    parser.add_argument('--config',default='config.json')
    parser.add_argument('--data-dir',default='Project1_export')
    parser.add_argument('--output',default='outputs')
    parser.add_argument('--quick',action='store_true',help='Smoke experiment only; not the report results')
    args=parser.parse_args();cfg=load_config(args.config)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if args.quick and args.output=='outputs':
        parser.error('--quick requires a distinct --output to protect full results')
    if args.task in ('simulation','all'):
        from src.simulation import run_simulations
        run_simulations(cfg['simulation'],out,args.quick)
    if args.task in ('prediction','all'):
        from src.prediction import run_predictions
        run_predictions(cfg['prediction'],args.data_dir,out,cfg['student']['id'])
    if args.task in ('dynamics','all'):
        from src.dynamics import run_dynamics
        run_dynamics(cfg['dynamics'],out,args.quick)
    if args.task in ('figures','all'):
        from src.figures import make_figures
        make_figures(out)
    if args.task in ('report','all'):
        from src.report import build_report
        build_report(cfg,out)


if __name__=='__main__':
    main()
