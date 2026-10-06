"""Reproduce the analysis in a local environment and record command logs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
EVENTS=ROOT/'outputs/clean_run_events.json'


def recorded(command,label):
    events=json.loads(EVENTS.read_text()) if EVENTS.exists() else []
    beginning=time.time()
    events.append(dict(label=label,elapsed=0.,text='$ '+' '.join(str(c) for c in command)))
    process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
    for line in process.stdout:
        events.append(dict(label=label,elapsed=time.time()-beginning,text=line.rstrip()))
        print(line.rstrip(),flush=True)
    code=process.wait()
    events.append(dict(label=label,elapsed=time.time()-beginning,text=f'Exit status: {code}; wall time: {time.time()-beginning:.2f} seconds'))
    EVENTS.parent.mkdir(exist_ok=True)
    EVENTS.write_text(json.dumps(events,indent=2),encoding='utf-8')
    if code:raise SystemExit(code)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    venv=ROOT/'.venv';python=venv/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if args.prepare:
        if venv.exists():raise SystemExit('Refusing to overwrite an existing .venv; inspect it first.')
        recorded([sys.executable,'-m','venv',str(venv)],'fresh_environment')
        recorded([str(python),'-m','pip','install','-r','requirements.txt'],'install_dependencies')
    else:
        recorded([str(python),'-m','unittest','discover','-s','tests','-v'],'numerical_tests')
        recorded([str(python),'run.py','--task','all','--output','reproduced_outputs'],'full_clean_run')
        recorded([str(python),'scripts/validate.py','--compare','reproduced_outputs'],'validate_artifacts')


if __name__=='__main__':main()
