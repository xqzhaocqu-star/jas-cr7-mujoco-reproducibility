#!/usr/bin/env python3
"""Reproduce simulations, independent validation, plots, video and report."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent


def run(script,*args):
    env=os.environ.copy()
    env['OPENBLAS_NUM_THREADS']='1'
    env['OMP_NUM_THREADS']='1'
    subprocess.run([sys.executable,str(HERE/script),*map(str,args)],cwd=HERE,env=env,check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-simulation',action='store_true',help='use existing raw logs')
    parser.add_argument('--skip-video',action='store_true')
    parser.add_argument('--refinement',action='store_true',help='also run the half-step nominal case')
    parser.add_argument('--robust-mode',choices=('paper_sign','boundary_layer'),default='paper_sign')
    parser.add_argument('--robust-width',type=float,default=.05)
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args()
    if not (HERE/'models/dual_cr7.xml').exists():
        run('build_scene.py')
    results=(args.output or HERE/('results' if args.robust_mode=='paper_sign'
                                 else 'results_low_chatter')).resolve()
    results.mkdir(parents=True,exist_ok=True)
    feedback_args=('--robust-mode',args.robust_mode,'--robust-width',args.robust_width)
    checked=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_algorithm.py'],
                           cwd=HERE,text=True,stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT,check=False)
    print(checked.stdout,end='',flush=True)
    (results/'test_validation.txt').write_text(checked.stdout)
    if checked.returncode:
        raise SystemExit(checked.returncode)
    if not args.skip_simulation:
        run('run_experiments.py','--case','all',*feedback_args,'--output',results)
    if args.refinement and not args.skip_simulation:
        run('run_experiments.py','--case','nominal','--control-dt','.00025',
            '--physics-dt','.000125',*feedback_args,'--output',results/'step_refinement')
    run('validate_results.py','--results',results)
    fine=results/'step_refinement/nominal/trajectory.npz'
    if fine.exists():
        run('validate_results.py','--results',results/'step_refinement','--case','nominal',
            '--reference',results/'nominal')
    run('plot_results.py','--results',results)
    if not args.skip_video:
        run('render_results.py','--input',results/'nominal/trajectory.npz',
            '--output',results/'nominal/video.mp4')
    run('make_report.py','--results',results)


if __name__=='__main__':
    main()
