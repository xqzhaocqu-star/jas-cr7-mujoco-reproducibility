#!/usr/bin/env python3
"""Reproduce the explicitly declared exponential-identification extension."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def run(script, *args):
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, '-B', str(HERE/script), *map(str, args)],
                   cwd=HERE, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-simulation', action='store_true')
    parser.add_argument('--skip-video', action='store_true')
    parser.add_argument('--refinement', action='store_true')
    parser.add_argument('--output', type=Path, default=HERE/'results_exponential')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    checked = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-p', 'no:cacheprovider', '-q',
                              'tests/test_algorithm.py', 'tests/test_exponential_identification.py'],
                             cwd=HERE, text=True, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, check=False,
                             env={**os.environ, 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
                                  'PYTHONDONTWRITEBYTECODE': '1'})
    print(checked.stdout, end='', flush=True)
    (output/'test_validation.txt').write_text(checked.stdout)
    if checked.returncode:
        raise SystemExit(checked.returncode)
    if not args.skip_simulation:
        run('run_identification.py', '--case', 'all', '--output', output)
        if args.refinement:
            run('run_identification.py', '--case', 'nominal', '--control-dt', '.00025',
                '--physics-dt', '.0000625', '--output', output/'step_refinement')
    run('validate_identification.py', '--results', output, '--case', 'all')
    if (output/'step_refinement/nominal/trajectory.npz').exists():
        run('validate_identification.py', '--results', output/'step_refinement', '--case', 'nominal',
            '--reference', output/'nominal')
    run('plot_identification.py', '--results', output)
    if not args.skip_video:
        run('render_identification.py', '--input', output/'nominal/trajectory.npz',
            '--output', output/'nominal/video.mp4')
    run('make_identification_report.py', '--results', output)


if __name__ == '__main__':
    main()
