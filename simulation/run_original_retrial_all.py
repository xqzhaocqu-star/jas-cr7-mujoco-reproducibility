#!/usr/bin/env python3
"""Reproduce and independently check the paper's original JAS retrial."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'results_original_retrial')
    parser.add_argument('--config', type=Path, default=HERE/'configs/original_retrial.json')
    parser.add_argument('--duration', type=float, default=24.)
    parser.add_argument('--refinement', action='store_true')
    parser.add_argument('--skip-simulation', action='store_true')
    parser.add_argument('--skip-video', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')

    def run(script, *arguments):
        subprocess.run([sys.executable, '-B', str(HERE/script), *map(str, arguments)],
                       cwd=HERE, env=env, check=True)

    if not args.skip_simulation:
        subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                        str(HERE/'tests/test_algorithm.py'), str(HERE/'tests/test_original_retrial.py')],
                       cwd=HERE, env=env, check=True)
        run('run_original_retrial.py', '--case', 'all', '--config', args.config,
            '--duration', args.duration, '--output', output)
        if args.refinement:
            run('run_original_retrial.py', '--case', 'nominal', '--config', args.config,
                '--duration', args.duration, '--control-dt', '.00025', '--physics-dt', '.0000625',
                '--output', output/'step_refinement')
    run('validate_original_retrial.py', '--results', output, '--duration-s', args.duration)
    if (output/'step_refinement/nominal/source_manifest.json').exists():
        run('validate_original_retrial.py', '--results', output/'step_refinement', '--case', 'nominal',
            '--duration-s', args.duration, '--reference', output/'nominal')
    run('diagnose_original_retrial.py', '--results', output)
    if (output/'step_refinement/nominal/source_manifest.json').exists():
        run('diagnose_original_retrial.py', '--results', output/'step_refinement', '--case', 'nominal')
    run('plot_original_retrial.py', '--results', output)
    if not args.skip_video:
        run('render_results.py', '--input', output/'nominal/trajectory.npz',
            '--output', output/'nominal/video.mp4')
    run('make_original_retrial_report.py', '--results', output)


if __name__ == '__main__':
    main()
