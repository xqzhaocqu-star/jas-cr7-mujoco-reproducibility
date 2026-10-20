#!/usr/bin/env python3
"""Create a frozen comparison protocol and run its independent trials.

Initial-domain checks use nominal kinematics only. No candidate is selected by
unknown true parameters or by post-hoc removal of unsuccessful test cases.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

from jas.barrier import transform, terminal_surface
from jas.reference import envelope
from comparison.controller import ComparisonSettings, common_bounds, METHODS
from run_comparison import run_trial
from run_experiments import HERE, CASES, initialize, kinematics


def build_protocol(settings_path=None, external_settings=None):
    source = settings_path or HERE/'configs/original_retrial.json'
    settings = ComparisonSettings(**json.loads(Path(source).read_text()))
    # One nominal task calibration: identical original/linear surface initial
    # reference velocity, with no simulation or unknown-parameter access.
    error0 = np.asarray(CASES['nominal']['offset'])
    bs = transform(error0,np.zeros(6),envelope(0.,np.asarray(settings.chi),np.asarray(settings.ell)),
                   np.asarray(settings.chi),np.asarray(settings.ell))
    _, xir, _ = terminal_surface(bs.xi,bs.xidot,settings.lam1,settings.lam2,
                                 settings.alpha,settings.beta,settings.delta)
    settings.linear_surface_gain = tuple((-xir/bs.xi).tolist())
    scenarios = []
    for case, spec in CASES.items():
        scenarios.append(dict(id=case, study='main', duration_s=24.,
                              reference_profile='moderate', excitation_duration=20., **spec))
    large = np.asarray(CASES['large_initial_error']['offset'])
    for i, scale in enumerate((-.25, .25, -.5, .5, -1., 1., -1.5, 1.5)):
        scenarios.append(dict(id=f'initial_{i+1:02d}', study='initial_sweep', duration_s=8.,
                              reference_profile='hold', nonmonotonic=False,
                              offset=(large*scale).tolist(), initial_scale=float(scale)))
    meta = json.loads((HERE/'models/scene_metadata.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(HERE/'models/dual_cr7.xml'))
    data = mujoco.MjData(model)
    kins = [kinematics(meta,h) for h in ('left','right')]
    center = np.r_[meta['payload_nominal']['initial_position_world_m'], np.zeros(3)]
    upper, lower, *_ = common_bounds(0.,settings)
    limits = np.tile(np.asarray(meta['joint_position_limits_rad']), (2,1))
    for scenario in scenarios:
        initialize(model,data,meta,kins,center,scenario['offset'])
        minimum_sigma = min(np.linalg.svd(kin.pose_jacobian(data.qpos[i*6:i*6+6])[1],compute_uv=False)[-1]
                            for i,kin in enumerate(kins))
        minimum_joint = np.min(np.minimum(data.qpos[:12]-limits[:,0],limits[:,1]-data.qpos[:12]))
        e = np.asarray(scenario['offset'])
        margin = float(np.min(np.minimum((upper-e)/upper,(lower+e)/lower)))
        if minimum_sigma < .02 or minimum_joint < .08 or margin <= .05:
            raise RuntimeError(f'Pre-registered initial condition not jointly feasible: {scenario["id"]}')
        scenario['preflight'] = dict(minimum_jacobian_sigma=float(minimum_sigma),
                                     joint_limit_margin_rad=float(minimum_joint),
                                     initial_common_margin=margin,
                                     initial_grasp_residual=float(np.linalg.norm(data.efc_pos[:12])))
    trials = []
    for scenario in scenarios:
        methods = METHODS if scenario['study']=='main' else ('proposed','linear_surface','linear_reaching')
        for method in methods:
            path = f'{scenario["study"]}/{scenario["id"]}/{method}'
            trials.append(dict(id=f'{scenario["id"]}__{method}',method=method,scenario=scenario['id'],path=path))
    for method in ('proposed','external_ppc'):
        trials.append(dict(id=f'refinement__{method}',method=method,scenario='nominal',
                           path=f'step_refinement/nominal/{method}',control_dt_s=.00025,physics_dt_s=.0000625))
    return dict(schema_version=1, description='Full original JAS, four structural ablations and an independent complete cooperative PPC protocol',
                settings=asdict(settings), external_settings=external_settings,
                control_dt_s=.0005,physics_dt_s=.000125,integrator='RK4',
                calibration=dict(source='Nominal zero-velocity initial offset only; no plant truth',
                                 linear_surface_gain='g(xi0)/xi0, held fixed for every trial',
                                 log_bf_scale='matches original dxi/de at e=0,t=0, fixed thereafter'),
                common_boundary_note='During [0,1) s a common finite assessment bound is extra to the proposed native BF guarantee. From 1 s every JAS-based native/assessment boundary coincides except the log coordinate formula.',
                evaluation=dict(position_tolerance_m=.00025,orientation_coordinate_tolerance_rad=float(np.deg2rad(.05)),
                                surface_component_abs=.001,hold_s=2.,tolerance_multipliers=[.5,1.,2.],
                                failed_or_unsettled='Retain outcome and right censor; never remove from denominator'),
                scenarios=scenarios,trials=trials)


def _job(arguments):
    return run_trial(*arguments)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=HERE/'results_comparison')
    p.add_argument('--settings',type=Path)
    p.add_argument('--external-config',type=Path)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--only-study',choices=('main','initial_sweep','step_refinement'))
    p.add_argument('--only-method',choices=METHODS)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    protocol_path=output/'protocol.json'
    if protocol_path.exists():
        if not args.resume:
            raise FileExistsError('Protocol exists; use --resume to preserve its frozen settings.')
        protocol=json.loads(protocol_path.read_text())
    else:
        ext=json.loads(args.external_config.read_text()) if args.external_config else None
        protocol=build_protocol(args.settings,ext)
        protocol_path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n')
    print(f'Protocol: {protocol_path}; {len(protocol["trials"])} pre-registered trials',flush=True)
    if args.prepare_only:return
    jobs=[]
    for trial in protocol['trials']:
        if args.only_study and not trial['path'].startswith(args.only_study+'/'):continue
        if args.only_method and trial['method']!=args.only_method:continue
        target=output/trial['path']
        if (target/'source_manifest.json').exists():
            print(f'Preserving existing outcome: {trial["id"]}',flush=True)
            continue
        scenario=next(s for s in protocol['scenarios'] if s['id']==trial['scenario'])
        jobs.append((trial['method'],scenario,target,protocol['settings'],
                     trial.get('control_dt_s',protocol['control_dt_s']),
                     trial.get('physics_dt_s',protocol['physics_dt_s']),protocol.get('external_settings')))
    # Independent processes have separate MuJoCo models/data; BLAS is single threaded.
    started=time.perf_counter(); failures=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending={pool.submit(_job,j):j[2] for j in jobs}
        for future in as_completed(pending):
            target=pending[future]
            try:
                result=future.result()
                print(f'DONE {target.relative_to(output)}: {result["run_status"]} at {result["actual_end_s"]:g}s',flush=True)
            except Exception as exc:
                failures.append(dict(path=str(target),exception=repr(exc)))
                print(f'INFRASTRUCTURE ERROR {target}: {exc!r}',file=sys.stderr,flush=True)
    (output/'orchestration.json').write_text(json.dumps(dict(wall_seconds=time.perf_counter()-started,
        submitted=len(jobs),infrastructure_failures=failures),indent=2)+'\n')
    if failures:raise RuntimeError(f'{len(failures)} trials had infrastructure errors; none hidden.')


if __name__=='__main__':main()
