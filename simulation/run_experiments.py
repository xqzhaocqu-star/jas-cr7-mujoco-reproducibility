#!/usr/bin/env python3
"""Run the revised JAS controller on full MuJoCo dual-CR7 dynamics."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys
import time
import numpy as np
import scipy
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import mujoco
from jas.geometry import LocalKinematics
from jas.dynamics import LinkDynamics
from jas.controller import JASController,Settings
from jas.reference import desired

HERE=Path(__file__).resolve().parent
CASES={
    'nominal': dict(nonmonotonic=False,offset=[-.015,.010,-.012,-.03,.02,-.025]),
    'nonmonotonic': dict(nonmonotonic=True,offset=[-.015,.010,-.012,-.03,.02,-.025]),
    'large_initial_error': dict(nonmonotonic=False,offset=[-.045,.030,-.035,-.07,.05,-.06]),
}


def kinematics(metadata,name):
    frame=metadata['grasp_frames'][name]
    r=Rotation.from_euler('z',frame['base_yaw_rad']).as_matrix()
    hand=np.array(frame['palm_rotation_object'])
    offset=-hand.T@np.array(frame['palm_position_object_m'])
    return LocalKinematics(HERE/'models/arm_nominal.xml',frame['base_position_world_m'],r,
                           offset,hand.T)


def initialize(model,data,meta,kins,center,offset):
    target=center+np.array(offset)
    data.qpos[:]=meta['q_initial']
    for i,kin in enumerate(kins):
        q0=data.qpos[6*i:6*i+6].copy()
        result=least_squares(lambda q:kin.pose_jacobian(q)[0]-target,q0,
                             jac=lambda q:kin.pose_jacobian(q)[1],
                             bounds=(-3.0543+1e-5,3.0543-1e-5),
                             xtol=1e-13,ftol=1e-13,gtol=1e-13,max_nfev=200)
        if np.max(np.abs(result.fun))>1e-8:
            raise RuntimeError('Initial inverse kinematics did not close the rigid chain.')
        data.qpos[6*i:6*i+6]=result.x
    rotation=Rotation.from_euler('xyz',target[3:]).as_quat()
    data.qpos[12:15]=target[:3]
    data.qpos[15:19]=rotation[[3,0,1,2]]
    data.qvel[:]=0.
    mujoco.mj_forward(model,data)
    return target


def run_case(name,duration=14.,control_dt=.0005,physics_dt=.00025,
             output=None,settings=None,progress=True):
    if name not in CASES:
        raise ValueError(name)
    output=Path(output or HERE/'results'/name)
    output.mkdir(parents=True,exist_ok=True)
    meta=json.loads((HERE/'models/scene_metadata.json').read_text())
    model=mujoco.MjModel.from_xml_path(str(HERE/'models/dual_cr7.xml'))
    model.opt.timestep=physics_dt
    ratio=int(round(control_dt/physics_dt))
    if ratio<1 or abs(ratio*physics_dt-control_dt)>1e-10:
        raise ValueError('control_dt must be an integer multiple of physics_dt.')
    data=mujoco.MjData(model)
    kins=[kinematics(meta,n) for n in ('left','right')]
    center=np.r_[meta['payload_nominal']['initial_position_world_m'],np.zeros(3)]
    start=initialize(model,data,meta,kins,center,CASES[name]['offset'])
    payload=meta['payload_nominal']
    controllers=[JASController(kins[i],LinkDynamics(HERE/'models/arm_nominal.xml'),center,
                               payload['mass_kg'],payload['diagonal_inertia_kg_m2'],i,
                               settings,CASES[name]['nonmonotonic']) for i in range(2)]
    limits=np.tile(meta['torque_limits_Nm'],2)
    velocity_limits=np.tile(meta['joint_velocity_limits_rad_s'],2)
    nsteps=int(round(duration/control_dt))
    rows=[]
    min_margin=1.; min_sigma=1.; min_joint_margin=10.; max_speed_ratio=0.
    max_torque_ratio=0.; max_weld=0.; max_contacts=0; saturated=0
    started=time.perf_counter()
    next_update=0.
    payload_body=model.body('payload').id
    for step in range(nsteps+1):
        t=step*control_dt
        commanded=[]; infos=[]
        for i,ctl in enumerate(controllers):
            torque,info=ctl.command(t,data.qpos[6*i:6*i+6],data.qvel[6*i:6*i+6])
            commanded.extend(torque); infos.append(info)
        commanded=np.asarray(commanded)
        theta_control=np.array([ctl.learner.theta.copy() for ctl in controllers])
        applied=np.clip(commanded,-limits,limits)
        saturated+=int(np.any(applied!=commanded))
        data.ctrl[:]=applied
        for i,ctl in enumerate(controllers):
            ctl.update(t,applied[6*i:6*i+6],infos[i])
        # Forward is for current-state diagnostics; it never advances or sets pose.
        mujoco.mj_forward(model,data)
        object_r=data.xmat[payload_body].reshape(3,3)
        actual=np.r_[data.xpos[payload_body],Rotation.from_matrix(object_r).as_euler('xyz')]
        ref=desired(t,center)[0]
        forces=[]
        for i,info in enumerate(infos):
            reaction=data.qfrc_constraint[6*i:6*i+6]
            forces.append(np.linalg.solve(info['jacobian'].T,reaction))
        forces=np.asarray(forces)
        internal=forces-.5*np.sum(forces,axis=0)
        sigmas=np.array([z['sigma_min'] for z in infos])
        margins=np.array([z['barrier_margin'] for z in infos])
        min_margin=min(min_margin,float(margins.min()))
        min_sigma=min(min_sigma,float(sigmas.min()))
        max_torque_ratio=max(max_torque_ratio,float(np.max(np.abs(commanded)/limits)))
        rows.append(dict(t=t,qpos=data.qpos.copy(),qvel=data.qvel.copy(),
                         object_pose=actual,reference=ref,error=actual-ref,
                         local_error=np.array([z['error'] for z in infos]),
                         torque_command=commanded,torque_applied=applied,
                         surface=np.array([z['surface'] for z in infos]),
                         xi=np.array([z['xi'] for z in infos]),barrier_margin=margins,
                         upper=infos[0]['upper'],lower=infos[0]['lower'],
                         sigma_min=sigmas,theta=np.array([z.learner.theta for z in controllers]),
                         theta_control=theta_control,
                         filtered_regressor=np.array([z.learner.filtered_regressor for z in controllers]),
                         filtered_torque=np.array([z.learner.tauf for z in controllers]),
                         lambda_min=np.array([z.learner.minimum_eigenvalue for z in controllers]),
                         prediction_error=np.array([z.learner.epsilon for z in controllers]),
                         internal_force=internal,interaction_force=forces,
                         desired_internal_force=np.array([z.fd for z in controllers]),
                         weld_residual=data.efc_pos[:12].copy(),
                         contact_count=data.ncon))
        if progress and t>=next_update:
            print(f'{name}: t={t:.2f}/{duration:.2f} s, position error={np.linalg.norm(actual[:3]-ref[:3])*1000:.3f} mm, '
                  f'margin={min_margin:.4g}, torque ratio={max_torque_ratio:.3f}, wall={time.perf_counter()-started:.1f} s',flush=True)
            next_update=t+2.
        if step==nsteps:
            break
        for substep in range(ratio):
            mujoco.mj_step(model,data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError('Nonfinite full dynamics; no resets performed.')
            if abs(data.time-(t+(substep+1)*physics_dt))>1e-7:
                raise RuntimeError('MuJoCo time reset detected.')
            min_joint_margin=min(min_joint_margin,float(np.min(3.0543-np.abs(data.qpos[:12]))))
            max_speed_ratio=max(max_speed_ratio,float(np.max(np.abs(data.qvel[:12])/velocity_limits)))
            max_weld=max(max_weld,float(np.max(np.abs(data.efc_pos[:12]))))
            max_contacts=max(max_contacts,int(data.ncon))
    arrays={k:np.asarray([row[k] for row in rows]) for k in rows[0]}
    np.savez_compressed(output/'trajectory.npz',**arrays)
    final_mask=arrays['t']>=max(0.,duration-2.)
    tracking_mask=arrays['t']>=min(2.,duration/2)
    truth=json.loads((HERE/'models/plant_truth.json').read_text())
    true_theta=np.array([truth[n+'_density_multipliers']+[truth[n+'_viscous_multiplier'],
                         truth['payload_density_multiplier']] for n in ('left','right')])
    ieee_warnings={mujoco.mjtWarning(i).name:int(w.number) for i,w in enumerate(data.warning) if w.number}
    gram=np.array([ctl.learner.gram for ctl in controllers])
    prediction_data=np.array([ctl.learner.p for ctl in controllers])
    stored_residual=prediction_data-np.einsum('aij,aj->ai',gram,true_theta)
    np.savez_compressed(output/'learning_state.npz',gram=gram,p=prediction_data,
                        true_parameters_offline_only=true_theta,stored_residual=stored_residual)
    summary=dict(case=name,duration_s=duration,control_dt_s=control_dt,physics_dt_s=physics_dt,
                 wall_seconds=time.perf_counter()-started,initial_object_pose=start.tolist(),
                 settings=controllers[0].configuration(),scenario=CASES[name],
                 position_rmse_after_2s_mm=float(np.sqrt(np.mean(np.sum(arrays['error'][tracking_mask,:3]**2,axis=1)))*1000),
                 orientation_rmse_after_2s_deg=float(np.sqrt(np.mean(np.sum(arrays['error'][tracking_mask,3:]**2,axis=1)))*180/np.pi),
                 final_position_rmse_mm=float(np.sqrt(np.mean(np.sum(arrays['error'][final_mask,:3]**2,axis=1)))*1000),
                 final_orientation_rmse_deg=float(np.sqrt(np.mean(np.sum(arrays['error'][final_mask,3:]**2,axis=1)))*180/np.pi),
                 minimum_normalized_barrier_gap=min_margin,minimum_jacobian_singular_value=min_sigma,
                 max_command_torque_limit_ratio=max_torque_ratio,saturated_control_samples=saturated,
                 minimum_joint_limit_margin_rad=min_joint_margin,max_joint_speed_limit_ratio=max_speed_ratio,
                 max_weld_residual=max_weld,max_contact_count=max_contacts,mujoco_warnings=ieee_warnings,
                 ie_detection_times_s=[ctl.learner.frozen_time for ctl in controllers],
                 final_excitation_min_eigenvalues=[ctl.learner.minimum_eigenvalue for ctl in controllers],
                 final_parameter_estimates=[ctl.learner.theta.tolist() for ctl in controllers],
                 true_parameters_offline_only=true_theta.tolist(),
                 final_parameter_error_norms=np.linalg.norm(arrays['theta'][-1]-true_theta,axis=1).tolist(),
                 stored_learning_residual_norms=np.linalg.norm(stored_residual,axis=1).tolist(),
                 excitation_condition_numbers=[float(np.linalg.cond(g)) for g in gram],
                 numerical_projection_counts=[ctl.learner.numerical_projection_count for ctl in controllers],
                 internal_force_error_final_rms=np.sqrt(np.mean((arrays['internal_force'][final_mask]-arrays['desired_internal_force'][final_mask])**2,axis=0)).tolist(),
                 software=dict(python=platform.python_version(),mujoco=mujoco.__version__,numpy=np.__version__,scipy=scipy.__version__),
                 algorithm_variant=('Original paper sign feedback with positive rho bound'
                                    if controllers[0].cfg.robust_mode=='paper_sign'
                                    else 'JAS boundary-layer engineering variant; original exact reaching proof does not apply'),
                 physics='full generalized-coordinate mj_step, torque-only actuation, two numerical welds',
                 claim_limit='Sampled ideal sign control; no claim of exact Filippov reaching or exact physical-robot inertia.')
    sources=[HERE.parent/'bare_jrnl_new_sample4.tex',HERE/'run_experiments.py']
    sources+=sorted((HERE/'jas').glob('*.py'))
    sources+=list((HERE/'models').glob('*.xml'))
    sources+=[HERE/'models/scene_metadata.json',HERE/'models/plant_truth.json']
    manifest={str(p.relative_to(HERE.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (output/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=list(CASES)+['all'],default='all')
    parser.add_argument('--duration',type=float,default=14.)
    parser.add_argument('--control-dt',type=float,default=.0005)
    parser.add_argument('--physics-dt',type=float,default=.00025)
    parser.add_argument('--robust-mode',choices=('paper_sign','boundary_layer'),default='paper_sign')
    parser.add_argument('--robust-width',type=float,default=.05,
                        help='Only boundary_layer uses this width in Y @ S coordinates.')
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args()
    settings=Settings(robust_mode=args.robust_mode,robust_width=args.robust_width)
    output=args.output or HERE/('results' if args.robust_mode=='paper_sign' else 'results_low_chatter')
    for case in (CASES if args.case=='all' else [args.case]):
        summary=run_case(case,args.duration,args.control_dt,args.physics_dt,output/case,settings)
        print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
