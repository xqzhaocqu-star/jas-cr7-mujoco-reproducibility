#!/usr/bin/env python3
"""Evaluate declared gain choices without changing the JAS feedback law."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from jas.controller import Settings
from run_experiments import HERE,run_case


CANDIDATES={
    'lower_feedback':dict(ka=(2.,2.,2.,.006,.006,.006),
                          k2=(1.,1.,1.,.003,.003,.003),
                          k3=(.2,.2,.2,.001,.001,.001)),
    'very_low_feedback':dict(ka=(.6,.6,.6,.002,.002,.002),
                             k2=(.3,.3,.3,.001,.001,.001),
                             k3=(.05,.05,.05,.00025,.00025,.00025)),
    'lower_fractional':dict(k3=(.1,.1,.1,.0005,.0005,.0005)),
    'gentler_surface':dict(lam1=.2,k3=(.1,.1,.1,.0005,.0005,.0005)),
    'boundary_005':dict(robust_mode='boundary_layer',robust_width=.05),
    'boundary_010':dict(robust_mode='boundary_layer',robust_width=.1),
    'boundary_030':dict(robust_mode='boundary_layer',robust_width=.3),
}


def ripple_metrics(path,start,end):
    with np.load(path) as z:
        mask=(z['t']>=start-1e-10)&(z['t']<=end+1e-10)
        force=z['internal_force'][mask,0]
        error=force-z['desired_internal_force'][mask,0]
        ac=force-np.mean(force,axis=0)
        return dict(interval_s=[start,end],samples=int(np.sum(mask)),
                    mean_left_internal_wrench=np.mean(force,axis=0).tolist(),
                    component_error_rms=np.sqrt(np.mean(error**2,axis=0)).tolist(),
                    component_ac_rms=np.sqrt(np.mean(ac**2,axis=0)).tolist(),
                    component_peak_to_peak=np.ptp(force,axis=0).tolist(),
                    force_ac_vector_rms_N=float(np.sqrt(np.mean(np.sum(ac[:,:3]**2,axis=1)))),
                    moment_ac_vector_rms_Nm=float(np.sqrt(np.mean(np.sum(ac[:,3:]**2,axis=1)))))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',choices=CANDIDATES,required=True)
    parser.add_argument('--duration',type=float,default=4.)
    parser.add_argument('--output',type=Path,default=HERE/'scratch/chattering_recheck')
    args=parser.parse_args()
    output=args.output/args.candidate/'nominal'
    cfg=replace(Settings(),**CANDIDATES[args.candidate])
    summary=run_case('nominal',duration=args.duration,output=output,settings=cfg)
    interval=(max(2.,args.duration-2.),args.duration)
    metrics=ripple_metrics(output/'trajectory.npz',*interval)
    baseline=HERE/'results/nominal/trajectory.npz'
    if baseline.exists():
        metrics['baseline_same_interval']=ripple_metrics(baseline,*interval)
    metrics.update(candidate=args.candidate,settings=summary['settings'],
                   position_rmse_mm=summary['position_rmse_after_2s_mm'],
                   orientation_rmse_deg=summary['orientation_rmse_after_2s_deg'],
                   minimum_barrier_gap=summary['minimum_normalized_barrier_gap'],
                   max_torque_ratio=summary['max_command_torque_limit_ratio'])
    (output.parent/'ripple_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print(json.dumps(metrics),flush=True)


if __name__=='__main__':
    main()
