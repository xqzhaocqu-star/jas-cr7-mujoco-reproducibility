#!/usr/bin/env python3
"""Comparative plots of saved, unsmoothed MuJoCo samples.

No time-series sample reduction or Matplotlib line simplification is applied.
All methods, including censored or terminated outcomes, remain represented.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from comparison.controller import ComparisonSettings, METHODS
from comparison.mechanism import scalar_entry_times
from comparison.metrics import tracking_entry_metrics
from plot_results import _style, _save, COMPONENT_COLORS

LABELS = dict(proposed='Full JAS',traditional_bf='Log-BF ablation',linear_surface='Linear-surface ablation',
              linear_reaching='No dual-power reaching',no_history='No-history ablation',external_ppc='Verginis PPC')
COLORS = dict(zip(METHODS,('#1665a5','#d17b0f','#8e55ad','#c34343','#168d71','#555555')))
STYLES = dict(zip(METHODS,('-','--','-.',':','--','-')))
KEYS=('t','error','upper','lower','actual_barrier_margin','torque_command','torque_applied',
      'internal_force','desired_internal_force','theta','lambda_min','surface','xi',
      'ppc_xi_s','ppc_xi_v','ppc_rho_s','ppc_rho_v')


def load_trial(directory):
    summary=json.loads((directory/'summary.json').read_text())
    with np.load(directory/'trajectory.npz',allow_pickle=False) as raw:
        arrays={k:raw[k] for k in KEYS if k in raw}
    return summary,arrays


def _line(ax,t,y,method,**kwargs):
    return ax.plot(t,y,color=COLORS[method],linestyle=STYLES[method],
                   linewidth=.85,label=LABELS[method],**kwargs)


def plot_main(root,scenario,protocol):
    out=root/'figures'/scenario;out.mkdir(parents=True,exist_ok=True)
    cases={m:load_trial(root/'main'/scenario/m) for m in METHODS}
    note='All saved samples retained; common finite assessment bounds; terminated outcomes retain their recorded prefix.'
    fig,axes=plt.subplots(3,2,figsize=(12,9))
    for j in range(6):
        ax=axes[j%3,j//3];scale,unit=(1000.,'mm') if j<3 else (180/np.pi,'deg')
        for m,(s,a) in cases.items():
            if len(a['t']):_line(ax,a['t'],a['error'][:,j]*scale,m)
        a=cases['proposed'][1]
        if len(a['t']):
            ax.plot(a['t'],a['upper'][:,j]*scale,'k--',lw=.8,label='Common bounds')
            ax.plot(a['t'],a['lower'][:,j]*scale,'k--',lw=.8)
        ax.set(title=f'{("x","y","z","roll","pitch","yaw")[j]} error',ylabel=unit,xlabel='Time (s)')
    axes[0,0].legend(fontsize=7,ncol=2)
    _save(fig,out,'01_tracking_and_bounds',f'{scenario} | six physical errors and common assessment bounds',note)

    fig,axes=plt.subplots(2,2,figsize=(12,7.8))
    for m,(s,a) in cases.items():
        if not len(a['t']):continue
        t,e=a['t'],a['error']
        _line(axes[0,0],t,np.linalg.norm(e[:,:3],axis=1)*1000,m)
        _line(axes[0,1],t,np.linalg.norm(e[:,3:],axis=1)*180/np.pi,m)
        _line(axes[1,0],t,a['actual_barrier_margin'].min(axis=1),m)
        tail=t>=max(0.,s['requested_duration_s']-2.)
        _line(axes[1,1],t[tail],np.linalg.norm(e[tail,:3],axis=1)*1000,m)
    axes[0,0].set(ylabel='Position error norm (mm)',yscale='log')
    axes[0,1].set(ylabel='Euler-coordinate error norm (deg)',yscale='log')
    axes[1,0].axhline(0,color='k',ls='--',lw=.8);axes[1,0].set(ylabel='Minimum normalized common gap')
    axes[1,1].set(ylabel='Final 2 s position norm (mm)')
    for ax in axes.flat:ax.set_xlabel('Time (s)')
    axes[0,0].legend(fontsize=7)
    _save(fig,out,'02_tracking_norms_and_margin',f'{scenario} | physical tracking and constraint margin',
          'A positive common gap means all six physical component errors are inside the assessment bounds. Log axes omit only exact zero.')

    fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
    limits=np.tile([194,194,102,66,66,29],2)
    for ax,m in zip(axes.flat,METHODS):
        s,a=cases[m]
        if len(a['t']):
            ax.plot(a['t'],np.max(np.abs(a['torque_command'])/limits,axis=1),color=COLORS[m],lw=.65,label='Maximum |raw command|/limit')
            ax.plot(a['t'],np.max(np.abs(a['torque_applied'])/limits,axis=1),color='k',alpha=.4,lw=.5,label='Maximum |applied|/limit')
        ax.axhline(1,color='#c43d3d',ls='--',lw=.8)
        ax.set(title=f'{LABELS[m]} | {s["run_status"]}',ylabel='Motor torque / limit',xlabel='Time (s)')
    axes.flat[0].legend(fontsize=7)
    _save(fig,out,'03_control_effort',f'{scenario} | all-motor torque envelope, raw and physically applied',
          'The maximum is taken over all 12 motors at every saved tick; saturation is retained as a performance result.')

    fig,axes=plt.subplots(3,2,figsize=(12,10))
    adaptive=METHODS[:-1]
    for hand in range(2):
        for m in adaptive:
            s,a=cases[m]
            if not len(a['t']):continue
            truth=np.asarray(s['true_parameters_offline_only'])
            _line(axes[0,hand],a['t'],np.linalg.norm(a['theta'][:,hand]-truth[hand],axis=1),m)
        for row,m in ((1,'proposed'),(2,'no_history')):
            s,a=cases[m];truth=np.asarray(s['true_parameters_offline_only'])
            if not len(a['t']):
                axes[row,hand].text(.5,.5,'No valid command samples: '+s['run_status'],ha='center',transform=axes[row,hand].transAxes)
                continue
            for j in range(8):
                axes[row,hand].plot(a['t'],a['theta'][:,hand,j]-truth[hand,j],lw=.7,label=f'Parameter {j+1}')
            axes[row,hand].axhline(0,color='k',lw=.6)
            axes[row,hand].set(title=f'{LABELS[m]} | {("Left","Right")[hand]} signed errors',ylabel='Multiplier error')
        axes[0,hand].set(title=f'{("Left","Right")[hand]} parameter error norm',ylabel='L2 error',yscale='log')
        axes[0,hand].legend(fontsize=7)
        axes[1,hand].legend(fontsize=6,ncol=4)
    for ax in axes.flat:ax.set_xlabel('Time (s)')
    _save(fig,out,'04_learning_comparison',f'{scenario} | actual parameter errors; offline truth is evaluation only',
          'External PPC has no parameter estimator. Nonzero frozen residuals permit nonzero ultimate errors; no exponential-to-zero fit is imposed.')

    for kind,index,unit,stem in (('force',slice(0,3),'N','05_internal_force'),
                               ('Euler-conjugate moment',slice(3,6),'Nm','06_internal_moment')):
        fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
        for ax,m in zip(axes.flat,METHODS):
            s,a=cases[m]
            if len(a['t']):
                err=a['internal_force'][:,0,index]-a['desired_internal_force'][:,0,index]
                for j in range(3):ax.plot(a['t'],err[:,j],color=COMPONENT_COLORS[j],lw=.55,label=f'{("x","y","z")[j]} error')
            ylabel='Internal force error (N)' if kind=='force' else 'Internal moment error (Nm)'
            ax.set(title=LABELS[m],xlabel='Time (s)',ylabel=ylabel)
            ax.axhline(0,color='k',lw=.5)
        axes.flat[0].legend(fontsize=7,ncol=3)
        _save(fig,out,stem,f'{scenario} | internal {kind} error, common task target',
              'Unsmoothed reaction diagnostics; both DC bias and high-frequency switching remain visible. Right internal errors have opposite sign.')

    fig,axes=plt.subplots(3,2,figsize=(12,9),sharex=True)
    s,a=cases['external_ppc']
    for j,ax in enumerate(axes.flat):
        if len(a['t']) and 'ppc_xi_s' in a:
            for h,style in ((0,'-'),(1,'--')):
                ax.plot(a['t'],a['ppc_xi_s'][:,h,j],color='#2166ac',ls=style,lw=.7,label=f'{("Left","Right")[h]} pose / native pose bound')
                ax.plot(a['t'],a['ppc_xi_v'][:,h,j],color='#d17b0f',ls=style,lw=.7,label=f'{("Left","Right")[h]} velocity error / native velocity bound')
        ax.axhline(1,color='k',ls=':',lw=.8);ax.axhline(-1,color='k',ls=':',lw=.8)
        ax.set(title=('x','y','z','roll','pitch','yaw')[j],ylabel='Native normalized error',xlabel='Time (s)')
    axes.flat[0].legend(fontsize=6.5)
    _save(fig,out,'07_external_native_funnels',f'{scenario} | external PPC native pose and velocity funnels | {s["run_status"]}',
          'These native symmetric monotonic funnels differ from the common asymmetric assessment bounds. JAS S/xi-layer criteria do not apply to this controller.')


def plot_sweep(root,protocol):
    out=root/'figures';out.mkdir(exist_ok=True)
    methods=('proposed','linear_surface','linear_reaching')
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    surface_counts=[]
    for method_index,m in enumerate(methods):
        rows=[]
        for scenario in protocol['scenarios']:
            if scenario.get('study')!='initial_sweep':continue
            s,a=load_trial(root/'initial_sweep'/scenario['id']/m)
            rows.append((np.linalg.norm(np.asarray(scenario['offset'])[:3])*1000,
                         s['tracking_entry']['simultaneous_entry_s'],
                         s.get('surface_entry_s'),s.get('xi_layer_entry_s'),scenario['initial_scale']))
        surface_counts.append((m,sum(r[2] is not None for r in rows),len(rows)))
        for sign,marker in ((-1,'v'),(1,'o')):
            subset=sorted([r for r in rows if r[-1]*sign>0])
            x=[r[0] for r in subset]
            for ax,k in ((axes[0,0],1),(axes[0,1],2),(axes[1,0],3)):
                # Censoring is not a measured entry at the observation endpoint.
                y=[r[k] if r[k] is not None else np.nan for r in subset]
                ax.plot(x,y,color=COLORS[m],marker=marker,ls='-' if sign>0 else '--',label=LABELS[m]+(' +scale' if sign>0 else ' -scale'))
                for xx,rr in zip(x,subset):
                    if rr[k] is None:
                        display_shift=(method_index-1)*.8+sign*.1
                        ax.scatter(xx+display_shift,8.,color=COLORS[m],marker='x',s=38,zorder=4)
    axes[0,0].set(ylabel='Joint physical-tolerance entry (s)',title='0.25 mm and 0.05 deg; both norms')
    axes[0,1].set(ylabel='Surface-threshold entry (s)',title='Diagnostic only: max |S| <= 0.001')
    axes[0,1].text(.04,.64,'S criterion: sustained entries',transform=axes[0,1].transAxes,fontsize=8)
    for j,(m,count,total) in enumerate(surface_counts):
        axes[0,1].text(.04,.55-j*.09,f'{LABELS[m]}: {count}/{total}',color=COLORS[m],
                       transform=axes[0,1].transAxes,fontsize=8)
    axes[1,0].set(ylabel='Transformed-layer entry (s)',title='Diagnostic only: max |xi| <= delta')
    for ax in (axes[0,0],axes[0,1],axes[1,0]):
        ax.set_xlabel('Initial position error norm (mm)');ax.set_ylim(-.1,8.4)
    axes[0,0].legend(fontsize=6.5,ncol=2)
    ax=axes[1,1]
    for color,multiplier in zip(COMPONENT_COLORS,(.5,1.,2.)):
        vals=[]
        for scenario in protocol['scenarios']:
            if scenario.get('study')!='initial_sweep':continue
            s,a=load_trial(root/'initial_sweep'/scenario['id']/'proposed')
            metric=tracking_entry_metrics(a['t'],a['error'],position_threshold_m=.00025*multiplier,
                orientation_threshold_rad=np.deg2rad(.05)*multiplier,requested_end_s=8.,completed=s['run_status']=='completed')
            vals.append((np.linalg.norm(np.asarray(scenario['offset'])[:3])*1000,
                         metric['simultaneous_entry_s'],scenario['initial_scale']))
        for sign,marker in ((-1,'v'),(1,'o')):
            subset=sorted([r for r in vals if r[2]*sign>0])
            valid=[r for r in subset if r[1] is not None]
            ax.scatter([r[0] for r in valid],[r[1] for r in valid],color=color,marker=marker,
                       label=f'Tolerances x{multiplier:g}, {"+" if sign>0 else "-"}scale')
            for xx,yy,_ in subset:
                if yy is None:ax.scatter(xx+sign*.1,8.,color=color,marker='x',s=38)
    ax.set(xlabel='Initial position error norm (mm)',ylabel='Full JAS physical entry (s)',title='Tolerance sensitivity; sign shown by marker',ylim=(-.1,8.4));ax.legend(fontsize=7,ncol=2)
    _save(fig,out,'07_initial_condition_sweep','Initial-condition study | fixed gains, static reference, eight feasible physical initial states',
          'Entries persist through 8 s for at least 2 s. Cross at 8 s = right censored; no connecting segment.\n'
          'Only censored symbols have small horizontal display offsets; raw initial states are unchanged. A finite sweep does not prove a global fixed-time bound.')


def plot_mechanism(root,protocol):
    out=root/'figures'
    data=scalar_entry_times(ComparisonSettings(**protocol['settings']))
    (root/'scalar_mechanism.json').write_text(json.dumps(data,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for ax,stage in zip(axes,('sliding','reaching')):
        r=np.asarray(data['initial_magnitude'])
        ax.semilogx(r,data[stage+'_dual_power_entry_s'],color=COLORS['proposed'],label='Dual-power scalar ODE')
        ax.semilogx(r,data[stage+'_linear_entry_s'],color=COLORS['linear_surface'],label='Linear scalar ODE')
        ax.axhline(data[stage+'_uniform_comparison_bound_s'],color='k',ls='--',label='Scalar uniform comparison bound')
        ax.set(xlabel='Scalar initial magnitude (not robot initial pose)',ylabel='Entry time to positive layer (s)',title=stage.title()+' mechanism')
        ax.legend(fontsize=7)
    _save(fig,out,'08_scalar_fixed_time_mechanisms','Independent scalar ODE illustration | uniform entry-time bound versus initial-state dependence',
          'Abstract scalar magnitudes extend beyond the robot workspace. These are not robot trajectories or its inertia-dependent theorem bounds.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True)
    p.add_argument('--case',default='all',choices=('all','nominal','nonmonotonic','large_initial_error','summary'))
    args=p.parse_args();root=args.results.resolve();protocol=json.loads((root/'protocol.json').read_text())
    _style()
    if args.case!='summary':
        for name in ('nominal','nonmonotonic','large_initial_error') if args.case=='all' else (args.case,):plot_main(root,name,protocol)
    if args.case in ('all','summary'):plot_sweep(root,protocol);plot_mechanism(root,protocol)


if __name__=='__main__':main()
