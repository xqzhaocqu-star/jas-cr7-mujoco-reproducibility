#!/usr/bin/env python3
"""Audit comparison formulas and saved physics separately from performance.

Only audit JSON/NPZ artifacts are written. Saturation, common-envelope
violations, failure to enter a tolerance, and explicitly retained termination
are outcomes, not automatic formula-validation failures. The command prefix,
failure state, offline truth diagnostics, and every registered trial remain
visible. This script never changes a controller, a state, or a saved rollout.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
import traceback

import mujoco
import numpy as np

from comparison.metrics import force_statistics, torque_statistics
from validate_results import HERE, PROJECT, plain, reconstruct_kinematics, replay_intervals, skew
from validate_original_retrial import (audit_original_filter_and_history,
    audit_offline_residual, project_direction)


METHODS = ('proposed', 'traditional_bf', 'linear_surface', 'linear_reaching',
           'no_history', 'external_ppc')
RAW_FIELDS = ('t', 'qpos', 'qvel', 'object_pose', 'reference', 'error', 'local_error',
    'torque_command', 'torque_applied', 'surface', 'xi', 'upper', 'lower',
    'barrier_margin', 'actual_barrier_margin', 'native_upper', 'native_lower',
    'native_barrier_margin', 'sigma_min', 'theta', 'theta_control',
    'filtered_regressor', 'filtered_torque', 'lambda_min', 'prediction_error',
    'regressor', 'momentum', 'remainder', 'A', 'learning_drive',
    'known_torque_left', 'known_torque_right', 'internal_force',
    'interaction_force', 'desired_internal_force', 'weld_residual', 'contact_count',
    'control_valid_mask', 'barrier_applicable')
EXTERNAL_FIELDS = ('ppc_rho_s','ppc_rho_v','ppc_xi_s','ppc_xi_v',
    'ppc_epsilon_s','ppc_epsilon_v','ppc_v_reference','ppc_object_velocity',
    'ppc_object_feedback','ppc_load_allocation','ppc_grasp_map','ppc_palm_wrench',
    'ppc_internal_palm_wrench','ppc_allocation_identity_error','ppc_internal_null_error')
EXTERNAL_DEFAULTS = dict(gs=.01, gv=10.,
    rho_s_initial=[.08,.08,.08,.22,.20,.22],
    rho_s_final=[.012,.010,.010,.045,.040,.035], rho_s_rate=[.5]*6,
    rho_v_initial=None, rho_v_final=[.4,.4,.4,.8,.8,.8],rho_v_rate=[.5]*6,
    velocity_initial_slack=1., desired_pitch_max=.35, virtual_mass=1.,virtual_inertia=1.,
    grasp_positions_object=[[0.,-.20,0.],[0.,.20,0.]])


class ComparisonAudit:
    def __init__(self, directory):
        self.report = dict(directory=str(directory.resolve()), validity_passed=False,
            checks={}, failures=[], performance={},
            scope='All saved ticks for FK, evaluation boundaries, entry and learning; '
                  'sampled independently reconstructed control formulas and full-MuJoCo '
                  'RK4 interval replay. Performance failure is retained separately. '
                  'No continuous-time, global fixed-time, or exact-truth convergence certificate.')

    def check(self, name, passed, **details):
        self.report['checks'][name] = dict(passed=bool(passed), **plain(details))
        if not passed:
            self.report['failures'].append(name)

    def finish(self, directory):
        self.report['validity_passed'] = not self.report['failures']
        self.report['passed'] = self.report['validity_passed']
        (directory/'comparison_validation.json').write_text(
            json.dumps(plain(self.report), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        return self.report


def discrepancy(actual, expected):
    """Finite max error, treating matching signed infinities as equal bounds."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    if actual.shape != expected.shape:
        return float('inf')
    same_special = (actual == expected) | (np.isnan(actual) & np.isnan(expected))
    finite = np.isfinite(actual) & np.isfinite(expected)
    if np.any(~(finite | same_special)):
        return float('inf')
    return float(np.max(np.abs(actual[finite]-expected[finite]))) if np.any(finite) else 0.


def close(actual, expected, absolute=2e-8, relative=2e-12):
    expected = np.asarray(expected)
    finite = expected[np.isfinite(expected)]
    tolerance = absolute+relative*(float(np.max(np.abs(finite))) if finite.size else 0.)
    error = discrepancy(actual, expected)
    return error <= tolerance, error, tolerance


def independent_envelope(times, settings, nonmonotonic=False):
    """Paper envelope and derivatives, without calling jas.reference."""
    t = np.asarray(times)[..., None]
    chi, ell = np.asarray(settings['chi']), np.asarray(settings['ell'])
    final = (np.array([.012, .014, .010, .045, .040, .050]),
             np.array([.018, .010, .016, .060, .055, .035]))
    envelopes = []
    for factor, terminal in zip((1., .8), final):
        limit = ell*np.tanh(chi*terminal)
        tail = (ell-limit)*np.exp(-.65*t)
        mu, dmu, ddmu = limit+tail, -.65*tail, .65**2*tail
        if nonmonotonic:
            z = np.clip((t-3.)/4., 0., 1.)
            inside = (t > 3.) & (t < 7.)
            bump = 64*z**3*(1-z)**3
            db = np.where(inside, 48*z*z*(1-z)**2*(1-2*z), 0.)
            ddb = np.where(inside, 24*z*(1-z)*(1-5*z+5*z*z), 0.)
            mu = mu+factor*.23*ell*bump
            dmu = dmu+factor*.23*ell*db
            ddmu = ddmu+factor*.23*ell*ddb
        envelopes.append((mu, dmu, ddmu))
    (u, du, ddu), (l, dl, ddl) = envelopes
    return u, l, du, dl, ddu, ddl


def independent_common_bounds(times, settings, nonmonotonic=False):
    """Finite common C2 assessment boundary, independent of comparison code."""
    t = np.asarray(times)[..., None]
    transition = float(settings['evaluation_transition_s'])
    z = np.clip(t/transition, 0., 1.)
    gate = z**3*(10-15*z+6*z*z)
    dg = np.where((t >= 0) & (t < transition), 30*z*z*(1-z)**2/transition, 0.)
    ddg = np.where((t >= 0) & (t < transition), 60*z*(1-z)*(1-2*z)/transition**2, 0.)
    chi, ell = np.asarray(settings['chi']), np.asarray(settings['ell'])
    u, l, du, dl, ddu, ddl = independent_envelope(times, settings, nonmonotonic)
    result = []
    for mu, dmu, ddmu, key in ((u, du, ddu, 'evaluation_initial_upper'),
                              (l, dl, ddl, 'evaluation_initial_lower')):
        initial_fraction = np.tanh(chi*np.asarray(settings[key]))
        kappa = initial_fraction+(1-initial_fraction)*gate
        dk, ddk = (1-initial_fraction)*dg, (1-initial_fraction)*ddg
        arg = kappa*mu/ell
        darg, ddarg = (dk*mu+kappa*dmu)/ell, (ddk*mu+2*dk*dmu+kappa*ddmu)/ell
        gap = 1-arg**2
        result.append((np.arctanh(arg)/chi, darg/(chi*gap),
                       (ddarg/gap+2*arg*darg*darg/gap**2)/chi))
    (upper, du, ddu), (lower, dl, ddl) = result
    return upper, lower, du, dl, ddu, ddl


def independent_log_transform(error, velocity, bounds, scale):
    u, l, du, dl, ddu, ddl = bounds
    e, ed, c = np.asarray(error), np.asarray(velocity), np.asarray(scale)
    plus, minus = l+e, u-e
    if np.any(plus <= 0) or np.any(minus <= 0):
        raise FloatingPointError('Independent logarithmic domain check failed.')
    xi = c*np.log((plus/l)/(minus/u))
    a = c*(1/plus+1/minus)
    b = c*(dl/plus-du/minus-dl/l+du/u)
    adot = c*(-(ed+dl)/plus**2+(ed-du)/minus**2)
    bdot = c*(ddl/plus-dl*(ed+dl)/plus**2-ddl/l+(dl/l)**2
              -ddu/minus+du*(du-ed)/minus**2+ddu/u-(du/u)**2)
    return SimpleNamespace(xi=xi, xidot=a*ed+b, a=a, b=b, adot=adot, bdot=bdot,
        upper=u, lower=-l, margin=np.minimum(plus/l, minus/u))


def independent_original_transform(error, velocity, bounds, settings):
    u, v, du, dv, ddu, ddv = bounds
    chi, ell = np.asarray(settings['chi']), np.asarray(settings['ell'])
    z = np.tanh(chi*error)
    eta = ell*z
    eta_e = chi*ell*(1-z*z)
    eta_dot = eta_e*velocity
    ga, gb = u-eta, v+eta
    if np.any(ga <= 0) or np.any(gb <= 0):
        raise FloatingPointError('Independent original JAS domain check failed.')
    den, dden = ga*gb, (du-eta_dot)*gb+ga*(dv+eta_dot)
    numerator = u*v+eta*eta
    a = eta_e*numerator/den**2
    adot = ((-2*chi*z*eta_e*velocity)*numerator
            +eta_e*(du*v+u*dv+2*eta*eta_dot))/den**2-2*a*dden/den
    n = du*v+u*dv+(du-dv)*eta
    dn = ddu*v+2*du*dv+u*ddv+(ddu-ddv)*eta+(du-dv)*eta_dot
    b = -eta*n/den**2
    bdot = -(eta_dot*n+eta*dn)/den**2-2*b*dden/den
    with np.errstate(divide='ignore'):
        upper, lower = np.arctanh(u/ell)/chi, -np.arctanh(v/ell)/chi
    return SimpleNamespace(xi=eta/den, xidot=a*velocity+b, a=a, b=b,
        adot=adot, bdot=bdot, upper=upper, lower=lower, margin=np.minimum(ga, gb)/ell)


def independent_surface(xi, xidot, settings):
    alpha, beta, delta = (float(settings[key]) for key in ('alpha', 'beta', 'delta'))
    inside = np.abs(xi) < delta
    h, dh = np.empty_like(xi), np.empty_like(xi)
    p1, p2 = (3-alpha)*delta**(alpha-1)/2, (alpha-1)*delta**(alpha-3)/2
    h[inside] = p1*xi[inside]+p2*xi[inside]**3
    dh[inside] = p1+3*p2*xi[inside]**2
    h[~inside] = np.sign(xi[~inside])*np.abs(xi[~inside])**alpha
    dh[~inside] = alpha*np.abs(xi[~inside])**(alpha-1)
    velocity = -settings['lam1']*h-settings['lam2']*np.sign(xi)*np.abs(xi)**beta
    derivative = -(settings['lam1']*dh+settings['lam2']*beta*np.abs(xi)**(beta-1))*xidot
    return xidot-velocity, velocity, derivative


def independent_reference(times, center, scenario):
    """Reference position, velocity, acceleration from specified moderate waves."""
    times = np.asarray(times)
    if scenario.get('reference_profile') == 'hold':
        zeros = np.zeros(times.shape+(6,))
        return np.broadcast_to(center, zeros.shape).copy(), zeros, zeros.copy()
    if scenario.get('reference_profile', 'moderate') != 'moderate':
        raise ValueError('Unrecognized comparison reference profile.')
    duration = float(scenario.get('excitation_duration', 20.))
    z = np.clip(times/2., 0., 1.)
    gate = z**3*(10-15*z+6*z*z)
    dg = np.where(times < 2., 15*z*z*(1-z)**2, 0.)
    ddg = np.where(times < 2., 15*z*(1-z)*(1-2*z), 0.)
    zs = np.clip((times-duration+2.)/2., 0., 1.)
    phase = np.where(times <= duration-2., times,
                     duration-2.+2*(zs-2.5*zs**4+3*zs**5-zs**6))
    speed = np.where(times <= duration-2., 1., 1-10*zs**3+15*zs**4-6*zs**5)
    acceleration = np.where((times > duration-2.) & (times < duration),
                             (-30*zs**2+60*zs**3-30*zs**4)/2., 0.)
    amplitudes = np.array([.055, .040, .035, .20, .16, .22])[:, None]*[.65, .25, .10]
    frequencies = np.array([1.7, 2.1, 2.7, 3., 3.7, 4.1])[:, None]*[1., 1.83, 2.71]
    shifts = np.array([[0., 1.1, 2.4], [.7, 2.2, .3], [1.2, .2, 1.8],
                       [.4, 1.7, 2.8], [1., 2.6, .6], [1.8, .5, 2.]])
    argument = phase[..., None, None]*frequencies+shifts
    wave = np.sum(amplitudes*np.sin(argument), axis=-1)
    dphase = np.sum(amplitudes*frequencies*np.cos(argument), axis=-1)
    ddphase = np.sum(-amplitudes*frequencies**2*np.sin(argument), axis=-1)
    ref = center+gate[..., None]*wave
    velocity = dg[..., None]*wave+gate[..., None]*dphase*speed[..., None]
    acc = (ddg[..., None]*wave+2*dg[..., None]*dphase*speed[..., None]
           +gate[..., None]*(ddphase*speed[..., None]**2+dphase*acceleration[..., None]))
    return ref, velocity, acc


def independent_entry(times, condition, hold_s, requested_end, completed):
    """Independent final-suffix scan; truncated outcomes are right censored."""
    if not completed or not len(times) or times[-1] < requested_end-1e-9:
        return None
    condition = np.asarray(condition, dtype=bool)
    if condition.ndim > 1:
        condition = condition.reshape(len(times), -1).all(axis=1)
    start = len(times)
    for tick in range(len(times)-1, -1, -1):
        if not condition[tick]:
            break
        start = tick
    if start == len(times) or times[-1]-times[start] < hold_s-1e-12:
        return None
    return float(times[start])


def external_settings(summary):
    """Resolved fixed design constants, independent of the controller module."""
    settings = dict(EXTERNAL_DEFAULTS)
    settings.update(summary.get('external_settings') or {})
    return settings


def as_six(value):
    value = np.asarray(value,dtype=float)
    return np.full(6,float(value)) if value.ndim == 0 else value


def external_pose_step(pose, jacobian, velocity, reference, settings, time):
    """Published (25)--(32); representation Jacobian inverted explicitly."""
    _, pitch,yaw = pose[3:]
    mapping = np.eye(6)
    mapping[3:,3:] = [[np.cos(yaw)*np.cos(pitch),-np.sin(yaw),0.],
        [np.sin(yaw)*np.cos(pitch),np.cos(yaw),0.],[-np.sin(pitch),0.,1.]]
    rho_s = (as_six(settings['rho_s_initial'])-as_six(settings['rho_s_final']))*np.exp(
        -as_six(settings['rho_s_rate'])*time)+as_six(settings['rho_s_final'])
    xi_s = (pose-reference)/rho_s
    if np.any(np.abs(xi_s) >= 1.):
        raise FloatingPointError('Independent external pose domain failed.')
    epsilon_s = 2*np.arctanh(xi_s)
    geometric_velocity = mapping@(jacobian@velocity)
    v_reference = -settings['gs']*mapping@(2*epsilon_s/(rho_s*(1-xi_s**2)))
    return mapping,rho_s,xi_s,epsilon_s,geometric_velocity,v_reference,geometric_velocity-v_reference


def independent_external_command(pose, jacobian, velocity, reference, settings,
                                 time, initial_velocity_envelope, arm, force_target):
    """Published PPC plus known Remark-5 task; no dynamics or JAS law."""
    from scipy.spatial.transform import Rotation
    b,rho_s,xi_s,epsilon_s,object_velocity,v_reference,velocity_error = external_pose_step(
        pose,jacobian,velocity,reference,settings,time)
    rho_v = (as_six(initial_velocity_envelope)-as_six(settings['rho_v_final']))*np.exp(
        -as_six(settings['rho_v_rate'])*time)+as_six(settings['rho_v_final'])
    xi_v = velocity_error/rho_v
    if np.any(np.abs(xi_v) >= 1.):
        raise FloatingPointError('Independent external velocity domain failed.')
    epsilon_v = 2*np.arctanh(xi_v)
    feedback = -settings['gv']*2*epsilon_v/(rho_v*(1-xi_v**2))
    rotation = Rotation.from_euler('xyz',pose[3:]).as_matrix()
    offsets = -(rotation@np.asarray(settings['grasp_positions_object']).T).T
    grasps = np.tile(np.eye(6),(2,1,1))
    for hand in range(2):
        grasps[hand,:3,3:] = skew(offsets[hand])
    g = np.concatenate(grasps,axis=0)
    # Independent energy metric construction of the paper's load allocation.
    # Opposite central grasps and equal isotropic designer inertia commute in
    # the printed (20), so this gives the same selected generalized inverse.
    weight = np.diag(np.tile([settings['virtual_mass']]*3+[settings['virtual_inertia']]*3,2))
    plus = weight@g@np.linalg.inv(g.T@weight@g)
    requested_fd = np.zeros((2,6));requested_fd[:,1] = [-force_target,force_target]
    desired_palm = np.concatenate([np.linalg.solve(grasps[hand].T,
        np.linalg.solve(b.T,-requested_fd[hand])) for hand in range(2)])
    internal = (np.eye(12)-plus@g.T)@desired_palm
    palm = plus[arm*6:arm*6+6]@feedback+internal[arm*6:arm*6+6]
    torque = (grasps[arm]@b@jacobian).T@palm
    return torque,dict(surface=velocity_error,xi=xi_s,native_upper=rho_s,native_lower=-rho_s,
        native_barrier_margin=1-np.abs(xi_s),ppc_rho_s=rho_s,ppc_rho_v=rho_v,
        ppc_xi_s=xi_s,ppc_xi_v=xi_v,ppc_epsilon_s=epsilon_s,ppc_epsilon_v=epsilon_v,
        ppc_v_reference=v_reference,ppc_object_velocity=object_velocity,ppc_object_feedback=feedback,
        ppc_load_allocation=plus[arm*6:arm*6+6],ppc_grasp_map=grasps[arm],
        ppc_palm_wrench=palm,ppc_internal_palm_wrench=internal[arm*6:arm*6+6],
        ppc_allocation_identity_error=np.linalg.norm(g.T@plus-np.eye(6)),
        ppc_internal_null_error=np.linalg.norm(g.T@internal))


def verify_sources(audit, manifest):
    required = {'REPRODUCIBILITY.md', 'simulation/run_comparison.py',
        'simulation/run_comparison_all.py', 'simulation/run_experiments.py',
        'simulation/design_excitation.py', 'simulation/models/dual_cr7.xml',
        'simulation/models/arm_nominal.xml', 'simulation/models/scene_metadata.json',
        'simulation/models/plant_truth.json', 'simulation/models/asset_sha256.json'}
    for folder in ('jas', 'original_retrial', 'comparison'):
        required.update(str(path.relative_to(PROJECT)) for path in (HERE/folder).glob('*.py'))
    mismatches = [name for name, digest in manifest.items()
                  if not (PROJECT/name).is_file()
                  or hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() != digest]
    missing = sorted(required-set(manifest))
    audit.check('source_manifest_sha256', not missing and not mismatches,
                checked_files=len(manifest), missing_required_sources=missing, mismatches=mismatches)
    assets = json.loads((HERE/'models/asset_sha256.json').read_text())
    bad = [name for name, digest in assets.items() if not (HERE/'models'/name).is_file()
           or hashlib.sha256((HERE/'models'/name).read_bytes()).hexdigest() != digest]
    audit.check('imported_asset_sha256', not bad, checked_files=len(assets), mismatches=bad)


def audit_online_provenance(audit):
    forbidden = {'qacc', 'qfrc_constraint', 'efc_force', 'qfrc_bias', 'qM',
                 'true_theta', 'data_target'}
    bad, imports, mutations = [], [], []
    names = ['jas/controller.py', 'jas/learning.py', 'original_retrial/controller.py',
             'original_retrial/learning.py', 'comparison/controller.py']
    if (HERE/'comparison/external_ppc.py').exists():
        names.append('comparison/external_ppc.py')
    for name in names:
        tree = ast.parse((HERE/name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in forbidden:
                bad.append(dict(file=name, line=node.lineno, attribute=node.attr))
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = [node.module or ''] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
                if any('exponential_identification' in module for module in modules):
                    imports.append(dict(file=name, line=node.lineno, modules=modules))
            if isinstance(node, ast.FunctionDef) and node.name != '__init__':
                for child in ast.walk(node):
                    if isinstance(child, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                        targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                        if any(any(isinstance(p, ast.Attribute) and p.attr in
                                   ('gain', 'gamma', 'adaptation_gain', 'linear_surface_gain',
                                    'ka', 'k2', 'k3', 'rho_hat', 'radius')
                                   for p in ast.walk(target)) for target in targets):
                            mutations.append(dict(file=name, line=child.lineno))
    runner = ast.parse((HERE/'run_comparison.py').read_text())
    function = next(n for n in runner.body if isinstance(n, ast.FunctionDef) and n.name == 'run_trial')
    loop = next(n for n in function.body if isinstance(n, ast.For))
    force = [n.lineno for n in ast.walk(loop) if isinstance(n, ast.Attribute) and n.attr == 'qfrc_constraint']
    updates = [n.lineno for n in ast.walk(loop) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Attribute) and n.func.attr == 'update']
    truth = [n.lineno for n in ast.walk(function) if isinstance(n, ast.Constant)
             and isinstance(n.value, str) and 'plant_truth.json' in n.value]
    resets = []
    for node in ast.walk(loop):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(any(isinstance(p, ast.Attribute) and p.attr in ('qpos', 'qvel')
                       for p in ast.walk(target)) for target in targets):
                resets.append(node.lineno)
    audit.check('online_laws_have_no_truth_reaction_acceleration_or_changing_gains',
        not bad and not imports and not mutations, forbidden_reads=bad,
        replacement_learning_imports=imports, outside_initialization_gain_mutations=mutations)
    if (HERE/'comparison/external_ppc.py').exists():
        external = ast.parse((HERE/'comparison/external_ppc.py').read_text())
        forbidden_imports, dynamic_reads = [], []
        for node in ast.walk(external):
            if isinstance(node,(ast.Import,ast.ImportFrom)):
                modules = [node.module or ''] if isinstance(node,ast.ImportFrom) else [a.name for a in node.names]
                forbidden_imports.extend(module for module in modules
                    if any(name in module for name in ('jas','learning','original_retrial','exponential_identification')))
            if (isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load)
                    and node.id in ('dynamics','payload_mass','payload_inertia')):
                dynamic_reads.append(node.lineno)
        audit.check('external_is_independent_and_does_not_use_nominal_or_true_dynamics',
            not forbidden_imports and not dynamic_reads,
            forbidden_imports=forbidden_imports, dynamic_placeholder_read_lines=dynamic_reads)
    audit.check('runner_only_diagnoses_force_after_control_and_reads_truth_after_rollout',
        bool(force and updates and min(force) > max(updates)
             and truth and min(truth) > loop.end_lineno and not resets),
        force_read_lines=force, update_lines=updates, truth_read_lines=truth,
        physical_loop_last_line=loop.end_lineno, hidden_state_reset_lines=resets)


def audit_protocol(audit, summary, trial, protocol):
    method = summary.get('method')
    audit.check('declared_method_and_adaptation_semantics', method in METHODS
        and bool(summary.get('has_adaptation')) == (method != 'external_ppc'),
        method=method, has_adaptation=summary.get('has_adaptation'))
    c = summary['settings']
    gain_names = ('chi','ell','ka','k2','k3','lam1','lam2','delta','rho_hat',
                  'filter_rate','window','omega','gamma','adaptation_gain',
                  'linear_surface_gain','evaluation_initial_upper','evaluation_initial_lower',
                  'evaluation_transition_s')
    positive = all(np.isfinite(np.asarray(c[key])).all() and np.all(np.asarray(c[key]) > 0)
                   for key in gain_names)
    paired = (0 < c['alpha'] < 1 < c['beta']
              and abs(c['alpha']+c['beta']-2.) <= 1e-12
              and 0 < c['upsilon2'] < 1 < c['upsilon1'] < 2
              and abs(c['upsilon1']+c['upsilon2']-2.) <= 1e-12)
    scalar = all(np.asarray(c[key]).ndim == 0 for key in
                 ('lam1','lam2','delta','gamma','rho_hat','prior_lower','prior_upper'))
    audit.check('fixed_original_positive_gains_paired_exponents_and_prior_sign_rule',
        positive and paired and scalar and c['prior_lower'] < .9 < c['prior_upper']
        and c['robust_mode'] == 'paper_sign' and c['learning_integrator'] == 'explicit_euler',
        all_required_gains_positive=positive, paired_exponents=paired, scalar_required_parameters=scalar,
        prior=[c['prior_lower'],c['prior_upper']], robust_mode=c['robust_mode'],
        prior_width_R=c['prior_upper']-c['prior_lower'],
        interpretation='Ablated terms remain positively configured and are deleted only by the named method; all other JAS gains are held fixed.')
    phase = summary['phase_entry_thresholds']
    phase_ok = (phase['hold_s'] >= 2. and all(np.isfinite(phase[key]) and phase[key] > 0
        for key in ('position_m','orientation_rad','surface_component_abs','xi_component_abs'))
        and phase['xi_component_abs'] == c['delta'])
    if protocol is not None:
        evaluation = protocol['evaluation']
        phase_ok &= (phase['position_m'] == evaluation['position_tolerance_m']
            and phase['orientation_rad'] == evaluation['orientation_coordinate_tolerance_rad']
            and phase['surface_component_abs'] == evaluation['surface_component_abs']
            and phase['hold_s'] == evaluation['hold_s'])
    audit.check('uniform_predeclared_entry_tolerances_and_at_least_two_second_hold',
                phase_ok, entry_thresholds=phase)
    if trial is None:
        audit.report['protocol_coverage'] = 'Standalone saved outcome; no registered trial was supplied.'
        return
    scenario = next((s for s in protocol['scenarios'] if s['id'] == trial['scenario']), None)
    settings = dict(protocol['settings'], method=trial['method'])
    mismatches = [key for key, value in settings.items()
                  if key not in summary['settings'] or plain(summary['settings'][key]) != plain(value)]
    expected_dt = [trial.get(key, protocol[key]) for key in ('control_dt_s', 'physics_dt_s')]
    audit.check('frozen_protocol_method_scenario_settings_and_sample_grid',
        method == trial['method'] and summary.get('scenario_id') == trial['scenario']
        and scenario == summary['scenario'] and not mismatches
        and summary.get('external_settings') == protocol.get('external_settings')
        and summary['integrator'] == protocol.get('integrator', 'RK4')
        and all(abs(summary[key]-value) < 1e-14 for key, value in
                zip(('control_dt_s', 'physics_dt_s'), expected_dt)),
        trial_id=trial['id'], mismatched_settings=mismatches,
        expected_intervals_s=expected_dt, scenario_matches=scenario == summary['scenario'])
    nominal = next((s for s in protocol['scenarios'] if s['id'] == 'nominal'),None)
    if nominal is not None:
        bs = independent_original_transform(np.asarray(nominal['offset']),np.zeros(6),
            independent_envelope(0.,c),c)
        _, reference_velocity, _ = independent_surface(bs.xi,bs.xidot,c)
        expected_linear_gain = -reference_velocity/bs.xi
        configured_linear_gain = np.asarray(c['linear_surface_gain'])
        expected_log_scale = ((np.asarray(c['chi'])/np.asarray(c['ell']))
            /(1/np.asarray(c['evaluation_initial_upper'])+1/np.asarray(c['evaluation_initial_lower'])))
        configured_log_scale = np.asarray(c['log_bf_scale'])
        log_ok = not configured_log_scale.size or close(configured_log_scale,expected_log_scale,1e-13,1e-12)[0]
        audit.check('comparison_gains_calibrated_once_from_registered_nominal_initial_state',
            close(configured_linear_gain,expected_linear_gain,1e-11,1e-12)[0] and log_ok,
            linear_surface_gain=configured_linear_gain, independently_rebuilt_linear_gain=expected_linear_gain,
            fixed_log_scale=expected_log_scale if not configured_log_scale.size else configured_log_scale,
            interpretation='No individual trajectory, truth, or online adjustment is used to retune either comparator.')


def audit_prefix(audit, arrays, summary, directory, model):
    times = arrays['t']
    n, dt, end = len(times), float(summary['control_dt_s']), float(summary['requested_duration_s'])
    status = summary.get('run_status')
    grid_ok = times.ndim == 1 and np.isfinite(times).all() and close(times, np.arange(n)*dt, 2e-10, 0)[0]
    shapes = arrays['qpos'].shape == (n, model.nq) and arrays['qvel'].shape == (n, model.nv)
    finite = bool(np.isfinite(arrays['qpos']).all() and np.isfinite(arrays['qvel']).all())
    audit.check('complete_saved_prefix_without_dropped_ticks_or_fabricated_states',
        grid_ok and shapes and finite and status in ('completed', 'terminated'),
        samples=n, requested_end_s=end, actual_end_s=float(times[-1]) if n else 0.,
        state_shapes_correct=shapes, saved_states_finite=finite, run_status=status)
    actual_end = float(times[-1]) if n else 0.
    complete = status == 'completed'
    complete_ok = (n > 0 and abs(actual_end-end) < 2e-10
                   and summary.get('termination_reason') is None
                   and summary.get('termination_stage') is None
                   and summary.get('failed_attempt_time_s') is None
                   and not (directory/'failure_state.npz').exists())
    failure = None
    if not complete and (directory/'failure_state.npz').exists():
        with np.load(directory/'failure_state.npz', allow_pickle=False) as source:
            failure = {key: source[key] for key in source.files}
    failed_time = summary.get('failed_attempt_time_s')
    failure_clock = (float(failure.get('expected_time',failure['t'])) if failure is not None else None)
    terminated_ok = (failure is not None and summary.get('termination_reason')
        and summary.get('termination_stage') in ('command', 'domain_guard', 'physics', 'physics_guard', 'update')
        and failed_time is not None and failed_time >= actual_end-1e-10
        and failed_time <= end+1e-10 and abs(failure_clock-failed_time) < 1e-10
        and failure['qpos'].shape == (model.nq,) and failure['qvel'].shape == (model.nv,))
    audit.check('completion_or_explicit_retained_termination_metadata',
        (complete_ok if complete else bool(terminated_ok))
        and abs(float(summary['actual_end_s'])-actual_end) < 2e-10,
        run_status=status, termination_stage=summary.get('termination_stage'),
        termination_reason=summary.get('termination_reason'), failed_attempt_time_s=failed_time,
        failure_state_present=failure is not None,
        actual_failed_mujoco_clock_s=float(failure['t']) if failure is not None else None,
        expected_failed_physical_clock_s=failure_clock,
        interpretation='Explicitly recorded termination is a retained performance outcome, not by itself an invalid formula.')
    if n:
        audit.check('raw_validity_masks_do_not_remove_unsuccessful_samples',
            np.all(arrays['control_valid_mask']) and np.all(arrays['barrier_applicable']),
            invalid_tick_count=int(np.count_nonzero(~arrays['control_valid_mask'])),
            barrier_not_applicable_count=int(np.count_nonzero(~arrays['barrier_applicable'])))
    return failure


def audit_all_tick_geometry(audit, model, arrays, summary, metadata):
    times, cfg = arrays['t'], summary['settings']
    obj, local, sigma = reconstruct_kinematics(model, arrays, metadata)
    center = np.r_[metadata['payload_nominal']['initial_position_world_m'], np.zeros(3)]
    ref = independent_reference(times, center, summary['scenario'])[0]
    error, local_error = obj-ref, local-ref[:, None, :]
    upper, lower, *_ = independent_common_bounds(times, cfg, summary['scenario'].get('nonmonotonic', False))
    common = np.minimum((upper[:, None]-local_error)/upper[:, None],
                        (lower[:, None]+local_error)/lower[:, None])
    actual_common = np.minimum((upper-error)/upper, (lower+error)/lower)
    comparisons = dict(object_pose=(arrays['object_pose'], obj), local_error=(arrays['local_error'], local_error),
        reference=(arrays['reference'], ref), error=(arrays['error'], error),
        sigma_min=(arrays['sigma_min'], sigma))
    errors = {name: discrepancy(*values) for name, values in comparisons.items()}
    audit.check('all_tick_full_scene_fk_local_pose_true_error_and_reference',
        max(errors.values()) <= 3e-9, maximum_errors=errors, samples=len(times),
        reference_profile=summary['scenario'].get('reference_profile', 'moderate'))
    expected_initial = center+np.asarray(summary['scenario']['offset'])
    audit.check('registered_initial_payload_pose_and_zero_velocity',
        discrepancy(obj[0],expected_initial) <= 3e-9 and np.max(np.abs(arrays['qvel'][0])) <= 1e-12,
        maximum_initial_pose_error=discrepancy(obj[0],expected_initial),
        maximum_initial_speed=float(np.max(np.abs(arrays['qvel'][0]))),
        note='Initial condition is checked against the registered scenario, not inferred from a successful later prefix.')
    bounds_errors = {name: discrepancy(arrays[name], value) for name, value in
        (('upper', upper), ('lower', -lower), ('barrier_margin', common),
         ('actual_barrier_margin', actual_common))}
    audit.check('all_tick_common_bounds_and_unclipped_signed_margins',
        max(bounds_errors.values()) <= 2e-8, maximum_errors=bounds_errors,
        actual_violating_samples=int(np.count_nonzero(np.any(actual_common < 0, axis=1))),
        local_violating_samples=int(np.count_nonzero(np.any(common < 0, axis=(1, 2)))),
        interpretation='Negative margins are retained. Common first-second bounds are an extra assessment requirement for original JAS.')
    native_upper = native_lower = native_margin = None
    method = summary['method']
    if method == 'traditional_bf':
        native_upper = np.broadcast_to(upper[:, None], local.shape)
        native_lower = np.broadcast_to(-lower[:, None], local.shape)
        native_margin = common
    elif method == 'external_ppc':
        ec = external_settings(summary)
        rho_s = ((as_six(ec['rho_s_initial'])-as_six(ec['rho_s_final']))
                 *np.exp(-times[:,None]*as_six(ec['rho_s_rate']))+as_six(ec['rho_s_final']))
        native_upper = np.broadcast_to(rho_s[:,None],local.shape)
        native_lower = -native_upper
        native_margin = 1-np.abs(local_error/native_upper)
        extra_errors = dict(ppc_rho_s=discrepancy(arrays['ppc_rho_s'],native_upper),
            ppc_xi_s=discrepancy(arrays['ppc_xi_s'],local_error/native_upper),
            xi=discrepancy(arrays['xi'],local_error/native_upper))
        audit.check('all_tick_external_native_pose_funnel_and_normalized_error',
            max(extra_errors.values()) <= 2e-8 and np.all(np.abs(arrays['ppc_xi_v']) < 1.),
            maximum_errors=extra_errors,
            maximum_recorded_velocity_normalized_error=float(np.max(np.abs(arrays['ppc_xi_v']))),
            note='External native symmetric monotonic funnel is retained independently of the common nonmonotonic assessment boundary.')
    else:
        u, v, *_ = independent_envelope(times, cfg, summary['scenario'].get('nonmonotonic', False))
        ell, chi = np.asarray(cfg['ell']), np.asarray(cfg['chi'])
        eta = ell*np.tanh(chi*local_error)
        native_margin = np.minimum(u[:, None]-eta, v[:, None]+eta)/ell
        with np.errstate(divide='ignore'):
            native_upper = np.broadcast_to((np.arctanh(u/ell)/chi)[:, None], local.shape)
            native_lower = np.broadcast_to((-np.arctanh(v/ell)/chi)[:, None], local.shape)
    if native_margin is not None:
        native_errors = {name: discrepancy(arrays[name], value) for name, value in
            (('native_upper', native_upper), ('native_lower', native_lower),
             ('native_barrier_margin', native_margin))}
        audit.check('native_control_envelope_is_distinct_and_not_clipped',
            max(native_errors.values()) <= 2e-8 and np.all(native_margin > 0),
            maximum_errors=native_errors, minimum_native_margin=float(native_margin.min()),
            note='Saved successful commands must be inside their own control domain; common-margin violations remain permitted outcomes.')
    reconstructed = dict(t=times, object_pose=obj, local_pose=local, error=error,
        local_error=local_error, reference=ref, sigma_min=sigma, upper=upper, lower=-lower,
        barrier_margin=common, actual_barrier_margin=actual_common,
        actual_violation_mask=actual_common < 0,
        local_violation_mask=common < 0,
        saturation_mask=np.abs(arrays['torque_command']-arrays['torque_applied']) > 1e-12)
    if native_margin is not None:
        reconstructed.update(native_upper=native_upper, native_lower=native_lower,
                             native_barrier_margin=native_margin)
    return reconstructed


def audit_learning(audit, arrays, summary, directory):
    method, c = summary['method'], summary['settings']
    if method == 'external_ppc':
        channels = ('theta', 'theta_control', 'filtered_regressor', 'filtered_torque',
                    'lambda_min', 'prediction_error','regressor','momentum','remainder',
                    'A','learning_drive','known_torque_left','known_torque_right')
        audit.check('external_controller_explicitly_has_no_adaptation_or_theta_metrics',
            not summary['has_adaptation'] and all(np.isnan(arrays[k]).all() for k in channels)
            and not (directory/'learning_state.npz').exists(),
            unavailable_learning_channels=list(channels))
        return None
    with np.load(directory/'learning_state.npz', allow_pickle=False) as source:
        learned = {key: source[key] for key in source.files}
    active = audit_original_filter_and_history(audit, arrays, learned, c, summary['ie_detection_times_s'])
    if active is None:
        return learned
    g, p = active
    theta, old = arrays['theta'], arrays['theta_control']
    gain, gamma = np.asarray(c['adaptation_gain']), float(c['gamma'])
    effective_gamma = 0. if method == 'no_history' else gamma
    lower, upper = float(c['prior_lower']), float(c['prior_upper'])
    audit.check('constant_positive_configured_gain_and_explicit_one_module_history_ablation',
        gain.shape == (8,) and np.isfinite(gain).all() and np.all(gain > 0)
        and gamma > 0 and c['learning_integrator'] == 'explicit_euler',
        Upsilon_diagonal=gain, configured_gamma=gamma, effective_gamma=effective_gamma,
        retained='Original -YS, causal filter, raw IE/frozen history and component projection for all five adaptive methods.')
    drive = np.einsum('napc,nac->nap', arrays['regressor'], arrays['surface'])
    epsilon = p-np.einsum('napq,naq->nap', g, old)
    direction = -drive+effective_gamma*epsilon
    tangent = project_direction(old[1:], direction[1:], lower, upper)
    trial = old[1:]+np.diff(arrays['t'])[:, None, None]*gain*tangent
    expected = np.clip(trial, lower, upper)
    checks = dict(initial_control_error=discrepancy(old[0], np.full((2, 8), .9)),
        initial_update_error=discrepancy(theta[0], old[0]),
        continuity_error=discrepancy(old[1:], theta[:-1]),
        learning_drive_error=discrepancy(drive, arrays['learning_drive']),
        preupdate_epsilon_error=discrepancy(epsilon, arrays['prediction_error']),
        updated_theta_error=discrepancy(expected, theta[1:]))
    audit.check('every_tick_complete_projected_euler_learning_update',
        all(np.isfinite(theta).ravel()) and checks['initial_control_error'] <= 1e-13
        and checks['initial_update_error'] <= 1e-13 and checks['continuity_error'] <= 1e-13
        and checks['learning_drive_error'] <= 2e-8 and checks['preupdate_epsilon_error'] <= 2e-6
        and checks['updated_theta_error'] <= 5e-9, **checks,
        effective_gamma=effective_gamma, projection='Original tangent blocking, then box-projected Euler step; no replaced history law.')
    counts = np.sum(np.any((trial < lower) | (trial > upper), axis=2), axis=0)
    audit.check('theta_prior_box_and_projection_count_are_not_hidden',
        np.isfinite(theta).all() and np.all(theta >= lower-1e-12) and np.all(theta <= upper+1e-12)
        and np.array_equal(counts, summary['numerical_projection_counts']),
        lower=lower, upper=upper, actual_component_min=theta.min(axis=0),
        actual_component_max=theta.max(axis=0), reconstructed_clip_step_counts=counts,
        recorded_clip_step_counts=summary['numerical_projection_counts'],
        interpretation='Measured projection invariance is not evidence of an exponential parameter-error envelope.')
    truth = np.asarray(summary['true_parameters_offline_only'])
    audit_offline_residual(audit, learned, truth)
    return learned


def audit_control_formulas(audit, arrays, summary, metadata, samples=31):
    """Nominal-model algebra, with no call to ComparisonController.command."""
    if summary['method'] == 'external_ppc':
        return audit_external_formulas(audit, arrays, summary, metadata, samples)
    from jas.dynamics import LinkDynamics, coupled_bases
    from run_experiments import kinematics
    from original_retrial.controller import OriginalController, OriginalSettings
    method, c = summary['method'], summary['settings']
    center = np.r_[metadata['payload_nominal']['initial_position_world_m'], np.zeros(3)]
    payload = metadata['payload_nominal']
    nominal = LinkDynamics(HERE/'models/arm_nominal.xml')
    kins = [kinematics(metadata, name) for name in ('left', 'right')]
    indices = np.unique(np.linspace(0, len(arrays['t'])-1, min(samples, len(arrays['t']))).astype(int))
    keys = ('torque_command', 'regressor', 'momentum', 'remainder', 'surface', 'xi',
            'A', 'known_torque_left', 'known_torque_right', 'learning_drive')
    maxima, scaled = dict.fromkeys(keys, 0.), dict.fromkeys(keys, 0.)
    original_equal = True
    original = None
    if method in ('proposed', 'no_history'):
        accepted = OriginalSettings.__dataclass_fields__
        settings = OriginalSettings(**{k:v for k,v in c.items() if k in accepted})
        reference = lambda t, center: independent_reference(np.asarray(t), center, summary['scenario'])
        original = [OriginalController(kins[i], nominal, center, payload['mass_kg'],
                    payload['diagonal_inertia_kg_m2'], i, settings,
                    summary['scenario'].get('nonmonotonic', False), reference=reference) for i in range(2)]
    for tick in indices:
        t = float(arrays['t'][tick])
        ref, dref, ddref = independent_reference(np.asarray(t), center, summary['scenario'])
        for arm in range(2):
            sl = slice(6*arm, 6*arm+6)
            q, qd = arrays['qpos'][tick, sl], arrays['qvel'][tick, sl]
            pose, j, jd = kins[arm].evaluate(q, qd)
            ed, error = j@qd-dref, pose-ref
            if method == 'traditional_bf':
                scale = np.asarray(c['log_bf_scale'])
                if not scale.size:
                    scale = ((np.asarray(c['chi'])/np.asarray(c['ell']))
                             /(1/np.asarray(c['evaluation_initial_upper'])+1/np.asarray(c['evaluation_initial_lower'])))
                bs = independent_log_transform(error, ed,
                    independent_common_bounds(t, c, summary['scenario'].get('nonmonotonic', False)), scale)
            else:
                bs = independent_original_transform(error, ed,
                    independent_envelope(t, c, summary['scenario'].get('nonmonotonic', False)), c)
            if method == 'linear_surface':
                xir = -np.asarray(c['linear_surface_gain'])*bs.xi
                xirdot = -np.asarray(c['linear_surface_gain'])*bs.xidot
                surface = bs.xidot-xir
            else:
                surface, xir, xirdot = independent_surface(bs.xi, bs.xidot, c)
            ji = np.linalg.inv(j)
            a = ji/bs.a
            ad = -ji@jd@a-a*(bs.adot/bs.a)
            known = bs.a*dref-bs.b
            v = a@known
            vd = ad@known+a@(bs.adot*dref+bs.a*ddref-bs.bdot)
            qr, qrr = a@xir+v, ad@xir+a@xirdot+vd
            db, cb, wb = coupled_bases(nominal.evaluate(q, qd), j, jd, pose, j@qd,
                payload['mass_kg'], payload['diagonal_inertia_kg_m2'])
            wb[6] = .05*qd
            y = np.stack([a.T@(db[p]@qrr+cb[p]@qr+wb[p]) for p in range(8)])
            momentum = np.stack([a.T@db[p]@qd for p in range(8)])
            remainder = np.stack([a.T@(wb[p]-cb[p].T@qd)-ad.T@db[p]@qd for p in range(8)])
            fd = np.array([0., (-1 if arm == 0 else 1)*c['internal_force'], 0., 0., 0., 0.])
            robust = y.T@((c['prior_upper']-c['prior_lower'])*np.sign(y@surface))
            rho = c['rho_hat']*np.linalg.norm(a, 2)**2
            feedback = y.T@arrays['theta_control'][tick, arm]-(np.asarray(c['ka'])+rho)*surface-robust
            if method != 'linear_reaching':
                feedback -= (np.asarray(c['k2'])*np.sign(surface)*np.abs(surface)**c['upsilon1']
                             +np.asarray(c['k3'])*np.sign(surface)*np.abs(surface)**c['upsilon2'])
            torque = j.T@(bs.a*feedback-fd)
            previous = arrays['torque_applied'][max(0, tick-1), sl]
            held = arrays['torque_applied'][tick, sl] if tick == 0 else previous
            rebuilt = dict(torque_command=torque, regressor=y, momentum=momentum,
                remainder=remainder, surface=surface, xi=bs.xi, A=a,
                known_torque_left=a.T@(held+j.T@fd),
                known_torque_right=a.T@(arrays['torque_applied'][tick, sl]+j.T@fd),
                learning_drive=y@surface)
            for key, value in rebuilt.items():
                actual = arrays[key][tick, sl] if key == 'torque_command' else arrays[key][tick, arm]
                ok, error_value, tolerance = close(actual, value)
                maxima[key] = max(maxima[key], error_value)
                scaled[key] = max(scaled[key], error_value/tolerance)
            if original is not None:
                original[arm].learner.theta = arrays['theta_control'][tick, arm].copy()
                tau_original, _ = original[arm].command(t, q, qd)
                # Independent reference arithmetic can differ at roundoff;
                # full bitwise equivalence is also a dedicated unit test.
                original_equal &= close(tau_original, arrays['torque_command'][tick, sl], 3e-8, 2e-12)[0]
    audit.check('sampled_torque_formula_and_exact_declared_module_ablation',
        scaled['torque_command'] <= 1 and original_equal,
        method=method, sampled_ticks=len(indices), maximum_errors=maxima,
        maximum_tolerance_ratios=scaled, matches_original_controller_when_applicable=original_equal,
        interpretation='Only traditional transform, linear surface, linear reaching, or zero history coefficient changes in its named ablation; original prior-width sign remains.')
    signal_keys = set(keys)-{'torque_command'}
    audit.check('sampled_local_filter_inputs_regressor_surface_and_known_torque',
        all(scaled[k] <= 1 for k in signal_keys), sampled_ticks=len(indices),
        maximum_errors={k:maxima[k] for k in signal_keys},
        maximum_tolerance_ratios={k:scaled[k] for k in signal_keys})


def audit_measured_force_and_physics(audit, model, arrays, summary, metadata, samples=31):
    """Offline Newton--Euler and reaction check; contact is not silently erased."""
    from validate_identification import NominalNewtonEuler
    from run_experiments import kinematics
    from jas.dynamics import LinkDynamics, coupled_bases
    from jas.geometry import angular_map
    from scipy.spatial.transform import Rotation
    actual_truth = json.loads((HERE/'models/plant_truth.json').read_text())
    truth = np.array([actual_truth[name+'_density_multipliers']
        +[actual_truth[name+'_viscous_multiplier'], actual_truth['payload_density_multiplier']]
        for name in ('left','right')])
    logged_truth = np.asarray(summary['true_parameters_offline_only'])
    audit.check('offline_truth_metrics_use_physics_asset_and_never_change_online_inputs',
                discrepancy(truth, logged_truth) <= 1e-14,
                actual_parameters_offline_only=truth, source='models/plant_truth.json')
    payload = metadata['payload_nominal']
    mass, inertia = payload['mass_kg'], np.diag(payload['diagonal_inertia_kg_m2'])
    nominal, matrices = NominalNewtonEuler(), LinkDynamics(HERE/'models/arm_nominal.xml')
    kins = [kinematics(metadata, name) for name in ('left','right')]
    indices = np.unique(np.linspace(0,len(arrays['t'])-1,min(samples,len(arrays['t']))).astype(int))
    robot_error = payload_error = saved_internal_error = saved_interaction_error = basis_error = 0.
    non_equality_reaction_peak = 0.
    for tick in indices:
        data = mujoco.MjData(model)
        data.qpos[:] = arrays['qpos'][tick]; data.qvel[:] = arrays['qvel'][tick]
        data.ctrl[:] = arrays['torque_applied'][tick]; data.time = arrays['t'][tick]
        mujoco.mj_forward(model,data)
        eq_force = np.asarray(data.efc_force).copy()
        eq_force[np.asarray(data.efc_type) != mujoco.mjtConstraint.mjCNSTR_EQUALITY] = 0.
        equality = np.zeros(model.nv)
        mujoco.mj_mulJacTVec(model,data,equality,eq_force)
        non_equality_reaction_peak = max(non_equality_reaction_peak,
            discrepancy(equality,data.qfrc_constraint))
        rotation = data.xmat[model.body('payload').id].reshape(3,3)
        omega, alpha = rotation@data.qvel[15:18], rotation@data.qacc[15:18]
        iw = rotation@inertia@rotation.T
        wrench = np.r_[data.qfrc_constraint[12:15], rotation@data.qfrc_constraint[15:18]]
        predicted = truth[0,7]*np.r_[mass*(data.qacc[12:15]-model.opt.gravity),
                                   iw@alpha+np.cross(omega,iw@omega)]
        payload_error = max(payload_error,discrepancy(wrench,predicted))
        forces = []
        for arm in range(2):
            sl = slice(arm*6,arm*6+6)
            q, qd, qdd = data.qpos[sl], data.qvel[sl], data.qacc[sl]
            robot_basis = nominal.forces(q,qd,qdd)
            predicted_robot = truth[arm,:6]@robot_basis+truth[arm,6]*.05*qd
            robot_error = max(robot_error,discrepancy(predicted_robot,
                data.qfrc_actuator[sl]+data.qfrc_constraint[sl]))
            pose,j,jd = kins[arm].evaluate(q,qd)
            xd, xdd = j@qd, j@qdd+jd@qd
            mapping = angular_map(pose[3:])
            _, pitch,yaw = pose[3:]
            cp,sp,cy,sy = np.cos(pitch),np.sin(pitch),np.cos(yaw),np.sin(yaw)
            dp = np.array([[-cy*sp,0.,0.],[-sy*sp,0.,0.],[-cp,0.,0.]])
            dy = np.array([[-sy*cp,-cy,0.],[cy*cp,-sy,0.],[0.,0.,0.]])
            mdot = dp*xd[4]+dy*xd[5]
            omega_local = mapping@xd[3:]
            alpha_local = mapping@xdd[3:]+mdot@xd[3:]
            rl = Rotation.from_euler('xyz',pose[3:]).as_matrix()
            il = rl@inertia@rl.T
            payload_basis = np.r_[mass*(xdd[:3]-model.opt.gravity),
                mapping.T@(il@alpha_local+np.cross(omega_local,il@omega_local))]
            db,cb,wb = coupled_bases(matrices.evaluate(q,qd),j,jd,pose,xd,mass,
                                    payload['diagonal_inertia_kg_m2'])
            wb[6] = .05*qd
            expected = np.vstack((robot_basis,.05*qd,.5*j.T@payload_basis))
            actual = np.stack([db[p]@qdd+cb[p]@qd+wb[p] for p in range(8)])
            basis_error = max(basis_error,discrepancy(actual,expected))
            forces.append(np.linalg.solve(j.T,data.qfrc_constraint[sl]))
        forces = np.asarray(forces)
        internal = forces-.5*forces.sum(axis=0)
        saved_interaction_error = max(saved_interaction_error,discrepancy(forces,arrays['interaction_force'][tick]))
        saved_internal_error = max(saved_internal_error,discrepancy(internal,arrays['internal_force'][tick]))
    audit.check('offline_newton_euler_balance_and_measured_force_reconstruction',
        max(robot_error,payload_error) <= 3e-6
        and max(saved_interaction_error,saved_internal_error) <= 3e-7,
        sampled_ticks=len(indices), maximum_robot_balance_error_Nm=robot_error,
        maximum_payload_balance_error=payload_error,
        maximum_saved_interaction_discrepancy=saved_interaction_error,
        maximum_saved_internal_discrepancy=saved_internal_error,
        maximum_non_equality_generalized_reaction=non_equality_reaction_peak,
        interpretation='qacc and truth are validation only. Full reaction, including any contact/limit contribution, is retained and disclosed.')
    audit.check('nominal_coupled_basis_matches_independent_local_newton_euler',
                basis_error <= 3e-6, maximum_basis_error=basis_error, sampled_ticks=len(indices))


def audit_external_formulas(audit, arrays, summary, metadata, samples):
    from run_experiments import kinematics
    ec = external_settings(summary)
    names = ('rho_s_initial','rho_s_final','rho_s_rate','rho_v_final','rho_v_rate')
    vectors_positive = all(as_six(ec[key]).shape == (6,) and np.isfinite(as_six(ec[key])).all()
                           and np.all(as_six(ec[key]) > 0) for key in names)
    constants_positive = all(np.isfinite(ec[key]) and ec[key] > 0
        for key in ('gs','gv','velocity_initial_slack','desired_pitch_max','virtual_mass','virtual_inertia'))
    pitch_ok = as_six(ec['rho_s_initial'])[4]+ec['desired_pitch_max'] < np.pi/2
    pose_limits_ordered = np.all(as_six(ec['rho_s_initial']) > as_six(ec['rho_s_final']))
    audit.check('external_fixed_positive_funnels_gains_and_pitch_assumption',
        vectors_positive and constants_positive and pose_limits_ordered and pitch_ok,
        fixed_design=ec, initial_pose_funnel_strictly_larger=pose_limits_ordered,
        paper_pitch_condition=pitch_ok,
        source='https://arxiv.org/html/1710.11088v4#S4.SS2')
    center = np.r_[metadata['payload_nominal']['initial_position_world_m'],np.zeros(3)]
    kins = [kinematics(metadata,name) for name in ('left','right')]
    rv0, configuration_errors, initial_domain, norm_initialization = [], [], [], []
    configurations = summary.get('controller_configurations',[])
    for arm in range(2):
        sl = slice(arm*6,arm*6+6)
        q0,qd0 = arrays['qpos'][0,sl],arrays['qvel'][0,sl]
        pose0,j0 = kins[arm].pose_jacobian(q0)
        ref0 = independent_reference(np.asarray(0.),center,summary['scenario'])[0]
        *_, ev0 = external_pose_step(pose0,j0,qd0,ref0,ec,0.)
        norm_initial = np.full(6,np.linalg.norm(ev0)+ec['velocity_initial_slack'])
        resolved = norm_initial if ec['rho_v_initial'] is None else as_six(ec['rho_v_initial'])
        rv0.append(resolved)
        norm_initialization.append(close(resolved,norm_initial,1e-12,1e-12)[0])
        initial_domain.append(bool(np.all(np.abs(ev0) < resolved)
            and np.all(resolved > as_six(ec['rho_v_final']))))
        if len(configurations) != 2:
            configuration_errors.append('Missing two resolved controller configurations.')
            continue
        config = configurations[arm]
        if config.get('has_adaptation') or config.get('has_dynamic_filter'):
            configuration_errors.append('Unexpected external adaptation/filter flag.')
        configured = config['settings']
        for key,value in ec.items():
            if plain(configured.get(key)) != plain(value):
                configuration_errors.append(f'arm {arm} fixed setting {key}')
        if not close(configured.get('resolved_rho_v_initial'),resolved,1e-12,1e-12)[0]:
            configuration_errors.append(f'arm {arm} resolved rho_v_initial')
    audit.check('external_velocity_funnel_initialization_uses_only_t0_encoders',
        not configuration_errors and all(initial_domain), errors=configuration_errors,
        independently_rebuilt_rho_v_initial=rv0, initial_velocity_domains=initial_domain,
        paper_Step_II_a_norm_plus_slack_satisfied=norm_initialization,
        initialization_construction=('paper Step II-a norm plus positive slack'
            if ec['rho_v_initial'] is None else 'componentwise constants fixed in the common protocol; paper experimental parameter class'),
        interpretation='The generic norm-plus-slack construction is distinguished from fixed componentwise experimental funnels. Both must contain the initial measured velocity error and remain fixed; no truth or later retuning is permitted.')
    indices = np.unique(np.linspace(0,len(arrays['t'])-1,min(samples,len(arrays['t']))).astype(int))
    fields = ('torque_command','surface','xi','native_upper','native_lower','native_barrier_margin')+EXTERNAL_FIELDS
    maxima, scaled = dict.fromkeys(fields,0.),dict.fromkeys(fields,0.)
    allocation_error = internal_null_error = 0.
    for tick in indices:
        t = arrays['t'][tick]
        ref = independent_reference(np.asarray(t),center,summary['scenario'])[0]
        for arm in range(2):
            sl = slice(arm*6,arm*6+6)
            q,qd = arrays['qpos'][tick,sl],arrays['qvel'][tick,sl]
            pose,j = kins[arm].pose_jacobian(q)
            torque, expected = independent_external_command(pose,j,qd,ref,ec,t,rv0[arm],arm,
                summary['settings']['internal_force'])
            expected['torque_command'] = torque
            for key,value in expected.items():
                actual = arrays[key][tick,sl] if key == 'torque_command' else arrays[key][tick,arm]
                _,error,tolerance = close(actual,value,3e-8,1e-11)
                maxima[key] = max(maxima[key],error);scaled[key] = max(scaled[key],error/tolerance)
            allocation_error = max(allocation_error,expected['ppc_allocation_identity_error'])
            internal_null_error = max(internal_null_error,expected['ppc_internal_null_error'])
    audit.check('independent_external_published_controller_formula',
        max(scaled.values()) <= 1, sampled_ticks=len(indices),
        maximum_errors=maxima, maximum_tolerance_ratios=scaled,
        source='https://arxiv.org/html/1710.11088v4',
        checked='IV-B (25)-(36), load allocation (20), known internal-wrench task Remark 5; no desired-velocity feedforward or JAS control block.')
    audit.check('external_load_distribution_and_internal_nullspace_identity',
        allocation_error <= 1e-11 and internal_null_error <= 1e-10,
        maximum_allocation_identity_error=allocation_error,
        maximum_internal_wrench_nullspace_error=internal_null_error,
        independent_construction='Positive designer metric W: G_M_plus=W G (G.T W G)^-1 for the central opposite symmetric grasp geometry.')


def audit_replay(audit, model, arrays, summary, failure, sample_count):
    n = len(arrays['t'])
    dt, physics_dt = float(summary['control_dt_s']), float(summary['physics_dt_s'])
    ratio = round(dt/physics_dt)
    interval_ok = summary['integrator'] == 'RK4' and ratio >= 1 and abs(ratio*physics_dt-dt) < 1e-12
    audit.check('configured_physics_is_RK4_with_integer_control_substeps', interval_ok,
        integrator=summary['integrator'], control_dt_s=dt, physics_dt_s=physics_dt, substeps=ratio)
    if n >= 2 and interval_ok:
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        replay = replay_intervals(model, arrays, dt, physics_dt, sample_count)
        audit.check('sampled_full_mujoco_held_torque_interval_replay',
            replay['maximum_qpos_error'] <= 3e-8 and replay['maximum_qvel_error'] <= 3e-7
            and replay['maximum_time_error_s'] <= 2e-10, **replay,
            note='Replay contacts/warnings are disclosed; their occurrence is a performance outcome.')
    if failure is not None and n and interval_ok:
        elapsed = float(failure.get('expected_time',failure['t']))-float(arrays['t'][-1])
        steps = round(elapsed/physics_dt)
        if elapsed < -1e-10 or abs(steps*physics_dt-elapsed) > 2e-10 or steps > ratio:
            audit.check('retained_failure_endpoint_is_causal', False, elapsed_s=elapsed, physical_steps=steps)
            return
        data = mujoco.MjData(model)
        data.qpos[:] = arrays['qpos'][-1]; data.qvel[:] = arrays['qvel'][-1]
        data.time = arrays['t'][-1]; data.ctrl[:] = arrays['torque_applied'][-1]
        mujoco.mj_forward(model, data)
        for _ in range(steps):
            mujoco.mj_step(model, data)
        qp, qv = discrepancy(data.qpos, failure['qpos']), discrepancy(data.qvel, failure['qvel'])
        time_error = abs(data.time-float(failure['t']))
        audit.check('retained_failure_endpoint_is_causal', qp <= 3e-8 and qv <= 3e-7 and time_error <= 2e-10,
            physical_steps=steps, maximum_qpos_error=qp, maximum_qvel_error=qv,
            failed_mujoco_clock_error_s=time_error,
            note='No command is fabricated for a failed endpoint. An explicitly retained MuJoCo automatic reset is replayed and its actual clock is disclosed.')


def theta_performance(arrays, summary):
    if not summary['has_adaptation']:
        return dict(applicable=False, reason='External protocol has no parameter learner.')
    theta, times = arrays['theta'], arrays['t']
    truth = np.asarray(summary['true_parameters_offline_only'])
    error = np.linalg.norm(theta-truth, axis=2)
    selected = times >= max(times[0], times[-1]-2.)-1e-12
    centered = times[selected]-times[selected].mean()
    denominator = centered@centered
    slope = (np.einsum('n,nap->ap', centered, theta[selected])/denominator
             if denominator > 0 else np.full((2, 8), np.nan))
    return dict(applicable=True, component_minimum=theta.min(axis=0),
        component_maximum=theta.max(axis=0), maximum_true_error_norm=error.max(axis=0),
        final_true_error_norm=error[-1], final_estimate=theta[-1],
        final_two_second_drift_per_s=slope,
        maximum_absolute_one_step_change=np.max(np.abs(np.diff(theta, axis=0)), axis=0) if len(times)>1 else np.zeros((2,8)),
        projection_step_counts=summary['numerical_projection_counts'],
        interpretation='Truth is evaluation-only. Box bounds and finite drift do not prove an exponential error envelope.')


def audit_metrics(audit, arrays, summary, rebuilt, metadata):
    times, errors = arrays['t'], rebuilt['error']
    thresholds = summary['phase_entry_thresholds']
    completed = summary['run_status'] == 'completed'
    end, hold = summary['requested_duration_s'], float(thresholds['hold_s'])
    entry = lambda inside: independent_entry(times, inside, hold, end, completed)
    finite = np.isfinite(errors).all(axis=1)
    pnorm, onorm = np.linalg.norm(errors[:, :3], axis=1), np.linalg.norm(errors[:, 3:], axis=1)
    p, o = float(thresholds['position_m']), float(thresholds['orientation_rad'])
    condition = finite & (pnorm <= p) & (onorm <= o)
    tracking = dict(position_entry_s=entry(finite & (pnorm <= p)),
        orientation_coordinate_entry_s=entry(finite & (onorm <= o)), simultaneous_entry_s=entry(condition),
        componentwise_entry_s=[entry(finite & (np.abs(errors[:, k]) <= (p if k<3 else o))) for k in range(6)])
    logged = summary['tracking_entry']
    tracking_matches = all(tracking[k] == logged.get(k) for k in tracking)
    jas_phase = summary['method'] != 'external_ppc'
    surface_entry = entry(np.max(np.abs(arrays['surface']), axis=(1,2)) <= thresholds['surface_component_abs']) if jas_phase else None
    xi_entry = entry(np.max(np.abs(arrays['xi']), axis=(1,2)) <= thresholds['xi_component_abs']) if jas_phase else None
    tolerance_table = []
    for factor in (.5, 1., 2.):
        tolerance_table.append(dict(multiplier=factor, position_norm_m=p*factor,
            orientation_coordinate_norm_rad=o*factor,
            simultaneous_entry_s=entry(finite & (pnorm <= p*factor) & (onorm <= o*factor))))
    margins = rebuilt['actual_barrier_margin']
    violating = int(np.count_nonzero(np.any(margins < 0, axis=1)))
    min_margin = float(margins.min())
    audit.check('independent_entry_times_violations_and_thresholds_not_filtered',
        tracking_matches and surface_entry == summary['surface_entry_s'] and xi_entry == summary['xi_layer_entry_s']
        and summary.get('jas_phase_criteria_applicable') == jas_phase
        and violating == summary['constraint_violating_samples']
        and abs(min_margin-summary['minimum_common_normalized_gap']) <= 2e-8,
        reconstructed_tracking_entry=tracking, reconstructed_surface_entry_s=surface_entry,
        reconstructed_xi_layer_entry_s=xi_entry, actual_violating_saved_ticks=violating,
        JAS_phase_criteria_applicable=jas_phase,
        minimum_common_normalized_gap=min_margin, hold_s=hold,
        interpretation='Entry must persist to the requested end for at least two seconds. Truncated trials are right censored.')
    limits = np.tile(metadata['torque_limits_Nm'], 2)
    torque = torque_statistics(times, arrays['torque_command'], arrays['torque_applied'], limits)
    audit.check('raw_commands_and_exact_physical_torque_clipping_retained',
        torque['all_finite'] and torque['expected_clipping_max_error_Nm'] <= 1e-12,
        expected_clip_error_Nm=torque['expected_clipping_max_error_Nm'],
        saturated_tick_count=torque['saturated_tick_count'], saturation_fraction=torque['saturation_fraction'],
        maximum_raw_limit_ratio=torque['maximum_raw_limit_ratio'],
        interpretation='Nonzero saturation is reported, never used to remove a sample or fail an otherwise correct physical clip.')
    logged_torque = summary['torque_metrics']
    scalar_fields = ('saturated_tick_count', 'saturation_fraction', 'maximum_raw_limit_ratio',
                     'maximum_applied_limit_ratio', 'expected_clipping_max_error_Nm')
    torque_errors = {key:abs(torque[key]-logged_torque[key]) for key in scalar_fields}
    audit.check('saved_torque_metrics_match_all_raw_commands', max(torque_errors.values()) <= 1e-12,
                scalar_metric_errors=torque_errors)
    poslimits = np.tile(np.asarray(metadata['joint_position_limits_rad']), (2,1))
    speedlimits = np.tile(metadata['joint_velocity_limits_rad_s'], 2)
    joint = np.minimum(arrays['qpos'][:,:12]-poslimits[:,0], poslimits[:,1]-arrays['qpos'][:,:12])
    velocity_ratio = np.abs(arrays['qvel'][:,:12])/speedlimits
    windows = dict(full=(times[0], times[-1]), final_two_seconds=(max(times[0],times[-1]-2.),times[-1]))
    excitation_end = float(summary['scenario'].get('excitation_duration', times[-1]))
    if times[-1] >= 2.:
        windows['motion_after_two_seconds'] = (2., min(times[-1], excitation_end))
    forces = {name:force_statistics(times, arrays['internal_force'], arrays['desired_internal_force'], interval)
              for name,interval in windows.items()}
    expected_fd = np.zeros((2,6))
    expected_fd[:,1] = [-summary['settings']['internal_force'],summary['settings']['internal_force']]
    target_error = discrepancy(arrays['desired_internal_force'],
        np.broadcast_to(expected_fd,arrays['desired_internal_force'].shape))
    force_errors = []
    logged_force = summary['internal_force_metrics_final_2s']
    recomputed_force = forces['final_two_seconds']
    for arm in ('left','right'):
        for unit in ('force_xyz_N','euler_moment_xyz_Nm'):
            for name in ('actual_mean','desired_mean','error_bias','error_rms_components',
                         'error_vector_rms','actual_ac_vector_rms','error_ac_vector_rms',
                         'actual_peak_to_peak_components','sampled_max_error_vector_norm'):
                force_errors.append(discrepancy(
                    np.asarray(logged_force['arms'][arm][unit][name]),
                    np.asarray(recomputed_force['arms'][arm][unit][name])))
    audit.check('desired_internal_force_and_logged_force_statistics_match_raw_samples',
        target_error <= 1e-12 and max(force_errors) <= 1e-10,
        maximum_desired_force_error=target_error, maximum_force_statistic_discrepancy=max(force_errors),
        units='Cartesian xyz N and generalized Euler-coordinate moments Nm are kept separate.')
    audit.report['performance'] = dict(run_status=summary['run_status'],
        termination_reason=summary.get('termination_reason'), requested_duration_s=end,
        recorded_end_s=float(times[-1]), all_recorded_samples_retained=len(times),
        tracking_entry=tracking, tolerance_sensitivity=tolerance_table,
        surface_entry_s=surface_entry, xi_layer_entry_s=xi_entry,
        phase_coordinate_semantics=('External geometric velocity error and normalized pose error; these are not JAS S/xi and are not ranked as common phases.'
            if summary['method']=='external_ppc' else 'Method-specific sliding variable and barrier coordinate; cross-structure ranking uses common physical tracking tolerances.'),
        minimum_common_normalized_gap=min_margin, common_violating_tick_count=violating,
        local_common_violating_tick_count=int(np.count_nonzero(np.any(rebuilt['local_violation_mask'],axis=(1,2)))),
        maximum_position_error_norm_m=float(pnorm.max()), final_position_error_norm_m=float(pnorm[-1]),
        final_orientation_coordinate_error_norm_rad=float(onorm[-1]),
        final_two_second_position_error_rms_m=float(np.sqrt(np.mean(pnorm[times >= times[-1]-2.]**2))),
        torque=torque, theta=theta_performance(arrays, summary), internal_force=forces,
        saved_tick_joint_limit_margin_rad=float(joint.min()), saved_tick_max_speed_limit_ratio=float(velocity_ratio.max()),
        all_physics_step_joint_limit_margin_rad=summary['minimum_joint_limit_margin_rad'],
        all_physics_step_max_speed_limit_ratio=summary['max_joint_speed_limit_ratio'],
        maximum_contact_count=summary['max_contact_count'], mujoco_warnings=summary['mujoco_warnings'],
        maximum_weld_residual=float(np.max(np.abs(arrays['weld_residual']))),
        interpretation='Performance outcomes include every registered success, termination, violation and saturated tick; no method is required to outperform proposed by the validity audit.')


def reference_sensitivity(arrays, summary, reference):
    with np.load(reference/'trajectory.npz', allow_pickle=False) as source:
        ref = {key:source[key] for key in ('t','error','internal_force','desired_internal_force','torque_applied','theta')}
    ref_summary = json.loads((reference/'summary.json').read_text())
    lo, hi = max(arrays['t'][0], ref['t'][0]), min(arrays['t'][-1], ref['t'][-1])
    selected = (arrays['t'] >= lo) & (arrays['t'] <= hi)
    target = arrays['t'][selected]
    errors, differences = {}, {}
    for key in ('error','torque_applied','internal_force'):
        flat = ref[key].reshape(len(ref['t']), -1)
        interpolated = np.column_stack([np.interp(target, ref['t'], flat[:, k]) for k in range(flat.shape[1])])
        difference = arrays[key][selected].reshape(len(target),-1)-interpolated
        differences[key] = difference
        errors[key] = dict(maximum_absolute_difference=float(np.max(np.abs(difference))),
                           componentwise_rms_difference=np.sqrt(np.mean(difference**2,axis=0)))
    interval = (max(lo, hi-2.), hi)
    force = {name:force_statistics(values['t'], values['internal_force'], values['desired_internal_force'], interval)
             for name, values in (('candidate',arrays),('reference',ref))}
    theta = dict(applicable=False,reason='External protocol has no parameter learner.')
    if summary['has_adaptation']:
        interpolated_theta = np.empty_like(arrays['theta'][selected])
        for arm in range(2):
            for parameter in range(8):
                interpolated_theta[:,arm,parameter] = np.interp(target,ref['t'],ref['theta'][:,arm,parameter])
        delta_theta = arrays['theta'][selected]-interpolated_theta
        theta = dict(applicable=True,common_endpoint_s=float(target[-1]),
            maximum_absolute_parameter_trace_difference=np.max(np.abs(delta_theta),axis=0),
            rms_parameter_trace_difference=np.sqrt(np.mean(delta_theta**2,axis=0)),
            final_common_parameter_difference=delta_theta[-1],
            final_common_parameter_difference_l2_by_arm=np.linalg.norm(delta_theta[-1],axis=1),
            interpretation='Fine estimate minus main estimate interpolated to common fine ticks; all eight parameters and both arms retained.')
    pose_difference = differences['error']
    return dict(reference_directory=str(reference.resolve()),
        common_interval_s=[float(lo),float(hi)], method_matches=summary['method']==ref_summary['method'],
        trace_sensitivity=errors,theta_sensitivity=theta,
        position_orientation_trace_difference=dict(
            position_vector_rms_m=float(np.sqrt(np.mean(np.sum(pose_difference[:,:3]**2,axis=1)))),
            orientation_coordinate_vector_rms_rad=float(np.sqrt(np.mean(np.sum(pose_difference[:,3:]**2,axis=1)))),
            interpretation='RMS norm of the pose-error vector difference; not a difference of error magnitudes.'),
        matched_final_two_seconds_force_statistics=force,
        interpretation='Numerical sensitivity of the sign-driven physical experiment; differences are disclosed rather than imposed as a performance-pass condition.')


def validate_case(directory, trial=None, protocol=None, samples=31, reference=None):
    directory = Path(directory)
    audit = ComparisonAudit(directory)
    try:
        summary = json.loads((directory/'summary.json').read_text())
        manifest = json.loads((directory/'source_manifest.json').read_text())
        audit.report.update(method=summary['method'], scenario=summary['scenario_id'])
        audit.report['input_sha256'] = {name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
            for name in ('trajectory.npz','summary.json','source_manifest.json','learning_state.npz','failure_state.npz')
            if (directory/name).is_file()}
        audit.report['validator_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        verify_sources(audit, manifest)
        audit_online_provenance(audit)
        audit_protocol(audit, summary, trial, protocol)
        model = mujoco.MjModel.from_xml_path(str(HERE/'models/dual_cr7.xml'))
        model.opt.timestep = summary['physics_dt_s']
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4
        with np.load(directory/'trajectory.npz', allow_pickle=False) as source:
            arrays = {key:source[key] for key in source.files}
        n = len(arrays['t'])
        required_raw = set(RAW_FIELDS)
        if summary['method'] == 'external_ppc':
            required_raw.update(EXTERNAL_FIELDS)
        missing = [] if not n else sorted(required_raw-set(arrays))
        audit.check('complete_raw_schema_for_every_saved_command', not missing, missing_channels=missing, saved_ticks=n)
        if missing:
            return audit.finish(directory)
        failure = audit_prefix(audit, arrays, summary, directory, model)
        if not n:
            audit.report['performance'] = dict(run_status=summary['run_status'],
                termination_reason=summary.get('termination_reason'), recorded_samples=0,
                tracking_entry_s=None, interpretation='Initial domain/command failure retained; no fake zero-error state or command.')
            return audit.finish(directory)
        metadata = json.loads((HERE/'models/scene_metadata.json').read_text())
        rebuilt = audit_all_tick_geometry(audit, model, arrays, summary, metadata)
        np.savez_compressed(directory/'comparison_reconstruction.npz', **rebuilt)
        audit_replay(audit, model, arrays, summary, failure, samples)
        audit_control_formulas(audit, arrays, summary, metadata, samples)
        audit_learning(audit, arrays, summary, directory)
        audit_measured_force_and_physics(audit, model, arrays, summary, metadata, min(32,samples))
        audit_metrics(audit, arrays, summary, rebuilt, metadata)
        if reference is not None:
            audit.report['step_sensitivity'] = reference_sensitivity(arrays, summary, Path(reference))
    except Exception as exc:
        audit.check('audit_executed_without_unhandled_artifact_or_schema_error', False,
                    exception=repr(exc), traceback=traceback.format_exc())
    return audit.finish(directory)


def select_trials(root, protocol, case='all', method='all', trial_ids=None):
    if protocol is not None:
        candidates = [(root/row['path'], row) for row in protocol['trials']]
    elif (root/'source_manifest.json').exists():
        candidates = [(root, None)]
    else:
        candidates = [(path.parent, None) for path in sorted(root.rglob('source_manifest.json'))
                      if 'scratch' not in path.relative_to(root).parts]
    selected = []
    for directory, row in candidates:
        if row is not None:
            method_name, case_name = row['method'], row['scenario']
            if trial_ids and row['id'] not in trial_ids:
                continue
        else:
            summary = json.loads((directory/'summary.json').read_text())
            method_name, case_name = summary['method'], summary['scenario_id']
        if method not in ('all', method_name) or case not in ('all', case_name):
            continue
        selected.append((directory, row))
    return selected


def write_aggregate(root, selected, reports, protocol_path, status):
    ordered = [reports[str(directory)] for directory,_ in selected if str(directory) in reports]
    pending = [dict(path=str(directory),trial_id=row['id'] if row else None)
               for directory,row in selected if str(directory) not in reports]
    aggregate = dict(validity_passed=bool(ordered) and not pending and all(r['validity_passed'] for r in ordered),
        watch_status=status,selected_registered_trials=len(selected),audited_trials=len(ordered),
        missing_outcomes=pending,protocol=str(protocol_path),
        check_count=sum(len(r['checks']) for r in ordered),
        failure_count=sum(len(r['failures']) for r in ordered),cases=ordered,
        interpretation='Every selected registered outcome is retained. Validity is formula/record/physics consistency; performance failure is not removed or converted into a validity failure.')
    temporary = root/'comparison_validation_summary.json.tmp'
    temporary.write_text(json.dumps(plain(aggregate),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(root/'comparison_validation_summary.json')
    return aggregate


def watch_and_validate(root, selected, protocol, protocol_path, samples=31,
                       timeout_s=5400.,poll_s=2.,reference=None):
    """Bounded completion-marker watch: exactly one audit per immutable trial.

    A saved report is reused on restart only if all its raw/summary/manifest
    input hashes still match. Fine runs wait for the corresponding main marker
    and always retain reference sensitivity. No physics source/raw is written.
    """
    reports = {}
    started = time.monotonic()
    while True:
        progress = False
        for directory,row in selected:
            key = str(directory)
            if key in reports or not (directory/'source_manifest.json').is_file():
                continue
            case_reference = reference
            if row is not None and row['path'].startswith('step_refinement/'):
                case_reference = reference or root/'main'/row['scenario']/row['method']
                if not (case_reference/'source_manifest.json').is_file():
                    continue
            path = directory/'comparison_validation.json'
            if path.is_file():
                report = json.loads(path.read_text())
                fingerprints = report.get('input_sha256')
                current = {name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
                    for name in ('trajectory.npz','summary.json','source_manifest.json','learning_state.npz','failure_state.npz')
                    if (directory/name).is_file()}
                if fingerprints != current or (case_reference is not None and 'step_sensitivity' not in report):
                    raise RuntimeError(f'Existing audit is stale; refusing silently to repeat trial {directory}.')
                print(f'REUSE {row["id"] if row else directory.name}: identical immutable inputs',flush=True)
            else:
                trial_id = row['id'] if row else directory.name
                print(f'AUDIT {trial_id}: completed source marker found',flush=True)
                report = validate_case(directory,row,protocol,samples,case_reference)
                print(f'{"PASS" if report["validity_passed"] else "FAIL"} {trial_id}: '
                      f'{len(report["checks"])-len(report["failures"])}/{len(report["checks"])} validity; '
                      f'outcome={report.get("performance",{}).get("run_status","unknown")}; '
                      f'failures={report["failures"]}',flush=True)
            reports[key] = report
            progress = True
            write_aggregate(root,selected,reports,protocol_path,'running')
        if len(reports) == len(selected):
            result = write_aggregate(root,selected,reports,protocol_path,'complete')
            print(f'COMPLETE {len(reports)}/{len(selected)} trials; '
                  f'{result["check_count"]-result["failure_count"]}/{result["check_count"]} validity checks; '
                  f'wall={time.monotonic()-started:.1f}s',flush=True)
            return result
        if time.monotonic()-started >= timeout_s:
            result = write_aggregate(root,selected,reports,protocol_path,'timeout')
            print(f'TIMEOUT {len(reports)}/{len(selected)} trials; pending outcomes retained',flush=True)
            return result
        if progress or not reports:
            print(f'WATCH {len(reports)}/{len(selected)} audited; awaiting immutable completion markers',flush=True)
        time.sleep(min(max(.1,poll_s),60.))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--case', default='all')
    parser.add_argument('--method', choices=('all',)+METHODS, default='all')
    parser.add_argument('--trial', action='append')
    parser.add_argument('--samples', type=int, default=31)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--watch-timeout-s',type=float,default=5400.)
    parser.add_argument('--poll-s',type=float,default=2.)
    args = parser.parse_args()
    root = args.results.resolve()
    protocol_path = args.protocol or root/'protocol.json'
    protocol = json.loads(protocol_path.read_text()) if protocol_path.is_file() else None
    selected = select_trials(root, protocol, args.case, args.method, args.trial)
    if args.watch:
        if protocol is None or not selected:
            raise ValueError('A frozen nonempty protocol is required for completion-marker watching.')
        result = watch_and_validate(root,selected,protocol,protocol_path,args.samples,
                                    args.watch_timeout_s,args.poll_s,args.reference)
        if not result['validity_passed']:
            raise SystemExit(1)
        return
    reports, missing = [], []
    for directory, row in selected:
        required = ('trajectory.npz','summary.json','source_manifest.json')
        absent = [name for name in required if not (directory/name).is_file()]
        if absent:
            missing.append(dict(path=str(directory), trial_id=row['id'] if row else None, missing=absent))
            print(f'MISSING {directory}: {absent}', flush=True)
            continue
        reference = args.reference
        if reference is None and row is not None and row['path'].startswith('step_refinement/'):
            candidate = root/'main'/row['scenario']/row['method']
            reference = candidate if (candidate/'source_manifest.json').is_file() else None
        report = validate_case(directory, row, protocol, args.samples, reference)
        reports.append(report)
        print(f'{"PASS" if report["validity_passed"] else "FAIL"} '
              f'{directory.relative_to(root) if directory != root else directory.name}: '
              f'{len(report["checks"])-len(report["failures"])}/{len(report["checks"])} validity checks; '
              f'outcome={report.get("performance",{}).get("run_status","unknown")}', flush=True)
    aggregate = dict(validity_passed=bool(reports) and not missing and all(r['validity_passed'] for r in reports),
        selected_registered_trials=len(selected), audited_trials=len(reports), missing_outcomes=missing,
        protocol=str(protocol_path) if protocol is not None else None,
        check_count=sum(len(r['checks']) for r in reports), failure_count=sum(len(r['failures']) for r in reports),
        cases=reports, interpretation='Validity is formula/record/physics consistency. Retained performance failure does not become a validity failure or disappear from coverage.')
    (root/'comparison_validation_summary.json').write_text(
        json.dumps(plain(aggregate), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    if not aggregate['validity_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
