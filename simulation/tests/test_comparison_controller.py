"""Independent mathematical checks of the comparative control implementation.

No physical rollout is performed. Real CR7 model/kinematic evaluations check
the original torque oracle; directional differences check the new BF chain
rule. Manufactured stored data test removal of only the history drive.
"""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import solve_ivp

SIMULATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATION))

from comparison.controller import (
    ComparisonController, ComparisonSettings, common_bounds,
    conventional_log_barrier, log_scale,
)
from comparison.mechanism import scalar_entry_times
from jas.barrier import transform
from jas.dynamics import LinkDynamics
from jas.geometry import LocalKinematics
from jas.reference import envelope
from original_retrial.controller import OriginalController
import original_retrial.controller as original_module


MODEL = SIMULATION/'models'/'arm_nominal.xml'
Q = np.array([.35, -.65, .9, .45, -.8, .3])
QD = np.array([.23, -.17, .14, .31, -.27, .19])
ESTIMATE = np.array([1.04, .91, .98, 1.02, 1.06, .89, 1.09, 1.01])
GAINS = (.0002, .0004, .0006, .0008, .0003, .0005, .0007, .0009)


def _reference(t, center):
    # A nonstationary, exactly differentiated reference which remains in the
    # finite comparison domain at every state used below.
    amplitude = np.array([.002, .0015, .001, .006, .004, .005])
    frequency = np.array([.7, 1.1, .8, .6, .9, 1.2])
    return (center+amplitude*np.sin(frequency*t),
            amplitude*frequency*np.cos(frequency*t),
            -amplitude*frequency**2*np.sin(frequency*t))


@pytest.fixture(scope='module')
def real_context():
    kin = LocalKinematics(MODEL, [.1, -.2, 0.], np.eye(3),
                          [.02, -.01, .08], np.eye(3))
    dynamics = LinkDynamics(MODEL)
    center = kin.pose_jacobian(Q)[0]-np.array([.004, -.003, .002, .008, -.006, .005])
    return kin, dynamics, center


def _settings(**overrides):
    settings = ComparisonSettings(
        learning_integrator='explicit_euler', adaptation_gain=GAINS,
        rho_hat=2e-4,
        linear_surface_gain=(5., 6., 7., 3., 4., 2.))
    return replace(settings, **overrides)


def _controller(context, settings, hand=0, nonmonotonic=True, cls=ComparisonController,
                reference=_reference, center=None):
    kin, dynamics, default_center = context
    controller = cls(kin, dynamics, default_center if center is None else center,
                     2., [.0290666666667, .0154666666667, .0397333333333], hand,
                     settings=settings, nonmonotonic=nonmonotonic, reference=reference)
    controller.learner.theta[:] = ESTIMATE
    return controller


def _same_info(actual, expected, bitwise=False):
    assert actual.keys() == expected.keys()
    for key in expected:
        if bitwise:
            assert_array_equal(actual[key], expected[key], err_msg=key)
        else:
            assert_allclose(actual[key], expected[key], rtol=4e-13, atol=2e-11,
                            err_msg=key)


@pytest.mark.parametrize('hand', [0, 1])
@pytest.mark.parametrize('nonmonotonic', [False, True])
def test_proposed_command_and_update_are_bitwise_original(real_context, hand, nonmonotonic):
    settings = _settings(method='proposed')
    original = _controller(real_context, settings, hand, nonmonotonic, cls=OriginalController)
    proposed = _controller(real_context, settings, hand, nonmonotonic)
    drives = []
    for t in (0., .04, .11, .23):
        expected_torque, expected = original.command(t, Q, QD)
        torque, observed = proposed.command(t, Q, QD)
        assert_array_equal(torque, expected_torque)
        _same_info(observed, expected, bitwise=True)
        drives.append(np.linalg.norm(expected['regressor']@expected['surface']))
        original.update(t, expected_torque, expected)
        proposed.update(t, torque, observed)
        for field in ('theta', 'gram', 'p', 'epsilon', 'z', 'tauf', 'filtered_regressor'):
            assert_array_equal(getattr(proposed.learner, field), getattr(original.learner, field),
                               err_msg=field)
    assert max(drives) > 1e-4


@pytest.mark.parametrize('method', ['traditional_bf', 'linear_surface', 'linear_reaching'])
def test_structural_change_matches_original_torque_with_only_declared_block_replaced(
        real_context, monkeypatch, method):
    settings = _settings(method=method)
    tested = _controller(real_context, settings)
    oracle_settings = settings
    if method == 'linear_reaching':
        # Existing Ka/rho are the retained linear dissipation. No extra
        # linear gain or softened sign is present in this ablation.
        oracle_settings = replace(settings, k2=(0.,)*6, k3=(0.,)*6)
    oracle = _controller(real_context, oracle_settings, cls=OriginalController)
    for t in (.2, 1.3, 4.2):
        with monkeypatch.context() as patch:
            if method == 'traditional_bf':
                patch.setattr(original_module, 'transform',
                              lambda e, ed, unused_bounds, unused_chi, unused_ell:
                              conventional_log_barrier(e, ed, common_bounds(t, settings, True),
                                                       log_scale(settings)))
            elif method == 'linear_surface':
                gain = np.asarray(settings.linear_surface_gain)
                patch.setattr(original_module, 'terminal_surface',
                              lambda xi, xidot, *unused: (xidot+gain*xi, -gain*xi, -gain*xidot))
            expected_torque, expected = oracle.command(t, Q, QD)
        torque, observed = tested.command(t, Q, QD)
        assert_allclose(torque, expected_torque, rtol=4e-13, atol=2e-11)
        _same_info(observed, expected)
        # Both changes to reference dynamics must propagate to the data
        # channels, rather than changing only the plotted surface.
        baseline = _controller(real_context, _settings(method='proposed'))
        base_torque, base_info = baseline.command(t, Q, QD)
        assert np.linalg.norm(torque-base_torque) > 1e-6
        if method == 'linear_reaching':
            assert_array_equal(observed['regressor'], base_info['regressor'])
            assert_array_equal(observed['momentum'], base_info['momentum'])
        elif method == 'linear_surface':
            assert np.linalg.norm(observed['regressor']-base_info['regressor']) > 1e-6
            assert_array_equal(observed['A'], base_info['A'])
        else:
            assert np.linalg.norm(observed['A']-base_info['A']) > 1e-6
            assert np.linalg.norm(observed['momentum']-base_info['momentum']) > 1e-6


def test_linear_reaching_torque_difference_is_exactly_removed_double_power_feedback(real_context):
    settings = _settings(method='proposed')
    full = _controller(real_context, settings)
    linear = _controller(real_context, replace(settings, method='linear_reaching'))
    for t in (.2, 4.2):
        full_torque, info = full.command(t, Q, QD)
        linear_torque, linear_info = linear.command(t, Q, QD)
        barrier = transform(info['error'], info['xd']-_reference(t, full.center)[1],
                            envelope(t, np.asarray(settings.chi), np.asarray(settings.ell), True),
                            np.asarray(settings.chi), np.asarray(settings.ell))
        s = info['surface']
        removed = (np.asarray(settings.k2)*np.sign(s)*np.abs(s)**settings.upsilon1
                   +np.asarray(settings.k3)*np.sign(s)*np.abs(s)**settings.upsilon2)
        expected_delta = info['jacobian'].T@(barrier.a*removed)
        assert_allclose(linear_torque-full_torque, expected_delta, rtol=3e-12, atol=3e-12)
        for field in ('surface', 'xi', 'regressor', 'A', 'momentum', 'remainder', 'rho', 'robust_power'):
            assert_array_equal(linear_info[field], info[field], err_msg=field)


@pytest.mark.parametrize('method', ['proposed', 'traditional_bf', 'linear_surface',
                                   'linear_reaching', 'no_history'])
def test_all_internal_methods_keep_full_prior_sign_and_positive_rho(real_context, method):
    settings = _settings(method=method, prior_lower=.82, prior_upper=1.21, rho_hat=.003)
    # Make YS very small and nonzero: a hidden boundary layer would differ
    # greatly from the stipulated sign even though tracking is near zero.
    pose = real_context[0].pose_jacobian(Q)[0]
    center = pose-np.array([1., -2., 3., -1., 2., -3.])*1e-9
    hold = lambda t, c: (np.asarray(c), np.zeros(6), np.zeros(6))
    controller = _controller(real_context, settings, reference=hold, center=center)
    torque, info = controller.command(0., Q, QD*1e-8)
    j, a = info['jacobian'], info['A']
    inverse_diagonal = j@a
    assert_allclose(inverse_diagonal, np.diag(np.diag(inverse_diagonal)), atol=2e-14)
    barrier_a = 1/np.diag(inverse_diagonal)
    s, y = info['surface'], info['regressor']
    w = y@s
    width = settings.prior_upper-settings.prior_lower
    robust = y.T@(width*np.sign(w))
    rho = settings.rho_hat*np.linalg.norm(a, ord=2)**2
    expected_task = y.T@ESTIMATE-(np.asarray(settings.ka)+rho)*s-robust
    if method != 'linear_reaching':
        expected_task -= (np.asarray(settings.k2)*np.sign(s)*np.abs(s)**settings.upsilon1
                          +np.asarray(settings.k3)*np.sign(s)*np.abs(s)**settings.upsilon2)
    assert info['rho'] == rho and rho > 0
    assert_allclose(torque, j.T@(barrier_a*expected_task-controller.fd),
                    rtol=5e-12, atol=5e-11)
    assert_allclose(info['robust_power'], width*np.sum(np.abs(w)), rtol=3e-15, atol=1e-20)
    assert np.any((np.abs(w) > 0) & (np.abs(w) < settings.robust_width))
    softened = y.T@(width*np.clip(w/settings.robust_width, -1., 1.))
    assert np.linalg.norm(j.T@(barrier_a*(robust-softened))) > 1e-3


def test_full_and_no_history_have_bitwise_identical_command_for_same_estimate(real_context):
    full = _controller(real_context, _settings(method='proposed'))
    no_history = _controller(real_context, _settings(method='no_history'))
    assert full.learner.gamma > 0 and no_history.learner.gamma == 0
    for t in (0., .2, 4.2):
        torque, info = full.command(t, Q, QD)
        other_torque, other_info = no_history.command(t, Q, QD)
        assert_array_equal(other_torque, torque)
        _same_info(other_info, info, bitwise=True)


def test_no_history_keeps_nonzero_tracking_drive_and_face_projection(real_context):
    controller = _controller(real_context, _settings(method='no_history'))
    learner = controller.learner
    zero = np.zeros((8, 6))
    learner.advance(0., zero, zero, np.zeros(6), zero, np.zeros(6))
    old = np.array([.85, 1.15, .93, 1.02, 1., .94, .85, 1.15])
    direction = np.array([-.6, .7, -.4, .2, .9, -.3, .25, -.1])
    regressor = np.zeros((8, 6)); regressor[:, 0] = -direction
    surface = np.array([1., 0., 0., 0., 0., 0.])
    learner.theta[:] = old
    learner.frozen_time = 0.
    learner.gram = np.eye(8)+.1*np.ones((8, 8))
    learner.p = learner.gram@old+np.linspace(10., 80., 8)
    dt = .02
    learner.advance(dt, zero, zero, np.zeros(6), regressor, surface)
    projected = direction.copy(); projected[:2] = 0.
    assert_allclose(learner.theta, old+dt*np.asarray(GAINS)*projected, rtol=0, atol=2e-16)
    assert_allclose(learner.epsilon, np.linspace(10., 80., 8), rtol=0, atol=2e-14)
    assert learner.theta[0] == old[0] and learner.theta[1] == old[1]
    assert learner.theta[6] > old[6] and learner.theta[7] < old[7]
    assert np.linalg.norm(learner.theta-old) > 1e-6


@pytest.mark.parametrize('nonmonotonic', [False, True])
@pytest.mark.parametrize('t', [0., .12, .4, 1., 1.7, 3., 4.2, 7., 12.])
def test_common_bounds_first_two_derivatives_and_finite_initialization(t, nonmonotonic):
    settings = _settings()
    current = common_bounds(t, settings, nonmonotonic)
    h = 1e-6
    plus = common_bounds(t+h, settings, nonmonotonic)
    minus = common_bounds(t-h if t else t+2*h, settings, nonmonotonic)
    for value, first, second in ((0, 2, 4), (1, 3, 5)):
        d1 = ((plus[value]-minus[value])/(2*h) if t else
              (-3*current[value]+4*plus[value]-minus[value])/(2*h))
        d2 = ((plus[first]-minus[first])/(2*h) if t else
              (-3*current[first]+4*plus[first]-minus[first])/(2*h))
        assert_allclose(d1, current[first], rtol=8e-7, atol=3e-9)
        assert_allclose(d2, current[second], rtol=2e-5, atol=3e-7)
        assert np.all(np.isfinite(current[value])) and np.all(current[value] > 0)
    if t == 0:
        assert_allclose(current[0], settings.evaluation_initial_upper, rtol=0, atol=3e-16)
        assert_allclose(current[1], settings.evaluation_initial_lower, rtol=0, atol=3e-16)
    if t >= settings.evaluation_transition_s:
        mu = envelope(t, np.asarray(settings.chi), np.asarray(settings.ell), nonmonotonic)
        assert_allclose(current[0], np.arctanh(mu[0]/settings.ell)/settings.chi,
                        rtol=2e-15, atol=0)
        assert_allclose(current[1], np.arctanh(mu[1]/settings.ell)/settings.chi,
                        rtol=2e-15, atol=0)


@pytest.mark.parametrize('transition', [.7, 1., 1.4])
def test_common_bounds_are_c2_at_configured_transition(transition):
    settings = _settings(evaluation_transition_s=transition)
    h = 1e-8
    left, at, right = [common_bounds(transition+dt, settings, True) for dt in (-h, 0., h)]
    for field in range(6):
        assert_allclose(left[field], at[field], rtol=0, atol=2e-7)
        assert_allclose(right[field], at[field], rtol=0, atol=2e-7)


@pytest.mark.parametrize('t', [0., .19, .65, 1., 2.1, 4.2, 6.6, 12.])
def test_log_bf_directional_derivatives_include_error_and_time(t):
    settings = _settings()
    bounds = common_bounds(t, settings, True)
    fraction = np.array([.47, -.38, .31, -.43, .29, -.34])
    error = np.where(fraction >= 0, fraction*bounds[0], fraction*bounds[1])
    velocity = np.array([.012, -.008, .009, -.035, .022, .028])
    scale = log_scale(settings)
    state = conventional_log_barrier(error, velocity, bounds, scale)
    # Independent value expression rather than reusing log1p implementation.
    expected_xi = scale*(np.log((error+bounds[1])/(bounds[0]-error))
                         -np.log(bounds[1]/bounds[0]))
    assert_allclose(state.xi, expected_xi, rtol=3e-15, atol=3e-17)
    h = 5e-7
    plus = conventional_log_barrier(error+h*velocity, velocity,
                                    common_bounds(t+h, settings, True), scale)
    minus = conventional_log_barrier(error-h*velocity, velocity,
                                     common_bounds(t-h, settings, True), scale) if t else (
        conventional_log_barrier(error+2*h*velocity, velocity,
                                  common_bounds(t+2*h, settings, True), scale))
    for field, expected in (('xi', state.xidot), ('a', state.adot), ('b', state.bdot)):
        numerical = ((getattr(plus, field)-getattr(minus, field))/(2*h) if t else
                     (-3*getattr(state, field)+4*getattr(plus, field)-getattr(minus, field))/(2*h))
        assert_allclose(numerical, expected, rtol=3e-6, atol=2e-8, err_msg=field)
    # Check xi_e=a independently; the directional xi check alone could hide
    # compensating errors between a and b.
    plus_e = conventional_log_barrier(error+h, velocity, bounds, scale)
    minus_e = conventional_log_barrier(error-h, velocity, bounds, scale)
    assert_allclose((plus_e.xi-minus_e.xi)/(2*h), state.a, rtol=2e-8, atol=1e-9)
    assert np.all(state.a > 0) and np.all(state.margin > 0)


def test_log_bf_origin_scale_and_strict_domain():
    settings = _settings()
    bounds = common_bounds(0., settings)
    scale = log_scale(settings)
    zero = np.zeros(6)
    state = conventional_log_barrier(zero, zero, bounds, scale)
    assert_array_equal(state.xi, zero)
    assert_allclose(state.b, zero, rtol=0, atol=1e-15)
    assert_allclose(state.a, np.asarray(settings.chi)/settings.ell, rtol=2e-15, atol=1e-16)
    assert_array_equal(log_scale(replace(settings, log_bf_scale=(.1,)*6)), [.1]*6)
    for illegal in (bounds[0], -bounds[1], 1.01*bounds[0], -1.01*bounds[1]):
        with pytest.raises(FloatingPointError, match='domain violated'):
            conventional_log_barrier(illegal, zero, bounds, scale)


def test_nonuniform_six_axis_linear_surface_gain_is_used_in_all_reference_channels(real_context):
    settings = _settings(method='linear_surface', linear_surface_gain=(1., 2., 3., 4., 5., 6.))
    controller = _controller(real_context, settings)
    t = .3
    _, info = controller.command(t, Q, QD)
    ref, dref, _ = _reference(t, controller.center)
    state = transform(info['x']-ref, info['xd']-dref,
                      envelope(t, np.asarray(settings.chi), np.asarray(settings.ell), True),
                      np.asarray(settings.chi), np.asarray(settings.ell))
    expected_surface = state.xidot+np.asarray(settings.linear_surface_gain)*state.xi
    assert_allclose(info['surface'], expected_surface, rtol=0, atol=2e-15)
    assert np.ptp(np.asarray(settings.linear_surface_gain)) > 0


@pytest.mark.parametrize('gain', [(1.,), (1.,)*5, (1.,)*7,
                                  (1., 2., 3., 0., 5., 6.),
                                  (1., 2., 3., 4., -5., 6.)])
def test_linear_surface_requires_a_strict_positive_six_vector(real_context, gain):
    with pytest.raises(ValueError, match='Positive comparison constants'):
        _controller(real_context, _settings(method='linear_surface', linear_surface_gain=gain))


def _independent_ode_entry(rhs, radius, threshold, maximum_time):
    def event(t, y):
        return y[0]-threshold
    event.direction = -1
    event.terminal = True
    result = solve_ivp(lambda t, y: [rhs(float(y[0]))], (0., maximum_time), [radius],
                       events=event, method='DOP853', rtol=2e-11, atol=2e-13,
                       max_step=.02)
    assert result.success and len(result.t_events[0]) == 1
    return float(result.t_events[0][0])


def test_scalar_sliding_quadrature_matches_closed_form_and_independent_ode():
    settings = _settings(lam1=.8, lam2=1.2, alpha=.6, beta=1.4, delta=.0007)
    data = scalar_entry_times(settings)
    radius, delta = np.asarray(data['initial_magnitude']), settings.delta
    nu = 1-settings.alpha
    scale = np.sqrt(settings.lam2/settings.lam1)
    exact = np.maximum(0., (np.arctan(scale*radius**nu)-np.arctan(scale*delta**nu))
                       /(nu*np.sqrt(settings.lam1*settings.lam2)))
    assert_allclose(data['sliding_dual_power_entry_s'], exact, rtol=4e-11, atol=2e-11)
    linear = max(0., 1.)/settings.linear_surface_gain[0]
    assert_allclose(data['sliding_linear_entry_s'],
                    np.maximum(0., np.log(radius/delta))*linear, rtol=2e-15, atol=2e-15)
    bound = 1/(settings.lam1*(1-settings.alpha))+1/(settings.lam2*(settings.beta-1))
    assert data['sliding_uniform_comparison_bound_s'] == bound
    assert np.max(exact) < bound and np.all(np.diff(exact) >= 0)
    # Solve the differential equation independently of the time quadrature.
    r0 = float(radius[12])
    rhs = lambda r: (-settings.lam1*np.sign(r)*abs(r)**settings.alpha
                     -settings.lam2*np.sign(r)*abs(r)**settings.beta)
    observed = _independent_ode_entry(rhs, r0, delta, bound+1.)
    assert_allclose(observed, exact[12], rtol=2e-9, atol=2e-10)
    assert radius[-1] > 1e15, 'The scalar illustration must actually test large magnitudes.'


def test_scalar_reaching_retains_linear_term_and_respects_uniform_bound():
    # For exponents 3/2 and 1/2, y=sqrt(r) yields a quadratic denominator;
    # its analytic integral detects accidental omission of the Ka term.
    settings = _settings(ka=(1.,)*6, k2=(2.,)*6, k3=(3.,)*6,
                         upsilon1=1.5, upsilon2=.5)
    data = scalar_entry_times(settings)
    radius, eps = np.asarray(data['initial_magnitude']), data['reaching_layer']
    ka, k2, k3 = settings.ka[0], settings.k2[0], settings.k3[0]
    discriminant_root = np.sqrt(4*k2*k3-ka**2)
    primitive = lambda r: 4/discriminant_root*np.arctan((2*k2*np.sqrt(r)+ka)/discriminant_root)
    exact = np.maximum(0., primitive(radius)-primitive(eps))
    assert_allclose(data['reaching_dual_power_entry_s'], exact, rtol=3e-11, atol=2e-11)
    assert_allclose(data['reaching_linear_entry_s'], np.maximum(0., np.log(radius/eps))/ka,
                    rtol=2e-15, atol=2e-15)
    bound = 1/(k2*(settings.upsilon1-1))+1/(k3*(1-settings.upsilon2))
    assert data['reaching_uniform_comparison_bound_s'] == bound
    assert np.max(exact) < bound and np.all(np.diff(exact) >= 0)
    r0 = float(radius[14])
    rhs = lambda r: (-ka*r-k2*np.sign(r)*abs(r)**settings.upsilon1
                     -k3*np.sign(r)*abs(r)**settings.upsilon2)
    observed = _independent_ode_entry(rhs, r0, eps, bound+1.)
    assert_allclose(observed, exact[14], rtol=2e-9, atol=2e-10)
