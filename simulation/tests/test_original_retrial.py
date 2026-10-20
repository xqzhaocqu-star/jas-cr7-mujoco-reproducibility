"""Independent checks of the original projected JAS retrial.

The implicit step is checked as a box variational inequality against an
independent bounded least-squares solver and manufactured KKT solutions.
These are numerical-integration checks of the original two-term ODE,
not a replacement learning law or a constructed parameter trajectory.
"""
from pathlib import Path
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import solve_triangular
from scipy.optimize import lsq_linear

SIMULATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATION))

from jas.controller import JASController, Settings
from jas.dynamics import LinkDynamics
from jas.geometry import LocalKinematics
from jas.learning import CompositeLearner
from jas.reference import desired
from original_retrial.controller import OriginalController, OriginalSettings
from original_retrial.learning import ProjectedCompositeLearner, box_quadratic


NOMINAL_ARM = SIMULATION/'models'/'arm_nominal.xml'


def _bounded_least_squares(hessian, rhs, lower, upper):
    """Independent generic bounded solver for the equivalent quadratic."""
    factor = np.linalg.cholesky(hessian)
    target = solve_triangular(factor, rhs, lower=True)
    result = lsq_linear(factor.T, target, bounds=(lower, upper), method='bvls',
                        tol=1e-13, max_iter=1000)
    assert result.success, result.message
    return result.x


def _assert_box_kkt(hessian, rhs, lower, upper, value, tolerance=2e-9):
    assert np.all(value >= lower-2e-13)
    assert np.all(value <= upper+2e-13)
    gradient = hessian@value-rhs
    at_lower = value <= lower+2e-11
    at_upper = value >= upper-2e-11
    free = ~(at_lower | at_upper)
    assert np.all(gradient[at_lower] >= -tolerance)
    assert np.all(gradient[at_upper] <= tolerance)
    assert np.all(np.abs(gradient[free]) <= tolerance)


def _frozen_learner(cls, gram, p, theta, gain, gamma, lower=.85, upper=1.15):
    """Manufactured full-rank stored-data fixture, independent of filtering."""
    size = len(theta)
    learner = cls(size=size, gain=np.asarray(gain), gamma=gamma,
                  lower=lower, upper=upper)
    zero = np.zeros((size, 6))
    learner.advance(0., zero, zero, np.zeros(6), zero, np.zeros(6))
    learner.theta[:] = theta
    learner.gram = np.asarray(gram).copy()
    learner.p = np.asarray(p).copy()
    learner.frozen_time = 0.
    return learner


@pytest.mark.parametrize('hand', [0, 1])
@pytest.mark.parametrize('nonmonotonic', [False, True])
def test_original_controller_and_explicit_update_are_bitwise_equivalent(hand, nonmonotonic):
    """Same state, reference and estimates must retain every torque term."""
    kin = LocalKinematics(NOMINAL_ARM, [.1, -.2, 0.], np.eye(3),
                          [.02, -.01, .08], np.eye(3))
    dynamics = LinkDynamics(NOMINAL_ARM)
    q = np.array([.35, -.65, .9, .45, -.8, .3])
    velocity = np.array([.23, -.17, .14, .31, -.27, .19])
    center = kin.pose_jacobian(q)[0]
    gains = np.array([.0002, .0004, .0006, .0008, .0003, .0005, .0007, .0009])
    args = (kin, dynamics, center, 2., [.0290666666667, .0154666666667, .0397333333333], hand)
    baseline = JASController(*args, settings=Settings(adaptation_gain=gains, rho_hat=2e-4),
                             nonmonotonic=nonmonotonic)
    retrial = OriginalController(
        *args, settings=OriginalSettings(adaptation_gain=tuple(gains), rho_hat=2e-4,
                                        learning_integrator='explicit_euler'),
        nonmonotonic=nonmonotonic, reference=desired)
    estimate = np.array([1.04, .91, .98, 1.02, 1.06, .89, 1.09, 1.01])
    baseline.learner.theta[:] = estimate
    retrial.learner.theta[:] = estimate
    drive_norms = []
    for time in (0., .04, .09, .17):
        expected_torque, expected = baseline.command(time, q, velocity)
        torque, observed = retrial.command(time, q, velocity)
        assert_array_equal(torque, expected_torque)
        assert set(observed) == set(expected)
        for key in expected:
            assert_array_equal(observed[key], expected[key], err_msg=key)
        drive_norms.append(np.linalg.norm(expected['regressor']@expected['surface']))
        baseline.update(time, expected_torque, expected)
        retrial.update(time, torque, observed)
        assert_array_equal(retrial.learner.theta, baseline.learner.theta)
        assert_array_equal(retrial.learner.gram, baseline.learner.gram)
        assert_array_equal(retrial.learner.p, baseline.learner.p)
        assert_array_equal(retrial.learner.epsilon, baseline.learner.epsilon)
    assert max(drive_norms) > 1e-4, 'The equality must exercise a nonzero -YS term.'
    assert_array_equal(retrial.learner.gain, gains)


def test_explicit_constant_diagonal_gain_matches_projected_paper_rhs():
    gram = np.array([[2., .4, -.2], [.4, 1.4, .3], [-.2, .3, 1.8]])
    old = np.array([.85, .98, 1.15])
    gain = np.array([.03, .07, .12])
    gamma, dt = .6, .025
    regressor = np.array([[.3, .1, 0., 0., 0., 0.],
                          [-.2, .4, 0., 0., 0., 0.],
                          [.5, -.3, 0., 0., 0., 0.]])
    surface = np.array([.6, -.4, 0., 0., 0., 0.])
    direction = np.array([-.8, .2, .7])
    p = gram@old+(direction+regressor@surface)/gamma
    learner = _frozen_learner(CompositeLearner, gram, p, old, gain, gamma)
    zero = np.zeros((3, 6))
    learner.advance(dt, zero, zero, np.zeros(6), regressor, surface)
    expected = old+dt*gain*np.array([0., direction[1], 0.])
    assert_allclose(learner.theta, expected, rtol=0, atol=2e-15)
    assert_array_equal(learner.gain, gain)


@pytest.mark.parametrize('size', [2, 5, 8])
def test_box_quadratic_matches_independent_scipy_bounded_solver(size):
    rng = np.random.default_rng(29341+size)
    lower, upper = np.full(size, .85), np.full(size, 1.15)
    for _ in range(4):
        matrix = rng.normal(size=(size, size))
        hessian = matrix.T@matrix+.3*np.eye(size)
        rhs = hessian@rng.uniform(.5, 1.5, size)
        expected = _bounded_least_squares(hessian, rhs, lower, upper)
        for start in (lower, upper, rng.uniform(lower, upper)):
            actual = box_quadratic(hessian, rhs, lower, upper, start)
            assert_allclose(actual, expected, rtol=2e-11, atol=2e-11)
            _assert_box_kkt(hessian, rhs, lower, upper, actual)


def test_box_quadratic_recovers_manufactured_eight_parameter_kkt_solution():
    rng = np.random.default_rng(6972)
    matrix = rng.normal(size=(8, 8))
    hessian = matrix.T@matrix+.6*np.eye(8)
    lower, upper = np.full(8, .85), np.full(8, 1.15)
    expected = np.array([.85, 1.15, .96, 1.03, .85, 1.15, .91, 1.12])
    # For a convex box QP these signed face gradients certify the optimum.
    gradient = np.array([1.2, -.7, 0., 0., .8, -1.8, 0., 0.])
    rhs = hessian@expected-gradient
    for start in (lower, upper, np.ones(8)):
        actual = box_quadratic(hessian, rhs, lower, upper, start)
        assert_allclose(actual, expected, rtol=0, atol=2e-12)
        _assert_box_kkt(hessian, rhs, lower, upper, actual)


def test_box_kkt_release_is_not_hidden_by_another_components_large_rhs():
    """Global rhs tolerance must not strand a weak coordinate on a face."""
    hessian = np.diag([1e12, 1.])
    rhs = np.array([9e11, .95])
    lower, upper = np.full(2, .85), np.full(2, 1.15)
    actual = box_quadratic(hessian, rhs, lower, upper, np.array([.9, .85]))
    assert_allclose(actual, [.9, .95], rtol=0, atol=2e-13)
    _assert_box_kkt(hessian, rhs, lower, upper, actual, tolerance=1e-10)


def test_implicit_step_retains_nonzero_tracking_drive_and_original_prediction():
    gram = np.array([[2., .4, -.2], [.4, 1.4, .3], [-.2, .3, 1.8]])
    old = np.array([.96, 1.02, .93])
    target = np.array([1.05, .98, 1.10])
    p, gains, gamma, dt = gram@target, np.array([.2, .7, .4]), .8, .03
    regressor = np.zeros((3, 6))
    regressor[:, :3] = np.array([[.3, .1, -.2], [-.2, .4, .3], [.5, -.3, .1]])
    surface = np.array([.6, -.4, .3, 0., 0., 0.])
    drive = regressor@surface
    learner = _frozen_learner(ProjectedCompositeLearner, gram, p, old, gains, gamma)
    zero = np.zeros_like(regressor)
    learner.advance(dt, zero, zero, np.zeros(6), regressor, surface)
    diagonal = 1/(dt*gains)
    expected = np.linalg.solve(np.diag(diagonal)+gamma*gram,
                               diagonal*old+gamma*p-drive)
    discarded_tracking = np.linalg.solve(np.diag(diagonal)+gamma*gram,
                                         diagonal*old+gamma*p)
    assert np.all((expected > .85) & (expected < 1.15))
    assert_allclose(learner.theta, expected, rtol=0, atol=4e-15)
    assert np.linalg.norm(learner.theta-discarded_tracking) > 1e-3
    assert_array_equal(learner.last_drive, drive)
    assert_array_equal(learner.epsilon, p-gram@old)
    assert learner.last_fixed_point_error < 2e-14


def test_implicit_coupled_gram_solves_box_vi_instead_of_clipping_unconstrained_solve():
    # The unconstrained optimum is (-.2,.4); its clipped value (0,.4)
    # has a nonzero free gradient and is not the constrained optimum (0,.26).
    gram = np.array([[1.5, 1.4], [1.4, 1.5]])
    old, gains, gamma, dt = np.array([.3, .8]), np.array([2., 2.]), 1., 1.
    hessian = np.diag(1/(dt*gains))+gram
    rhs = hessian@np.array([-.2, .4])
    regressor = np.zeros((2, 6)); regressor[:, 0] = [.08, -.04]
    surface = np.array([1., 0., 0., 0., 0., 0.])
    p = rhs-old/(dt*gains)+regressor@surface
    learner = _frozen_learner(ProjectedCompositeLearner, gram, p, old, gains,
                               gamma, lower=0., upper=1.)
    zero = np.zeros_like(regressor)
    learner.advance(dt, zero, zero, np.zeros(6), regressor, surface)
    expected = _bounded_least_squares(hessian, rhs, np.zeros(2), np.ones(2))
    assert_allclose(expected, [0., .26], rtol=0, atol=2e-14)
    assert_allclose(learner.theta, expected, rtol=0, atol=2e-14)
    wrong = np.clip(np.linalg.solve(hessian, rhs), 0., 1.)
    assert np.linalg.norm(wrong-learner.theta) > .13
    _assert_box_kkt(hessian, rhs, np.zeros(2), np.ones(2), learner.theta)
    assert learner.last_fixed_point_error < 2e-14
    assert learner.active_projection_samples == 1


def test_implicit_step_converges_to_original_two_term_ode_as_step_tends_to_zero():
    gram = np.array([[2., .4, -.2], [.4, 1.4, .3], [-.2, .3, 1.8]])
    old, gains, gamma = np.array([.96, 1.02, .93]), np.array([.2, .7, .4]), .8
    p = gram@np.array([1.05, .98, 1.10])
    regressor = np.zeros((3, 6)); regressor[:, 0] = [.13, -.21, .17]
    surface = np.array([1., 0., 0., 0., 0., 0.])
    derivative = gains*(-regressor@surface+gamma*(p-gram@old))
    errors = []
    for dt in (.01, .005, .0025, .00125):
        learner = _frozen_learner(ProjectedCompositeLearner, gram, p, old, gains, gamma)
        zero = np.zeros_like(regressor)
        learner.advance(dt, zero, zero, np.zeros(6), regressor, surface)
        errors.append(np.linalg.norm((learner.theta-old)/dt-derivative))
    ratios = np.asarray(errors[:-1])/errors[1:]
    assert np.all((ratios > 1.9) & (ratios < 2.05)), errors
    assert errors[-1] < .14*errors[0]


@pytest.mark.parametrize('omega', [1e-6, 1e6])
def test_implicit_filter_sliding_history_and_ie_are_identical_to_original(omega):
    """Only the ODE integrator changes, including input left/right jumps."""
    rng = np.random.default_rng(9017)
    m0, mdot = .1*rng.normal(size=(3, 6)), .04*rng.normal(size=(3, 6))
    remainder0 = np.column_stack((np.eye(3), .1*rng.normal(size=(3, 3))))
    remainder_rate = .03*rng.normal(size=(3, 6))
    gain = np.array([.01, .015, .02])
    baseline = CompositeLearner(size=3, gain=gain, gamma=.7, filter_rate=4.,
                                window=.16, omega=omega)
    implicit = ProjectedCompositeLearner(size=3, gain=gain, gamma=.7,
                                          filter_rate=4., window=.16, omega=omega)
    times = np.linspace(0., .6, 31)
    commands = [(-1.)**j*np.array([.2+.03*j, -.4, .1, .3, -.2, .15])
                for j in range(len(times))]
    frozen = None
    for j, time in enumerate(times):
        left = commands[max(j-1, 0)]*(1+.08*time)
        right = commands[j]*(1+.08*time)
        momentum = m0+time*mdot
        remainder = remainder0+time*remainder_rate
        regressor = remainder0+.1*np.sin(time)*remainder_rate
        surface = np.array([.13, -.08, .04, -.02, .09, -.06])
        for learner in (baseline, implicit):
            learner.advance(float(time), momentum, remainder, left, regressor, surface,
                            next_known_torque=right)
        for field in ('z', 'tauf', 'filtered_regressor', 'gram', 'p', 'minimum_eigenvalue'):
            assert_array_equal(getattr(implicit, field), getattr(baseline, field), err_msg=field)
        assert implicit.frozen_time == baseline.frozen_time
        assert_array_equal(implicit.previous[2], right)
        assert_array_equal(implicit.gain, gain)
        if implicit.frozen_time is not None:
            if frozen is None:
                frozen = (implicit.gram.copy(), implicit.p.copy())
            assert_array_equal(implicit.gram, frozen[0])
            assert_array_equal(implicit.p, frozen[1])
    if omega < 1:
        assert implicit.frozen_time == pytest.approx(.16)
        assert frozen is not None
    else:
        assert implicit.frozen_time is None
        assert len(implicit.history) < len(times)-1, 'The unfrozen window must remove old intervals.'


def test_projected_implicit_small_step_keeps_outward_face_derivative_zero():
    gram = np.array([[2., .5], [.5, 1.]])
    old, gains, gamma = np.array([.85, 1.]), np.array([.4, .7]), .6
    # The first raw derivative points outward; projection must remove it.
    regressor = np.zeros((2, 6)); regressor[:, 0] = [.3, -.2]
    surface = np.array([1., 0., 0., 0., 0., 0.])
    p = gram@old+np.array([-.2, .1])
    derivative = gains*np.array([0., .2+gamma*.1])
    errors = []
    for dt in (.01, .005, .0025):
        learner = _frozen_learner(ProjectedCompositeLearner, gram, p, old, gains, gamma)
        zero = np.zeros_like(regressor)
        learner.advance(dt, zero, zero, np.zeros(6), regressor, surface)
        assert learner.theta[0] == .85
        errors.append(np.linalg.norm((learner.theta-old)/dt-derivative))
    assert 1.9 < errors[0]/errors[1] < 2.05
    assert 1.9 < errors[1]/errors[2] < 2.05
