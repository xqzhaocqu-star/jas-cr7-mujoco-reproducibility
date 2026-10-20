"""Independent numerical checks of the revised JAS equations and model bases.

These tests use finite differences, MuJoCo's full rigid-body dynamics, and
manufactured analytic filter inputs as independent references. They do not
reproduce the existing paper's simulation section.
"""
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

SIMULATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATION))

from jas.barrier import terminal_surface, transform
from jas.dynamics import LinkDynamics, coupled_bases, object_dynamics
from jas.geometry import LocalKinematics
from jas.learning import CompositeLearner
from jas.reference import desired, envelope
from jas.controller import JASController, Settings


CHI = np.array([20., 18., 22., 8., 7., 9.])
ELL = np.array([50., 45., 55., 20., 18., 22.])
MODEL = SIMULATION / 'models' / 'arm_nominal.xml'


@pytest.mark.parametrize('nonmonotonic', [False, True])
@pytest.mark.parametrize('t', [0., .2, 2., 3., 3.8, 4.5, 5.3, 7., 12.])
def test_envelope_derivatives_and_domain(t, nonmonotonic):
    step = 2e-6
    current = envelope(t, CHI, ELL, nonmonotonic)
    plus = envelope(t+step, CHI, ELL, nonmonotonic)
    # The admissible envelope starts at t=0; use a second-order right
    # derivative there rather than evaluating an invalid negative time.
    minus = (envelope(t-step, CHI, ELL, nonmonotonic) if t > 0
             else envelope(t+2*step, CHI, ELL, nonmonotonic))
    for value, first, second in ((0, 2, 4), (1, 3, 5)):
        derivative = ((plus[value]-minus[value])/(2*step) if t > 0 else
                      (-3*current[value]+4*plus[value]-minus[value])/(2*step))
        second_derivative = ((plus[first]-minus[first])/(2*step) if t > 0 else
                             (-3*current[first]+4*plus[first]-minus[first])/(2*step))
        assert_allclose(derivative, current[first],
                        rtol=2e-7, atol=3e-6)
        assert_allclose(second_derivative, current[second],
                        rtol=3e-6, atol=4e-5)
        assert np.all(current[value] > 0)
        assert np.all(current[value] <= ELL*(1+1e-13))
    if t == 0:
        assert_allclose(current[0], ELL, rtol=0, atol=0)
        assert_allclose(current[1], ELL, rtol=0, atol=0)


def test_envelope_is_asymmetric_and_relaxes_smoothly():
    samples = np.array([envelope(t, CHI, ELL, True)
                        for t in np.linspace(0., 20., 801)])
    assert np.max(np.abs(samples[-1, 0]-samples[-1, 1])) > .1
    assert np.any(samples[:, 2] > 0), 'Upper boundary must actually relax.'
    assert np.any(samples[:, 3] > 0), 'Lower boundary must actually relax.'
    for endpoint in (3., 7.):
        left = envelope(endpoint-1e-7, CHI, ELL, True)
        right = envelope(endpoint+1e-7, CHI, ELL, True)
        for derivative_order in range(6):
            assert_allclose(left[derivative_order], right[derivative_order],
                            rtol=0, atol=4e-5)


@pytest.mark.parametrize('t', [0., .7, 3.5, 4.6, 6.4, 12.])
def test_barrier_time_and_error_directional_derivatives(t):
    bounds = envelope(t, CHI, ELL, True)
    # Start in a nontrivial, independently selected interior point.
    fraction = np.array([.36, -.28, .19, -.32, .24, -.21])
    eta = np.where(fraction >= 0, fraction*bounds[0], fraction*bounds[1])
    error = np.arctanh(eta/ELL)/CHI
    velocity = np.array([.012, -.008, .009, -.035, .022, .028])
    current = transform(error, velocity, bounds, CHI, ELL)
    step = 1e-6
    plus = transform(error+step*velocity, velocity,
                     envelope(t+step, CHI, ELL, True), CHI, ELL)
    minus = (transform(error-step*velocity, velocity,
                       envelope(t-step, CHI, ELL, True), CHI, ELL) if t > 0
             else transform(error+2*step*velocity, velocity,
                            envelope(t+2*step, CHI, ELL, True), CHI, ELL))

    def differentiate(name):
        if t > 0:
            return (getattr(plus, name)-getattr(minus, name))/(2*step)
        return (-3*getattr(current, name)+4*getattr(plus, name)
                -getattr(minus, name))/(2*step)

    assert_allclose(differentiate('xi'), current.xidot,
                    rtol=2e-7, atol=2e-9)
    assert_allclose(differentiate('a'), current.adot,
                    rtol=2e-7, atol=2e-8)
    assert_allclose(differentiate('b'), current.bdot,
                    rtol=2e-7, atol=2e-9)
    assert np.all(current.a >= CHI/ELL*(1-1e-13))
    assert np.all(current.margin > 0)
    assert np.all(current.lower < error)
    assert np.all(error < current.upper)


def test_barrier_accepts_finite_initial_errors_and_rejects_violation():
    initial = envelope(0., CHI, ELL)
    error = np.array([.15, -.12, .1, -.4, .3, -.35])
    state = transform(error, np.zeros(6), initial, CHI, ELL)
    assert np.all(np.isfinite(state.xi))
    assert np.all(np.isposinf(state.upper))
    assert np.all(np.isneginf(state.lower))
    later = envelope(20., CHI, ELL)
    with pytest.raises(FloatingPointError, match='domain violated'):
        transform(error, np.zeros(6), later, CHI, ELL)


def test_regularized_terminal_surface_is_c1_and_has_correct_derivative():
    alpha, beta, delta = .5, 1.5, .013
    lam1, lam2 = .8, 1.3
    xi = np.array([-2*delta, -delta, -.4*delta, 0., .3*delta,
                   delta, 1.8*delta])
    direction = np.array([.2, -.4, .3, .8, -.5, .7, -.2])
    surface, reference, reference_dot = terminal_surface(
        xi, direction, lam1, lam2, alpha, beta, delta)
    assert_allclose(surface, direction-reference)
    # Sig^beta has a Holder-continuous derivative at zero: its central
    # quotient error is O(step^(beta-1)), so use a smaller step at zero.
    step = np.where(xi == 0, 1e-15, 1e-8)
    plus = terminal_surface(xi+step*direction, direction,
                            lam1, lam2, alpha, beta, delta)[1]
    minus = terminal_surface(xi-step*direction, direction,
                             lam1, lam2, alpha, beta, delta)[1]
    assert_allclose((plus-minus)/(2*step), reference_dot,
                    rtol=2e-6, atol=3e-8)
    for sign in (-1., 1.):
        points = sign*delta+np.array([-1e-9, 0., 1e-9])
        _, values, derivatives = terminal_surface(
            points, np.ones(3), lam1, lam2, alpha, beta, delta)
        expected_value = -sign*(lam1*delta**alpha+lam2*delta**beta)
        expected_derivative = -(lam1*alpha*delta**(alpha-1)
                                +lam2*beta*delta**(beta-1))
        assert_allclose(values, expected_value, atol=3e-8, rtol=0)
        assert_allclose(derivatives, expected_derivative, atol=3e-6, rtol=0)
    assert np.isfinite(reference_dot[3])
    assert reference[3] == 0


@pytest.mark.parametrize('t', [0., .3, 1., 4., 8., 8.7, 9.5, 10., 12.])
def test_reference_first_two_derivatives_and_smooth_stop(t):
    center = np.array([.1, -.2, .8, .15, -.1, .25])
    step = 2e-6
    x, velocity, acceleration = desired(t, center)
    plus = desired(t+step, center)
    minus = desired(t-step, center)
    assert_allclose((plus[0]-minus[0])/(2*step), velocity,
                    atol=2e-9, rtol=2e-6)
    # The trajectory is C2, while its third derivative can jump at the gate.
    # A smaller step controls the O(step) acceleration estimate at that join.
    second_step = 2e-8
    velocity_plus = desired(t+second_step, center)[1]
    velocity_minus = desired(t-second_step, center)[1]
    assert_allclose((velocity_plus-velocity_minus)/(2*second_step), acceleration,
                    atol=2e-7, rtol=2e-6)
    if t == 0:
        assert_allclose(x, center)
        assert_allclose(velocity, 0)
        assert_allclose(acceleration, 0)
    if t >= 10:
        assert_allclose(velocity, 0, atol=0)
        assert_allclose(acceleration, 0, atol=0)
        assert_allclose(x, desired(t+2., center)[0], atol=0)


@pytest.fixture(scope='module')
def link_model():
    return LinkDynamics(MODEL)


CONFIGURATIONS = [
    (np.array([.35, -.65, .9, .45, -.8, .3]),
     np.array([.23, -.17, .14, .31, -.27, .19])),
    (np.array([-.6, .4, -.75, -.9, .65, -.3]),
     np.array([-.19, .29, -.11, .16, .24, -.21])),
    (np.array([.1, .7, -.4, .2, -.55, .8]), np.zeros(6)),
]


def _mujoco_mass_and_bias(model, q, qdot):
    data = mujoco.MjData(model)
    data.qpos[:] = q
    data.qvel[:] = qdot
    mujoco.mj_forward(model, data)
    full_mass = np.zeros((6, 6))
    mujoco.mj_fullM(model, data, full_mass)
    return full_mass, data.qfrc_bias.copy()


@pytest.mark.parametrize('q,qdot', CONFIGURATIONS)
def test_link_basis_matches_full_mujoco_dynamics(link_model, q, qdot):
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    assert model.nq == model.nv == 6
    assert_allclose(model.dof_armature, 0, atol=0)
    mass, coriolis, gravity = link_model.evaluate(q, qdot)
    assert mass.shape == coriolis.shape == (6, 6, 6)
    assert gravity.shape == (6, 6)
    expected_mass, expected_bias = _mujoco_mass_and_bias(model, q, qdot)
    assert_allclose(mass.sum(axis=0), expected_mass, rtol=2e-10, atol=3e-11)
    assert_allclose(coriolis.sum(axis=0)@qdot+gravity.sum(axis=0),
                    expected_bias, rtol=3e-8, atol=3e-8)
    assert np.linalg.eigvalsh(mass.sum(axis=0))[0] > 0
    for matrix in mass:
        assert_allclose(matrix, matrix.T, atol=1e-12)
        assert np.linalg.eigvalsh(matrix)[0] >= -1e-12


def test_density_scaling_uses_same_exact_regression_parameterization(link_model):
    q, qdot = CONFIGURATIONS[0]
    scales = np.array([.91, 1.09, .96, 1.13, .88, 1.04])
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    for index, scale in enumerate(scales, start=1):
        body = model.body('link'+str(index)).id
        model.body_mass[body] *= scale
        model.body_inertia[body] *= scale
    mujoco.mj_setConst(model, mujoco.MjData(model))
    expected_mass, expected_bias = _mujoco_mass_and_bias(model, q, qdot)
    mass, coriolis, gravity = link_model.evaluate(q, qdot)
    assert_allclose(np.einsum('b,bij->ij', scales, mass), expected_mass,
                    rtol=3e-10, atol=3e-11)
    assert_allclose(np.einsum('b,bij,j->i', scales, coriolis, qdot)
                    +np.einsum('b,bi->i', scales, gravity), expected_bias,
                    rtol=3e-8, atol=3e-8)


@pytest.mark.parametrize('q,qdot', CONFIGURATIONS[:2])
def test_each_link_basis_satisfies_mass_coriolis_skew_identity(link_model, q, qdot):
    step = 1e-5
    mass_plus = link_model.evaluate(q+step*qdot, qdot)[0]
    mass_minus = link_model.evaluate(q-step*qdot, qdot)[0]
    coriolis = link_model.evaluate(q, qdot)[1]
    skew = (mass_plus-mass_minus)/(2*step)-2*coriolis
    assert_allclose(skew+skew.transpose(0, 2, 1), 0, atol=3e-9)


def _independent_world_angular_map(rpy):
    # R = Rz(yaw) Ry(pitch) Rx(roll): columns are instantaneous world axes.
    _, pitch, yaw = rpy
    return np.array([[np.cos(yaw)*np.cos(pitch), -np.sin(yaw), 0.],
                     [np.sin(yaw)*np.cos(pitch), np.cos(yaw), 0.],
                     [-np.sin(pitch), 0., 1.]])


def test_object_dynamics_agrees_with_world_newton_euler():
    x = np.array([.2, -.3, .9, -.23, .31, .42])
    velocity = np.array([.1, -.08, .06, .27, -.19, .23])
    mass = 2.4
    principal = np.array([.047, .061, .083])
    matrix, coriolis, gravity = object_dynamics(x, velocity, mass, principal)
    rotation = Rotation.from_euler('xyz', x[3:]).as_matrix()
    inertia_world = rotation@np.diag(principal)@rotation.T
    angular_map = _independent_world_angular_map(x[3:])
    expected_mass = np.zeros((6, 6))
    expected_mass[:3, :3] = mass*np.eye(3)
    expected_mass[3:, 3:] = angular_map.T@inertia_world@angular_map
    assert_allclose(matrix, expected_mass, atol=1e-12)
    step = 1e-5
    map_dot = (_independent_world_angular_map(x[3:]+step*velocity[3:])
               -_independent_world_angular_map(x[3:]-step*velocity[3:]))/(2*step)
    omega = angular_map@velocity[3:]
    # Set coordinate acceleration to zero; world angular acceleration is not zero.
    torque_world = inertia_world@(map_dot@velocity[3:])
    torque_world += np.cross(omega, inertia_world@omega)
    expected_bias = np.r_[0., 0., mass*9.81, angular_map.T@torque_world]
    assert_allclose(coriolis@velocity+gravity, expected_bias,
                    rtol=5e-8, atol=1e-10)
    plus = object_dynamics(x+step*velocity, velocity, mass, principal)[0]
    minus = object_dynamics(x-step*velocity, velocity, mass, principal)[0]
    skew = (plus-minus)/(2*step)-2*coriolis
    assert_allclose(skew+skew.T, 0, atol=2e-10)


@pytest.mark.parametrize('t', [1.3, 4.4])
def test_transformed_momentum_identity_without_acceleration_measurement(link_model, t):
    kinematics = LocalKinematics(MODEL, np.zeros(3), np.eye(3),
                                [.02, -.01, .08], np.eye(3))
    q0 = CONFIGURATIONS[0][0]
    amplitude = np.array([.04, -.03, .025, .035, -.027, .03])
    frequency = np.array([.4, .7, .5, .6, .8, .45])
    offset = np.arange(6)*.2
    e_amp = np.array([.002, .0015, -.002, .008, -.007, .009])

    def terms(time):
        phase = frequency*time+offset
        q = q0+amplitude*np.sin(phase)
        qdot = amplitude*frequency*np.cos(phase)
        qddot = -amplitude*frequency**2*np.sin(phase)
        x, jacobian, jdot = kinematics.evaluate(q, qdot)
        error = e_amp*np.sin(.3*time+offset)
        error_dot = .3*e_amp*np.cos(.3*time+offset)
        barrier = transform(error, error_dot,
                            envelope(time, CHI, ELL, True), CHI, ELL)
        inverse = np.linalg.inv(jacobian)
        a = inverse/barrier.a
        a_dot = -inverse@jdot@a-a*(barrier.adot/barrier.a)
        d, c, w = coupled_bases(link_model.evaluate(q, qdot), jacobian,
                                jdot, x, jacobian@qdot, 2.4,
                                np.array([.047, .061, .083]))
        w[6] = .05*qdot
        momentum = np.array([a.T@basis@qdot for basis in d])
        remainder = np.array([a.T@(w[k]-c[k].T@qdot)
                              -a_dot.T@d[k]@qdot for k in range(8)])
        actual = np.array([a.T@(d[k]@qddot+c[k]@qdot+w[k])
                           for k in range(8)])
        return momentum, remainder, actual

    step = 1e-5
    momentum_dot = (terms(t+step)[0]-terms(t-step)[0])/(2*step)
    _, remainder, actual = terms(t)
    assert_allclose(momentum_dot+remainder, actual, rtol=1e-6, atol=2e-7)


def test_learning_filter_retains_initial_momentum_and_matches_analytic_convolution():
    learner = CompositeLearner(size=2, filter_rate=3.7, window=2.,
                               omega=1., gain=0.)
    rate = learner.rate
    a = np.array([[.7, -.2, .3, .1, -.4, .8], [-.3, .6, .2, -.5, .4, .1]])
    b = np.array([[.2, .3, -.1, .4, .1, -.2], [.3, -.2, .4, .2, -.3, .1]])
    c = np.array([[.1, -.3, .2, -.2, .4, .3], [.2, .1, -.3, .4, -.1, .2]])
    d = .2*a
    e = -.3*b
    omega, nu = 1.3, .9
    theta = np.array([1.07, .94])
    for t in np.linspace(0., .8, 1601):
        momentum = a+b*np.sin(omega*t)+c*np.cos(omega*t)
        remainder = d+e*np.sin(nu*t)
        actual = omega*b*np.cos(omega*t)-omega*c*np.sin(omega*t)+remainder
        learner.advance(t, momentum, remainder, actual.T@theta,
                        np.zeros((2, 6)), np.zeros(6))
    decay = np.exp(-rate*t)

    def filtered_sin(w):
        return rate/(rate**2+w**2)*(rate*np.sin(w*t)-w*np.cos(w*t)+w*decay)

    def filtered_cos(w):
        return rate/(rate**2+w**2)*(rate*np.cos(w*t)+w*np.sin(w*t)-rate*decay)

    expected = omega*b*filtered_cos(omega)-omega*c*filtered_sin(omega)
    expected += d*(1-decay)+e*filtered_sin(nu)
    assert_allclose(learner.filtered_regressor, expected, atol=3e-7, rtol=2e-6)
    assert_allclose(learner.tauf, expected.T@theta, atol=3e-7, rtol=2e-6)


def test_local_ie_freezes_history_but_continues_filtering():
    learner = CompositeLearner(size=2, filter_rate=10., window=.2,
                               omega=.01, gain=0.)
    regressor = np.zeros((2, 6))
    regressor[0, 0] = 1.
    regressor[1, 1] = 1.
    theta_true = np.array([1.08, .95])
    for t in np.linspace(0., .25, 251):
        learner.advance(t, np.zeros((2, 6)), regressor,
                        regressor.T@theta_true, np.zeros((2, 6)), np.zeros(6))
    assert learner.frozen_time is not None
    assert learner.frozen_time >= learner.window
    assert np.linalg.eigvalsh(learner.gram)[0] >= learner.omega
    frozen_gram, frozen_p = learner.gram.copy(), learner.p.copy()
    old_filtered = learner.filtered_regressor.copy()
    for t in np.linspace(.251, .4, 150):
        learner.advance(t, np.zeros((2, 6)), 3*regressor,
                        3*regressor.T@theta_true,
                        np.zeros((2, 6)), np.zeros(6))
    assert_allclose(learner.gram, frozen_gram, atol=0, rtol=0)
    assert_allclose(learner.p, frozen_p, atol=0, rtol=0)
    assert np.linalg.norm(learner.filtered_regressor-old_filtered) > .1
    assert_allclose(learner.epsilon,
                    frozen_gram@(theta_true-learner.theta), atol=1e-12)


def test_learning_projection_blocks_outward_updates_and_preserves_prior_box():
    learner = CompositeLearner(size=2, gain=10., gamma=0.)
    learner.theta[:] = [learner.upper[0], learner.lower[1]]
    zeros = np.zeros((2, 6))
    learner.advance(0., zeros, zeros, np.zeros(6), zeros, np.ones(6))
    outward = zeros.copy()
    outward[0, 0], outward[1, 0] = -1., 1.
    learner.advance(.01, zeros, zeros, np.zeros(6), outward, np.ones(6))
    assert_allclose(learner.theta, [1.15, .85], atol=0, rtol=0)
    learner.advance(.02, zeros, zeros, np.zeros(6), -outward, np.ones(6))
    assert np.all(learner.theta < learner.upper)
    assert np.all(learner.theta > learner.lower)
    learner.advance(1., zeros, zeros, np.zeros(6), 100*outward, np.ones(6))
    assert np.all(learner.theta <= learner.upper)
    assert np.all(learner.theta >= learner.lower)


def test_torque_filter_respects_held_motor_torque_with_changing_coordinate_map():
    """A left endpoint jump starts the next interval; it must not be ramped.

    The physical motor torque is piecewise constant, while the known
    coordinate map A(t) changes continuously. Direct high-order Gaussian
    quadrature gives the independent zero-state filter reference. Coarse,
    unequal intervals and large sign changes make the old interpolation of
    newly computed endpoint commands fail by a visible amount.
    """
    learner = CompositeLearner(size=1, filter_rate=3.2, window=2.,
                               omega=1., gain=0.)
    times = np.r_[0., np.cumsum([.07, .11, .05, .13, .09, .06, .12])]
    base = np.array([1.2, -.8, .4, 1.3, -1.1, .6])
    commands = np.array([(-1.)**index*(base+.13*index*np.arange(1., 7.))
                         for index in range(len(times))])
    derivative = np.arange(1., 37.).reshape(6, 6)/150.
    derivative -= .15*np.eye(6)
    fixed_jacobian = np.diag([.8, 1.1, .9, 1.2, .7, 1.05])
    desired_force = np.array([.4, -.3, .2, -.1, .15, -.25])
    force_torque = fixed_jacobian.T@desired_force

    def mapped_torque(time, command):
        return (np.eye(6)+time*derivative).T@(command+force_torque)

    zeros = np.zeros((1, 6))
    first = mapped_torque(0., commands[0])
    learner.advance(0., zeros, zeros, first, zeros, np.zeros(6),
                    next_known_torque=first)
    nodes, weights = np.polynomial.legendre.leggauss(12)
    expected = np.zeros(6)
    wrong_command_ramp = np.zeros(6)
    maximum_old_scheme_error = 0.
    for index in range(1, len(times)):
        start, end = times[index-1:index+1]
        dt = end-start
        quadrature_times = start+.5*dt*(nodes+1)
        exponential_weight = .5*dt*weights*learner.rate*np.exp(
            -learner.rate*(end-quadrature_times))
        physical = np.array([mapped_torque(time, commands[index-1])
                             for time in quadrature_times])
        expected = np.exp(-learner.rate*dt)*expected
        expected += exponential_weight@physical
        # This intentionally models the rejected, acausal command ramp so the
        # manufactured case proves sensitive to the timing error being fixed.
        fraction = (quadrature_times-start)/dt
        wrong_input = ((1-fraction[:, None])*mapped_torque(start, commands[index-1])
                        +fraction[:, None]*mapped_torque(end, commands[index]))
        wrong_command_ramp = np.exp(-learner.rate*dt)*wrong_command_ramp
        wrong_command_ramp += exponential_weight@wrong_input
        maximum_old_scheme_error = max(maximum_old_scheme_error,
                                        np.max(np.abs(wrong_command_ramp-expected)))
        endpoint_left_limit = mapped_torque(end, commands[index-1])
        endpoint_right_limit = mapped_torque(end, commands[index])
        learner.advance(end, zeros, zeros, endpoint_left_limit, zeros, np.zeros(6),
                        next_known_torque=endpoint_right_limit)
        assert_allclose(learner.tauf, expected, rtol=2e-13, atol=2e-13)
        assert_allclose(learner.previous[2], endpoint_right_limit, rtol=0, atol=0)
    assert maximum_old_scheme_error > .1


@pytest.mark.parametrize('robust_mode', [None, 'boundary_layer'])
def test_complete_paper_controller_in_joint_coordinates_includes_positive_rho(link_model,
                                                                              robust_mode):
    """Verify the whole torque by eliminating A from the paper's regression.

    For square J, (aJ)^T Y^T equals the joint reference-dynamics basis.
    Constructing that basis directly gives an independent equivalent formula
    for feedforward and parameter-box switching. The reference acceleration
    is obtained from differentiated coordinate rates, rather than the
    controller's A-dot/v-dot implementation. A visible positive rho tests the
    literal Property 3/5 gain even when constant load shares make L zero.
    """
    q, qdot = CONFIGURATIONS[0]
    kinematics = LocalKinematics(MODEL, [.1, -.2, 0.], np.eye(3),
                                [.02, -.01, .08], np.eye(3))
    center = kinematics.pose_jacobian(q)[0]
    settings = Settings(rho_hat=2e-4)
    assert settings.robust_mode == 'paper_sign'
    if robust_mode is not None:
        settings.robust_mode = robust_mode
    controller = JASController(kinematics, link_model, center, 2.4,
                               [.047, .061, .083], 0, settings)
    estimate = np.array([.88, 1.07, .96, 1.11, .91, 1.04, 1.02, .94])
    controller.learner.theta[:] = estimate
    time = .7
    actual_torque, info = controller.command(time, q, qdot)
    x, jacobian, jdot = kinematics.evaluate(q, qdot)
    reference, velocity, acceleration = desired(time, center)
    barrier = transform(x-reference, jacobian@qdot-velocity,
                        envelope(time, np.array(settings.chi), np.array(settings.ell)),
                        np.array(settings.chi), np.array(settings.ell))
    surface, xi_ref, xi_ref_acc = terminal_surface(
        barrier.xi, barrier.xidot, settings.lam1, settings.lam2,
        settings.alpha, settings.beta, settings.delta)
    task_reference_rate = velocity+(xi_ref-barrier.b)/barrier.a
    task_reference_acc = (acceleration+(xi_ref_acc-barrier.bdot)/barrier.a
                          -barrier.adot*(xi_ref-barrier.b)/barrier.a**2)
    joint_reference_rate = np.linalg.solve(jacobian, task_reference_rate)
    joint_reference_acc = np.linalg.solve(
        jacobian, task_reference_acc-jdot@joint_reference_rate)
    d, c, w = coupled_bases(link_model.evaluate(q, qdot), jacobian, jdot,
                            x, jacobian@qdot, 2.4, np.array([.047, .061, .083]))
    w[6] = .05*qdot
    joint_basis = np.array([d[k]@joint_reference_acc+c[k]@joint_reference_rate+w[k]
                            for k in range(8)])
    inverse = np.linalg.inv(jacobian)/barrier.a
    expected_rho = settings.rho_hat*np.linalg.norm(inverse, ord=2)**2
    parameter_diameter = settings.prior_upper-settings.prior_lower
    parameter_power = joint_basis@(qdot-joint_reference_rate)
    if robust_mode is None:
        switching_direction = np.sign(parameter_power)
    else:
        switching_direction = np.where(
            np.abs(parameter_power) < settings.robust_width,
            parameter_power/settings.robust_width, np.sign(parameter_power))
        assert np.any((np.abs(parameter_power) > 0)
                      & (np.abs(parameter_power) < settings.robust_width))
        assert np.any(np.abs(parameter_power) > settings.robust_width)
    switching = joint_basis.T@(parameter_diameter*switching_direction)
    smooth_feedback = ((np.array(settings.ka)+expected_rho)*surface
                       +np.array(settings.k2)*np.sign(surface)*np.abs(surface)**settings.upsilon1
                       +np.array(settings.k3)*np.sign(surface)*np.abs(surface)**settings.upsilon2)
    expected_torque = (joint_basis.T@estimate-switching
                       -jacobian.T@(barrier.a*smooth_feedback+controller.fd))
    assert expected_rho > 0
    assert_allclose(info['rho'], expected_rho, rtol=2e-14, atol=0)
    assert_allclose(actual_torque, expected_torque, rtol=3e-11, atol=3e-10)
    assert_allclose(info['regressor'], joint_basis@inverse, rtol=3e-11, atol=3e-10)
    assert_allclose(info['robust_power'],
                    parameter_power@(parameter_diameter*switching_direction),
                    rtol=3e-11, atol=3e-10)
    rho_torque = -jacobian.T@(barrier.a*expected_rho*surface)
    assert np.linalg.norm(rho_torque) > 1e-5
    without_rho = expected_torque-rho_torque
    assert_allclose(actual_torque-without_rho, rho_torque, rtol=2e-10, atol=5e-11)


@pytest.mark.parametrize('width', [1e-4, .05, .3, 10.])
def test_boundary_layer_prior_box_residual_bound_is_sharp(width):
    """Saturation has a bounded positive residual; the sign proof is changed.

    The two parameter vectors are selected independently from the entire
    prior box, rather than assuming a fortunate estimate or a half-width
    mismatch. The worst mismatch equals the box diameter, and is attained
    at opposite faces. The saturation residual peaks at |w_j|=epsilon/2.
    """
    rng = np.random.default_rng(74812)
    center = np.linspace(.7, 1.3, 8)
    radius = np.linspace(.05, .35, 8)
    lower, upper = center-radius, center+radius
    diameter = upper-lower
    estimate = rng.uniform(lower, upper, (5000, 8))
    truth = rng.uniform(lower, upper, (5000, 8))
    w = width*rng.uniform(-2., 2., (5000, 8))
    # Independent piecewise definition of the engineering boundary layer.
    direction = np.where(np.abs(w) <= width, w/width, np.sign(w))
    residual = w*(estimate-truth)-diameter*w*direction
    expected_envelope = diameter*np.where(
        np.abs(w) < width, np.abs(w)*(1-np.abs(w)/width), 0.)
    assert np.all(residual <= expected_envelope+2e-14*max(1., width))
    assert np.all(residual <= diameter*width/4+2e-14*max(1., width))
    assert np.all(np.sum(residual, axis=1) <= np.sum(diameter)*width/4
                  +2e-13*max(1., width))
    outside = np.abs(w) >= width
    assert np.all(residual[outside] <= 2e-14*max(1., width))
    paper_residual = w*(estimate-truth)-diameter*np.abs(w)
    assert np.all(paper_residual <= 2e-14*max(1., width))

    # Attain the exact upper bound independently at both sign choices.
    for sign in (-1., 1.):
        worst_w = np.full(8, sign*width/2)
        worst_estimate = upper if sign > 0 else lower
        worst_truth = lower if sign > 0 else upper
        worst_residual = (worst_w*(worst_estimate-worst_truth)
                          -diameter*worst_w*(worst_w/width))
        assert_allclose(worst_residual, diameter*width/4, rtol=3e-15, atol=0)
        assert np.all(worst_residual > 0), 'The exact sign-domination proof is lost.'


def test_paper_sign_default_torque_is_independent_of_boundary_width(link_model):
    q, qdot = CONFIGURATIONS[0]
    kin = LocalKinematics(MODEL, [.1, -.2, 0.], np.eye(3),
                          [.02, -.01, .08], np.eye(3))
    center = kin.pose_jacobian(q)[0]
    settings_narrow = Settings(robust_width=1e-5)
    settings_wide = Settings(robust_width=1e3)
    assert settings_narrow.robust_mode == settings_wide.robust_mode == 'paper_sign'
    commands = []
    for settings in (settings_narrow, settings_wide):
        controller = JASController(kin, link_model, center, 2.4,
                                   [.047, .061, .083], 0, settings)
        commands.append(controller.command(.7, q, qdot)[0])
    assert_allclose(commands[0], commands[1], rtol=0, atol=0)
