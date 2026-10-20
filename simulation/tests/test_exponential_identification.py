"""Independent checks of the explicitly added measured-force identifier.

Manufactured signals check the actual filter and data-driven update rather
than assigning a desired weight curve. The physical identity test uses
MuJoCo acceleration and plant truth only as an offline verification oracle;
neither quantity is passed to the production observation or learner.
"""
from pathlib import Path
import json
import sys

import mujoco
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

SIMULATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATION))

from exponential_identification.controller import IdentificationController, IdentificationSettings
from exponential_identification.learning import HistoricalIdentifier
from exponential_identification.regression import observe
from jas.controller import JASController, Settings
from jas.dynamics import LinkDynamics
from jas.geometry import LocalKinematics, angular_map
from jas.reference import desired
from run_experiments import initialize, kinematics


NOMINAL_ARM = SIMULATION/'models'/'arm_nominal.xml'


def _well_conditioned_basis():
    rng = np.random.default_rng(51321)
    return np.column_stack((np.eye(8)+.03*rng.normal(size=(8, 8)),
                             .1*rng.normal(size=(8, 4))))


def test_manufactured_unbiased_momentum_filter_and_true_exponential_update():
    """Linear momentum/source make endpoint filter integration exact."""
    rng = np.random.default_rng(7468)
    truth = np.array([.94, 1.06, 1.04, .97, 1.05, .96, .98, 1.08])
    m0 = .3*rng.normal(size=(8, 12))
    mdot = .1*rng.normal(size=(8, 12))
    r0 = _well_conditioned_basis()-mdot
    rdot = .04*rng.normal(size=(8, 12))
    learner = HistoricalIdentifier(filter_rate=3.7, window=.4,
                                    normalized_threshold=1e-4, rate=1.4)
    frozen = None
    for t in np.linspace(0., 2., 101):
        momentum = m0+t*mdot
        remainder = r0+t*rdot
        # This is a manufactured measured output, not a supplied learner target.
        known_force = (mdot+remainder).T@truth
        learner.advance(float(t), momentum, remainder, known_force)
        scalar_filter = -np.expm1(-learner.filter_rate*t)
        linear_filter = t-scalar_filter/learner.filter_rate
        expected_regressor = (mdot+r0)*scalar_filter+rdot*linear_filter
        assert_allclose(learner.regressor_f, expected_regressor,
                        rtol=2e-12, atol=2e-14)
        assert_allclose(learner.force_f, expected_regressor.T@truth,
                        rtol=2e-12, atol=2e-14)
        if learner.frozen_time is not None:
            if frozen is None:
                frozen = (learner.gram.copy(), learner.p.copy())
                assert_allclose(learner.data_target, truth, rtol=3e-13, atol=3e-13)
                assert_allclose(learner.theta_at_freeze, np.full(8, .9), rtol=0, atol=0)
            assert_allclose(learner.gram, frozen[0], rtol=0, atol=0)
            assert_allclose(learner.p, frozen[1], rtol=0, atol=0)
            expected = truth+np.exp(-learner.rate*(t-learner.frozen_time))*(
                learner.theta_at_freeze-truth)
            assert_allclose(learner.theta, expected, rtol=2e-13, atol=3e-14)
    assert learner.frozen_time == pytest.approx(.4)
    assert learner.projection_count == 0
    assert frozen is not None


def test_zero_state_filter_rejects_a_missing_initial_momentum_time():
    learner = HistoricalIdentifier(size=2, outputs=3)
    with pytest.raises(ValueError, match='start at t=0'):
        learner.advance(.01, np.ones((2, 3)), np.zeros((2, 3)), np.zeros(3))


def test_measured_force_filter_respects_endpoint_left_and_right_limits():
    """Jumps in measured force must start the next physical interval."""
    learner = HistoricalIdentifier(size=2, outputs=4, filter_rate=3.2, window=2.)
    times = np.r_[0., np.cumsum([.07, .11, .05, .13, .09, .06, .12])]
    commands = np.array([(-1.)**k*np.array([1.+.2*k, -.8, .5+.1*k, 1.3])
                         for k in range(len(times))])
    derivative = np.arange(1., 17.).reshape(4, 4)/70.

    def measured(time, command):
        return (np.eye(4)+time*derivative)@command

    zeros = np.zeros((2, 4))
    learner.advance(0., zeros, zeros, measured(0., commands[0]),
                    force_right=measured(0., commands[0]))
    nodes, weights = np.polynomial.legendre.leggauss(12)
    expected = np.zeros(4)
    wrong_ramp = np.zeros(4)
    maximum_wrong_error = 0.
    for k in range(1, len(times)):
        start, end = times[k-1:k+1]
        dt = end-start
        quadrature_t = start+.5*dt*(nodes+1)
        exponential_weights = (.5*dt*weights*learner.filter_rate
                               *np.exp(-learner.filter_rate*(end-quadrature_t)))
        true_signal = np.array([measured(t, commands[k-1]) for t in quadrature_t])
        expected = np.exp(-learner.filter_rate*dt)*expected+exponential_weights@true_signal
        fraction = (quadrature_t-start)/dt
        rejected_signal = ((1-fraction[:, None])*measured(start, commands[k-1])
                            +fraction[:, None]*measured(end, commands[k]))
        wrong_ramp = (np.exp(-learner.filter_rate*dt)*wrong_ramp
                      +exponential_weights@rejected_signal)
        maximum_wrong_error = max(maximum_wrong_error, np.max(np.abs(wrong_ramp-expected)))
        learner.advance(float(end), zeros, zeros, measured(end, commands[k-1]),
                        force_right=measured(end, commands[k]))
        assert_allclose(learner.force_f, expected, rtol=3e-13, atol=3e-13)
        assert_allclose(learner.previous[2], measured(end, commands[k]), rtol=0, atol=0)
    assert maximum_wrong_error > .1
    assert learner.frozen_time is None


def test_biased_measured_data_converge_to_data_target_with_nonzero_truth_floor():
    """Preconditioning changes speed, and does not remove a measured bias."""
    truth = np.array([.94, 1.06, 1.04, .97, 1.05, .96, .98, 1.08])
    bias = np.array([.035, -.025, .018, .022, -.017, .026, -.021, -.03])
    basis = _well_conditioned_basis()
    learner = HistoricalIdentifier(filter_rate=4.1, window=.4,
                                    normalized_threshold=1e-4, rate=2.5)
    zeros = np.zeros_like(basis)
    force = basis.T@(truth+bias)
    for t in np.linspace(0., 8., 401):
        learner.advance(float(t), zeros, basis, force)
        if learner.frozen_time is not None:
            assert_allclose(learner.data_target, truth+bias, rtol=2e-13, atol=2e-13)
            decay = np.exp(-learner.rate*(t-learner.frozen_time))
            exact_truth_error = (decay*(learner.theta_at_freeze-truth)
                                 +(1-decay)*bias)
            assert_allclose(learner.theta-truth, exact_truth_error,
                            rtol=5e-12, atol=3e-14)
            envelope = (decay*np.linalg.norm(learner.theta_at_freeze-truth)
                        +(1-decay)*np.linalg.norm(bias))
            assert np.linalg.norm(learner.theta-truth) <= envelope+2e-13
    stored_residual = learner.p-learner.gram@truth
    assert_allclose(stored_residual, learner.gram@bias, rtol=1e-12, atol=2e-14)
    assert_allclose(learner.theta, learner.data_target, rtol=0, atol=2e-9)
    assert np.linalg.norm(learner.theta-truth) > .04


def test_scaled_full_rank_gram_has_data_driven_exponential_contraction():
    """A badly scaled raw Gram must not set the parameter decay speed."""
    rng = np.random.default_rng(2818)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(12, 8)))
    scales = np.geomspace(1e-3, 1e3, 8)
    correlation = np.eye(8)+.04*np.ones((8, 8))
    basis = scales[:, None]*(correlation@orthogonal.T)
    truth = np.array([1.04, .94, .98, 1.02, .96, 1.06, 1.01, 1.08])
    learner = HistoricalIdentifier(filter_rate=3., window=.5,
                                    normalized_threshold=.1, rate=2.3)
    frozen = False
    for t in np.linspace(0., 2., 81):
        learner.advance(float(t), np.zeros_like(basis), basis, basis.T@truth)
        if learner.frozen_time is not None:
            frozen = True
            assert np.linalg.cond(learner.gram) > 1e11
            assert learner.normalized_minimum_eigenvalue > .5
            # Forming a measured output with a 1e6 row-scale range loses a
            # little precision before preconditioning can operate on it.
            assert_allclose(learner.data_target, truth, rtol=3e-10, atol=3e-10)
            expected_error = np.exp(-learner.rate*(t-learner.frozen_time))*(
                learner.theta_at_freeze-learner.data_target)
            assert_allclose(learner.theta-learner.data_target, expected_error,
                            rtol=1e-9, atol=3e-11)
    assert frozen
    assert learner.projection_count == 0


def test_rank_deficient_history_is_not_made_exciting_by_normalization():
    basis = np.ones((8, 12))
    learner = HistoricalIdentifier(filter_rate=3., window=.2,
                                    normalized_threshold=1e-5)
    for t in np.linspace(0., 1., 51):
        learner.advance(float(t), np.zeros_like(basis), basis, basis.T@np.ones(8))
    assert learner.frozen_time is None
    assert learner.data_target is None
    assert np.linalg.matrix_rank(learner.gram) == 1
    assert_allclose(learner.theta, np.full(8, .9), rtol=0, atol=0)


def test_outside_prior_data_target_is_rejected_without_hiding_the_bias():
    basis = np.column_stack((np.eye(8), np.zeros((8, 4))))
    learner = HistoricalIdentifier(filter_rate=3., window=.2,
                                    normalized_threshold=1e-5)
    for t in np.linspace(0., 1., 51):
        learner.advance(float(t), np.zeros_like(basis), basis,
                        basis.T@np.full(8, 1.22))
    assert learner.minimum_eigenvalue > 0
    assert learner.normalized_minimum_eigenvalue > .99
    assert learner.rejected_outside_prior > 0
    assert learner.frozen_time is None
    assert learner.data_target is None
    assert_allclose(learner.theta, np.full(8, .9), rtol=0, atol=0)


@pytest.mark.parametrize('mode', ['paper_sign', 'boundary_layer'])
@pytest.mark.parametrize('hand', [0, 1])
def test_identification_controller_keeps_same_jas_torque_for_same_reference_and_estimate(mode,
                                                                                     hand):
    kin = LocalKinematics(NOMINAL_ARM, [.1, -.2, 0.], np.eye(3),
                          [.02, -.01, .08], np.eye(3))
    dynamics = LinkDynamics(NOMINAL_ARM)
    q = np.array([.35, -.65, .9, .45, -.8, .3])
    qdot = np.array([.23, -.17, .14, .31, -.27, .19])
    center = kin.pose_jacobian(q)[0]
    args = (kin, dynamics, center, 2., [.0290666666667, .0154666666667, .0397333333333], hand)
    original = JASController(*args, settings=Settings(robust_mode=mode, rho_hat=2e-4))
    identifier = HistoricalIdentifier()
    extension = IdentificationController(
        *args, settings=IdentificationSettings(robust_mode=mode, rho_hat=2e-4),
        reference=desired, identifier=identifier)
    estimate = np.array([1.04, .91, .98, 1.02, 1.06, .89, 1.09, 1.01])
    original.learner.theta[:] = estimate
    identifier.theta[:] = estimate
    for t in (0., .7, 1.2):
        expected_torque, expected_info = original.command(t, q, qdot)
        torque, info = extension.command(t, q, qdot)
        assert_allclose(torque, expected_torque, rtol=0, atol=0)
        for key in ('x', 'xd', 'reference', 'error', 'jacobian', 'surface', 'xi',
                    'barrier_margin', 'upper', 'lower', 'sigma_min'):
            assert_allclose(info[key], expected_info[key], rtol=0, atol=0)


@pytest.mark.parametrize('orientation', [(0., 0., 0.), (.14, -.11, .16)])
def test_measured_force_momentum_identity_matches_full_mujoco_physics(orientation):
    """Use independent directional dynamics, including actual payload rotation.

    Acceleration and truth occur only in this offline finite-difference oracle.
    Production observe() receives neither, and its inertia bases are nominal.
    """
    metadata = json.loads((SIMULATION/'models'/'scene_metadata.json').read_text())
    plant_truth = json.loads((SIMULATION/'models'/'plant_truth.json').read_text())
    model = mujoco.MjModel.from_xml_path(str(SIMULATION/'models'/'dual_cr7.xml'))
    data = mujoco.MjData(model)
    kins = [kinematics(metadata, name) for name in ('left', 'right')]
    center = np.r_[metadata['payload_nominal']['initial_position_world_m'], np.zeros(3)]
    target = initialize(model, data, metadata, kins, center,
                        [.012, -.008, .01, *orientation])
    task_rate = np.array([.013, -.008, .01, .09, -.07, .05])
    for arm in range(2):
        jacobian = kins[arm].pose_jacobian(data.qpos[arm*6:arm*6+6])[1]
        data.qvel[arm*6:arm*6+6] = np.linalg.solve(jacobian, task_rate)
    rotation = Rotation.from_euler('xyz', target[3:]).as_matrix()
    data.qvel[12:15] = task_rate[:3]
    data.qvel[15:18] = rotation.T@angular_map(target[3:])@task_rate[3:]
    data.ctrl[:] = np.array([2., -3., 1., .6, -.5, .3, -1., 2., -.7, .4, .3, -.2])
    mujoco.mj_forward(model, data)
    nominal = LinkDynamics(NOMINAL_ARM)
    payload = metadata['payload_nominal']

    def measurement(state):
        bases = [nominal.evaluate(state.qpos[6*arm:6*arm+6],
                                  state.qvel[6*arm:6*arm+6]) for arm in range(2)]
        return observe(model, state, bases, payload['mass_kg'],
                       payload['diagonal_inertia_kg_m2'])

    observed = measurement(data)
    assert data.ncon == 0
    truth = np.array([plant_truth[name+'_density_multipliers']
                      +[plant_truth[name+'_viscous_multiplier'],
                        plant_truth['payload_density_multiplier']]
                      for name in ('left', 'right')])
    step = 2e-6
    perturbed = []
    for sign in (-1., 1.):
        state = mujoco.MjData(model)
        state.qpos[:] = data.qpos
        mujoco.mj_integratePos(model, state.qpos, data.qvel, sign*step)
        state.qvel[:] = data.qvel+sign*step*data.qacc
        state.ctrl[:] = data.ctrl
        mujoco.mj_forward(model, state)
        perturbed.append(measurement(state))
    momentum_derivative = (perturbed[1].momentum-perturbed[0].momentum)/(2*step)
    predicted = np.einsum('aik,ai->ak', momentum_derivative+observed.remainder, truth)
    assert_allclose(predicted, observed.applied_force, rtol=3e-7, atol=8e-7)
    assert_allclose(observed.applied_force[:, :6], observed.robot_applied, rtol=0, atol=0)
    assert_allclose(observed.applied_force[0, 6:], observed.payload_wrench, rtol=0, atol=0)
    assert_allclose(observed.applied_force[1, 6:], observed.payload_wrench, rtol=0, atol=0)
