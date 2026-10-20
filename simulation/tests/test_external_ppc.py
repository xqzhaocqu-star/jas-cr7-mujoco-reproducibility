"""Independent formula/frame checks of the complete external PPC baseline.

These tests use no physical rollout.  The allocation oracle is a separately
formed weighted right inverse, and the real-model frame oracle is MuJoCo's
geometric palm Jacobian rather than the implementation's object Jacobian.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import mujoco
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.linalg import block_diag
from scipy.spatial.transform import Rotation

SIMULATION = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SIMULATION))

from comparison.controller import ComparisonSettings, common_bounds
from comparison.external_ppc import ExternalPPC, _log_transform
from run_experiments import kinematics


GRASP_POSITIONS = np.array([[0., -.20, 0.], [0., .20, 0.]])
POSE = np.array([.5, -.1, .7, .15, -.22, .4])
JACOBIAN = np.array([
    [.8, .1, 0., .03, 0., 0.],
    [0., .9, .05, 0., -.04, 0.],
    [.02, 0., 1.1, 0., 0., .03],
    [0., .03, 0., .9, .08, 0.],
    [.01, 0., 0., -.04, 1.2, .05],
    [0., 0., -.02, .04, 0., 1.1],
])
QD = np.array([.015, -.02, .013, .03, -.022, .012])
OFFSET = np.array([.008, -.006, .004, .025, -.015, .02])


class ForbiddenInput:
    """Any attempt to inspect unknown dynamics/parameters must fail."""

    def __getattribute__(self, name):
        raise AssertionError('The external controller inspected a forbidden input: '+name)


class EncoderKinematics:
    def __init__(self, pose, hand):
        self.pose = np.asarray(pose).copy()
        self.jacobian = JACOBIAN.copy()
        self.rotation = Rotation.from_euler('xyz', [.13, -.07, .21]).as_matrix()
        self.offset = -self.rotation@GRASP_POSITIONS[hand]
        self.calls = 0

    def pose_jacobian(self, q):
        assert np.asarray(q).shape == (6,)
        self.calls += 1
        return self.pose.copy(), self.jacobian.copy()

    def evaluate(self, *args):
        raise AssertionError('PPC must not request a Jacobian derivative or acceleration.')


def reference(t, center):
    amplitude = np.array([.001, .0015, .0007, .002, .001, .0015])
    frequency = np.array([.7, .9, .8, 1.1, .6, 1.2])
    return (np.asarray(center)+amplitude*np.sin(frequency*t),
            amplitude*frequency*np.cos(frequency*t),
            -amplitude*frequency**2*np.sin(frequency*t))


def twist_map_oracle(rpy):
    # The world rotation axes of extrinsic xyz Euler rates.  This uses
    # independently rotated unit vectors, not the implementation's entries.
    rotation = Rotation.from_euler('xyz', rpy).as_matrix()
    yaw_rotation = Rotation.from_euler('z', rpy[2]).as_matrix()
    axes = np.column_stack((rotation[:, 0], yaw_rotation[:, 1], [0., 0., 1.]))
    return block_diag(np.eye(3), axes)


def grasp_and_allocation_oracle(rpy, virtual_mass, virtual_inertia):
    rotation = Rotation.from_euler('xyz', rpy)
    grasps = []
    for point in GRASP_POSITIONS:
        # d points from the palm to object CoM in world coordinates.
        d = -rotation.apply(point)
        cross_matrix = np.column_stack([np.cross(d, axis) for axis in np.eye(3)])
        grasps.append(np.block([[np.eye(3), cross_matrix],
                                [np.zeros((3, 3)), np.eye(3)]]))
    g = np.vstack(grasps)
    weight = block_diag(*[block_diag(virtual_mass*np.eye(3),
                                    virtual_inertia*np.eye(3)) for _ in range(2)])
    # Independent variational right inverse: minimum complementary weighted
    # wrench norm subject to a prescribed total object wrench.
    allocation = np.linalg.solve(g.T@weight@g, (weight@g).T).T
    return np.asarray(grasps), g, allocation


def make_controller(hand=0, *, pose=POSE, settings=None, force=5.,
                    ref=reference, center=None, kin=None):
    known = dict(gs=.013, gv=7.5, virtual_mass=2.3, virtual_inertia=.7,
                 rho_v_initial=(2.4, 2.5, 2.6, 2.7, 2.8, 2.9))
    if settings:
        known.update(settings)
    kin = EncoderKinematics(pose, hand) if kin is None else kin
    return ExternalPPC(kin, ForbiddenInput(),
                       np.asarray(pose)-OFFSET if center is None else center,
                       ForbiddenInput(), ForbiddenInput(), hand,
                       SimpleNamespace(internal_force=force),
                       reference=ref, external_settings=known)


def formula_oracle(controller, t, qd, hand):
    # Equations (25)--(36), with all transformed variables reconstructed
    # from encoder quantities.  Desired derivatives are intentionally unused.
    c = controller.external_settings
    x, j = controller.kinematics.pose, controller.kinematics.jacobian
    desired, _, _ = controller.reference(t, controller.center)
    error = x-desired
    rho_s = ((np.asarray(c['rho_s_initial'])-np.asarray(c['rho_s_final']))
             *np.exp(-np.asarray(c['rho_s_rate'])*t)+c['rho_s_final'])
    xi_s = error/rho_s
    eps_s = np.log((rho_s+error)/(rho_s-error))
    r_s = 2*rho_s**2/(rho_s**2-error**2)
    b = twist_map_oracle(x[3:])
    vr = -c['gs']*b@(r_s*eps_s/rho_s)
    velocity = b@j@qd
    ev = velocity-vr
    rho_v = ((controller.rho_v_initial-np.asarray(c['rho_v_final']))
             *np.exp(-np.asarray(c['rho_v_rate'])*t)+c['rho_v_final'])
    xi_v = ev/rho_v
    eps_v = np.log((rho_v+ev)/(rho_v-ev))
    r_v = 2*rho_v**2/(rho_v**2-ev**2)
    feedback = -c['gv']*r_v*eps_v/rho_v
    grasps, g, allocation = grasp_and_allocation_oracle(
        x[3:], c['virtual_mass'], c['virtual_inertia'])
    # The two requested Euler-CoM actuator wrenches are opposite.  Their
    # independently converted physical palm wrenches are already in null(G.T).
    fd = np.array([0., (-1. if hand == 0 else 1.)*controller.cfg.internal_force,
                   0., 0., 0., 0.])
    internal = np.linalg.solve(grasps[hand].T, np.linalg.solve(b.T, -fd))
    palm = allocation[6*hand:6*hand+6]@feedback+internal
    torque = (grasps[hand]@b@j).T@palm
    return torque, dict(error=error, rho_s=rho_s, rho_v=rho_v, xi_s=xi_s,
                        xi_v=xi_v, eps_s=eps_s, eps_v=eps_v, vr=vr,
                        velocity=velocity, ev=ev, feedback=feedback,
                        allocation=allocation[6*hand:6*hand+6],
                        grasp=grasps[hand], internal=internal, palm=palm, g=g)


@pytest.mark.parametrize('hand', [0, 1])
def test_complete_published_ppc_matches_independent_equation_and_frame_oracle(hand):
    controller = make_controller(hand)
    for t in (0., .4, 1.3):
        torque, observed = controller.command(t, np.zeros(6), QD)
        expected, oracle = formula_oracle(controller, t, QD, hand)
        assert_allclose(torque, expected, rtol=7e-14, atol=8e-14)
        mapping = dict(ppc_rho_s='rho_s', ppc_rho_v='rho_v', ppc_xi_s='xi_s',
                       ppc_xi_v='xi_v', ppc_epsilon_s='eps_s', ppc_epsilon_v='eps_v',
                       ppc_v_reference='vr', ppc_object_velocity='velocity',
                       ppc_object_feedback='feedback', ppc_load_allocation='allocation',
                       ppc_grasp_map='grasp', ppc_internal_palm_wrench='internal',
                       ppc_palm_wrench='palm', surface='ev', error='error')
        for actual, name in mapping.items():
            assert_allclose(observed[actual], oracle[name], rtol=8e-14, atol=8e-14,
                            err_msg=actual)
        assert observed['ppc_allocation_identity_error'] < 1e-14
        assert observed['ppc_internal_null_error'] < 1e-13


@pytest.mark.parametrize('rpy', [[0., 0., 0.], [.3, -.4, .7],
                                 [-1.1, .6, -1.8], [.9, -1.2, 2.2]])
def test_printed_allocation_equals_independent_weighted_right_inverse_and_null_projector(rpy):
    controller = make_controller()
    grasps, shares = controller._grasp_matrices(rpy)
    expected_grasps, g, allocation = grasp_and_allocation_oracle(rpy, 2.3, .7)
    assert_allclose(grasps, expected_grasps, rtol=1e-15, atol=1e-15)
    assert_allclose(np.vstack(shares), allocation, rtol=3e-15, atol=2e-16)
    assert_allclose(g.T@np.vstack(shares), np.eye(6), rtol=0, atol=4e-16)
    n = np.eye(12)-allocation@g.T
    assert_allclose(g.T@n, np.zeros((6, 12)), rtol=0, atol=5e-16)
    assert_allclose(n@n, n, rtol=0, atol=5e-16)
    arbitrary = np.arange(12.)-4.3
    assert_allclose(g.T@(n@arbitrary), np.zeros(6), rtol=0, atol=8e-15)


@pytest.mark.parametrize('hand', [0, 1])
@pytest.mark.parametrize('rpy', [[0., 0., 0.], [.15, -.22, .4], [-.2, .17, -.6]])
def test_remark5_internal_wrench_has_exact_original_euler_generalized_target(hand, rpy):
    pose = POSE.copy(); pose[3:] = rpy
    with_force = make_controller(hand, pose=pose, center=pose-OFFSET)
    no_force = make_controller(hand, pose=pose, center=pose-OFFSET, force=0.)
    with_torque, info = with_force.command(0., np.zeros(6), QD)
    without_torque, no_info = no_force.command(0., np.zeros(6), QD)
    assert_allclose(with_torque-without_torque, -JACOBIAN.T@with_force.fd,
                    rtol=0, atol=4e-15)
    assert_array_equal(info['ppc_object_feedback'], no_info['ppc_object_feedback'])
    point_wrench = info['ppc_internal_palm_wrench']
    object_wrench = info['ppc_grasp_map'].T@point_wrench
    assert_allclose(twist_map_oracle(rpy).T@object_wrench, -with_force.fd,
                    rtol=0, atol=3e-15)


@pytest.fixture(scope='module')
def scene_metadata():
    return json.loads((SIMULATION/'models'/'scene_metadata.json').read_text())


@pytest.mark.parametrize('hand_name,hand', [('left', 0), ('right', 1)])
@pytest.mark.parametrize('perturbation', [np.zeros(6), [.07, -.06, .04, .02, -.03, .05]])
def test_object_to_palm_jacobian_matches_independent_mujoco_site_jacobian(
        scene_metadata, hand_name, hand, perturbation):
    kin = kinematics(scene_metadata, hand_name)
    q = np.asarray(scene_metadata['q_initial'])[6*hand:6*hand+6]+perturbation
    pose, j = kin.pose_jacobian(q)
    controller = make_controller(hand, pose=pose, center=pose, kin=kin)
    grasp = controller._grasp_matrices(pose[3:])[0][hand]
    composed = grasp@twist_map_oracle(pose[3:])@j
    linear, angular = np.zeros((3, 6)), np.zeros((3, 6))
    mujoco.mj_jacSite(kin.model, kin.data, linear, angular, kin.site)
    expected = np.vstack((kin.base_r@linear, kin.base_r@angular))
    assert_allclose(composed, expected, rtol=5e-13, atol=5e-13)


@pytest.mark.parametrize('hand', [0, 1])
def test_norm_based_velocity_funnel_initializes_once_from_own_encoders(hand):
    controller = make_controller(hand, settings={'rho_v_initial': None,
                                                'velocity_initial_slack': 1.1})
    _, initial = controller.command(0., np.zeros(6), QD)
    expected = np.full(6, np.linalg.norm(initial['surface'])+1.1)
    assert_allclose(controller.rho_v_initial, expected, rtol=0, atol=0)
    assert np.all(np.abs(initial['ppc_xi_v']) < 1.)
    controller.command(.2, np.ones(6), QD*.5)
    assert_array_equal(controller.rho_v_initial, expected)
    config = controller.configuration()
    assert_array_equal(config['settings']['resolved_rho_v_initial'], expected)


@pytest.mark.parametrize('hand', [0, 1])
def test_reference_derivatives_and_unknown_dynamics_never_enter_command(hand):
    def derivative_poison(t, center):
        pose, _, _ = reference(t, center)
        return pose, ForbiddenInput(), ForbiddenInput()

    first = make_controller(hand, ref=reference)
    second = make_controller(hand, ref=derivative_poison)
    for t in (0., .2, .7):
        torque, info = first.command(t, np.zeros(6), QD)
        other, other_info = second.command(t, np.zeros(6), QD)
        assert_array_equal(torque, other)
        for key in ('ppc_v_reference', 'ppc_object_feedback', 'ppc_palm_wrench'):
            assert_array_equal(info[key], other_info[key])
    assert first.kinematics.calls == 3
    assert not any(hasattr(first, name) for name in ('dynamics', 'mass', 'inertia', 'learner'))


def test_external_has_no_learning_or_filter_and_marks_jas_quantities_not_applicable():
    controller = make_controller()
    _, info = controller.command(0., np.zeros(6), QD)
    before = deepcopy(controller.configuration())
    assert controller.update(ForbiddenInput(), ForbiddenInput(), ForbiddenInput()) is None
    assert controller.configuration() == before
    assert not before['has_adaptation'] and not before['has_dynamic_filter']
    for key in ('regressor', 'momentum', 'remainder', 'A', 'rho', 'robust_power'):
        assert np.all(np.isnan(info[key]))
    assert 'JAS S' in before['not_applicable']
    assert 'IE' in before['not_applicable']


def test_native_symmetric_monotone_funnels_remain_distinct_from_common_evaluation():
    hold = lambda t, center: (center, np.zeros(6), np.zeros(6))
    controller = make_controller(ref=hold, center=POSE)
    cfg = ComparisonSettings()
    all_native = []
    controller.command(0., np.zeros(6), np.zeros(6))
    for t in (0., .2, 1., 3., 7.):
        _, info = controller.command(t, np.zeros(6), np.zeros(6))
        native = info['ppc_rho_s']
        all_native.append(native)
        assert_array_equal(info['upper'], native)
        assert_array_equal(info['lower'], -native)
        common = common_bounds(t, cfg, nonmonotonic=True)
        assert not np.allclose(native, common[0])
        assert not np.allclose(native, common[1])
    assert np.all(np.diff(all_native, axis=0) < 0)


@pytest.mark.parametrize('value', [-1., 1., -1.00001, 1.00001, np.nan, np.inf])
def test_native_log_transform_rejects_exact_boundary_and_invalid_values_without_clipping(value):
    xi = np.zeros(6); xi[2] = value
    with pytest.raises(FloatingPointError, match='domain violated'):
        _log_transform(xi, 'manufactured')


@pytest.mark.parametrize('sign', [-1., 1.])
def test_actual_pose_funnel_violation_is_explicit(sign):
    error = np.zeros(6); error[0] = sign*.081
    hold = lambda t, center: (center, np.zeros(6), np.zeros(6))
    controller = make_controller(ref=hold, center=POSE-error)
    with pytest.raises(FloatingPointError, match='pose domain violated'):
        controller.command(0., np.zeros(6), np.zeros(6))


@pytest.mark.parametrize('sign', [-1., 1.])
def test_actual_velocity_funnel_violation_is_explicit(sign):
    hold = lambda t, center: (center, np.zeros(6), np.zeros(6))
    controller = make_controller(ref=hold, center=POSE,
                                 settings={'rho_v_initial': (1.,)*6})
    desired_velocity = np.array([1.01*sign, 0., 0., 0., 0., 0.])
    qd = np.linalg.solve(twist_map_oracle(POSE[3:])@JACOBIAN, desired_velocity)
    with pytest.raises(FloatingPointError, match='velocity domain violated'):
        controller.command(0., np.zeros(6), qd)


@pytest.mark.parametrize('settings', [
    {'gs': 0.}, {'gv': -1.}, {'gs': np.nan},
    {'rho_s_initial': (.01,)*6}, {'rho_s_rate': (.1,)*5},
    {'rho_v_final': (.4, .4, .4, .8, .8, 0.)},
    {'rho_s_initial': (.08, .08, .08, .22, 1.3, .22)},
    {'virtual_mass': -1.}, {'virtual_inertia': 0.},
    {'grasp_positions_object': ((0., -.2, 0.), (0., .19, 0.))},
    {'unpublished_feedback_term': 1.},
])
def test_invalid_native_or_load_sharing_constants_are_rejected(settings):
    with pytest.raises(ValueError):
        make_controller(settings=settings)


@pytest.mark.parametrize('time', [.1, -1e-3, np.nan])
def test_initial_encoder_sample_must_be_at_finite_nonnegative_zero_time(time):
    controller = make_controller(settings={'rho_v_initial': None})
    with pytest.raises(ValueError):
        controller.command(time, np.zeros(6), QD)
