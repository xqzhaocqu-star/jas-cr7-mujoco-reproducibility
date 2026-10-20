"""Acceleration-free momentum signals using measured interaction forces.

This is an explicitly additional sensing architecture, rather than the JAS
paper's nominal-internal-force regression. Each local eight-parameter model
has twelve observation channels: its six measured robot joint equations and
the shared payload's six world-frame COM momentum equations. Measured grasp
reactions make those equations valid even while numerical welds deform.

The only plant quantities read here are joint velocity, payload pose/twist,
actual actuator generalized force, and measured generalized constraint force.
Model reads are names, joint addresses, kinematic Jacobians, and known gravity.
No plant mass, inertia, damping, acceleration, inverse dynamics, or true
parameters enter the observations. All inertia bases must be supplied by the
separate nominal robot models. The module performs no filtering or learning.

At every observation call, callers must first run ``mj_forward`` with the
held motor input for that endpoint. Thus velocity and force describe the same
state and input. ``qfrc_constraint`` is a virtual force sensor in these
experiments: physical hardware would need measured motor/grasp wrenches.
The experiment checks absence of ground/contact/joint-limit reactions; if
other reactions are present this sensor includes them as measured forces.
"""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class Observation:
    """Signals satisfying ``d(momentum.T @ theta)/dt + remainder.T @ theta = u``.

    ``momentum`` and ``remainder`` have axes [arm, parameter, observation]
    and shape (2, 8, 12). ``applied_force`` has shape (2, 12). The last six
    observation channels use world-frame force and torque about the payload
    COM; robot channels use local generalized joint coordinates. Payload
    information is shared by both learners, not counted twice within either
    learner. The remaining fields expose sensor values for independent QA.
    """

    momentum: np.ndarray
    remainder: np.ndarray
    applied_force: np.ndarray
    robot_applied: np.ndarray
    payload_wrench: np.ndarray
    payload_twist: np.ndarray


def _joint_dofs(model, prefix):
    return np.array([model.jnt_dofadr[model.joint(f'{prefix}_joint{i}').id]
                     for i in range(1, 7)], dtype=int)


def observe(model, data, nominal_robot_bases, mass, inertia, *,
            applied_torque=None):
    """Construct unbiased-in-interaction-force momentum regression signals.

    Args:
        model, data: Current dual-CR7 plant and synchronously evaluated data.
        nominal_robot_bases: Two ``(M, C, g)`` tuples, each returned by the
            existing ``LinkDynamics.evaluate(q, qd)`` on a *nominal* model at
            the corresponding measured local joint position and velocity.
            Their shapes are (6,6,6), (6,6,6), (6,6), respectively.
        mass: Known nominal payload mass before its unknown density scaling.
        inertia: Known nominal payload COM inertia in body coordinates,
            either its three diagonal entries or a 3x3 matrix.
        applied_torque: Optional measured/held motor torques, shape (2,6).
            Defaults to current ``data.qfrc_actuator`` joint components.

    Parameter order remains six independent link density scales, the common
    scale on nominal viscous damping 0.05, and payload density. Actual passive
    damping is not read: it is represented by its unknown seventh parameter.
    Neither desired internal force nor an ideal rigid-grasp acceleration map
    occurs in these equations. Small discretization/filtering errors remain
    possible; zero interaction-force modelling bias alone does not remove
    those errors or guarantee excitation.
    """
    if len(nominal_robot_bases) != 2:
        raise ValueError('Two nominal robot basis tuples are required.')
    mass = float(mass)
    inertia = np.asarray(inertia, dtype=float)
    if inertia.shape == (3,):
        inertia = np.diag(inertia)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('Nominal payload mass must be positive.')
    if inertia.shape != (3, 3) or not np.all(np.isfinite(inertia)):
        raise ValueError('Nominal payload inertia must be a 3-vector or 3x3 matrix.')
    if not np.allclose(inertia, inertia.T) or np.linalg.eigvalsh(inertia)[0] <= 0:
        raise ValueError('Nominal payload inertia must be symmetric positive definite.')

    dofs = [_joint_dofs(model, name) for name in ('left', 'right')]
    torque = (np.array([data.qfrc_actuator[idx] for idx in dofs])
              if applied_torque is None else np.asarray(applied_torque, dtype=float))
    if torque.shape != (2, 6) or not np.all(np.isfinite(torque)):
        raise ValueError('Measured motor torques must have shape (2,6).')

    momentum = np.zeros((2, 8, 12))
    remainder = np.zeros_like(momentum)
    robot_applied = np.empty((2, 6))
    for arm, (indices, bases) in enumerate(zip(dofs, nominal_robot_bases)):
        matrices, coriolis, gravity = (np.asarray(value, dtype=float) for value in bases)
        if (matrices.shape != (6, 6, 6) or coriolis.shape != (6, 6, 6)
                or gravity.shape != (6, 6)):
            raise ValueError('Nominal robot basis shapes must be (6,6,6), (6,6,6), (6,6).')
        velocity = np.asarray(data.qvel[indices])
        momentum[arm, :6, :6] = np.einsum('bij,j->bi', matrices, velocity)
        remainder[arm, :6, :6] = gravity-np.einsum('bji,j->bi', coriolis, velocity)
        remainder[arm, 6, :6] = .05*velocity
        robot_applied[arm] = torque[arm]+data.qfrc_constraint[indices]

    payload = model.body('payload').id
    free_joint = model.joint('payload_free').id
    if model.jnt_type[free_joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('Payload must have a free joint.')
    first_dof = int(model.jnt_dofadr[free_joint])
    free_dofs = slice(first_dof, first_dof+6)
    # This purely kinematic map handles MuJoCo's body-local rotational free-
    # joint velocity and any COM offset without guessing a force convention.
    linear_jacobian = np.zeros((3, model.nv))
    angular_jacobian = np.zeros_like(linear_jacobian)
    mujoco.mj_jacBodyCom(model, data, linear_jacobian, angular_jacobian, payload)
    free_jacobian = np.vstack((linear_jacobian[:, free_dofs],
                              angular_jacobian[:, free_dofs]))
    payload_twist = free_jacobian @ data.qvel[free_dofs]
    payload_wrench = np.linalg.solve(free_jacobian.T,
                                     data.qfrc_constraint[free_dofs])
    rotation = np.asarray(data.xmat[payload]).reshape(3, 3)
    inertia_world = rotation @ inertia @ rotation.T
    payload_momentum = np.r_[mass*payload_twist[:3],
                              inertia_world @ payload_twist[3:]]
    payload_remainder = np.r_[-mass*np.asarray(model.opt.gravity), np.zeros(3)]
    momentum[:, 7, 6:] = payload_momentum
    remainder[:, 7, 6:] = payload_remainder
    applied_force = np.concatenate((robot_applied,
                                    np.broadcast_to(payload_wrench, (2, 6))), axis=1)
    return Observation(momentum, remainder, applied_force, robot_applied,
                       payload_wrench, payload_twist)
