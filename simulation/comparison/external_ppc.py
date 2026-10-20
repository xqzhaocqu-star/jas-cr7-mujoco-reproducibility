"""Verginis et al., arXiv:1710.11088v4, IV-B, equations (25)--(36).

An independent, dynamics-free, encoder-only complete cooperative controller.
No JAS error transform, sliding surface, learning law or controller is used.
See docs/external_ppc_baseline.md for the Euler/grasp/force conventions.
"""
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_EXTERNAL_SETTINGS = {
    'gs': .01,
    'gv': 10.,
    'rho_s_initial': (.08, .08, .08, .22, .20, .22),
    'rho_s_final': (.012, .010, .010, .045, .040, .035),
    'rho_s_rate': (.5,)*6,
    'rho_v_initial': None,
    'rho_v_final': (.4, .4, .4, .8, .8, .8),
    'rho_v_rate': (.5,)*6,
    'velocity_initial_slack': 1.,
    'desired_pitch_max': .35,
    # Designer load-sharing constants, unrelated to actual robot/object mass.
    'virtual_mass': 1.,
    'virtual_inertia': 1.,
    'grasp_positions_object': ((0., -.20, 0.), (0., .20, 0.)),
}


def _skew(v):
    a, b, c = np.asarray(v, dtype=float)
    return np.array([[0., -c, b], [c, 0., -a], [-b, a, 0.]])


def _twist_map(rpy):
    """B: [position rates; xyz-Euler rates] -> world geometric twist.

    The paper's representation Jacobian J_O is inv(B).
    """
    _, p, y = rpy
    cp, sp, cy, sy = np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    if abs(cp) < 1e-6:
        raise FloatingPointError('External PPC Euler representation is singular.')
    b = np.eye(6)
    b[3:, 3:] = np.array([[cy*cp, -sy, 0.], [sy*cp, cy, 0.], [-sp, 0., 1.]])
    return b


def _vector(value, name):
    a = np.asarray(value, dtype=float)
    if a.ndim == 0:
        a = np.repeat(a, 6)
    if a.shape != (6,) or not np.all(np.isfinite(a)) or np.min(a) <= 0:
        raise ValueError(name+' must contain six finite positive constants.')
    return a


def _log_transform(xi, stage):
    """Exact (29)/(30) or (34); never clip an out-of-domain state."""
    if not np.all(np.isfinite(xi)) or np.any(np.abs(xi) >= 1.):
        raise FloatingPointError('External PPC '+stage+' domain violated; max |xi|='
                                 +str(float(np.max(np.abs(xi)))))
    epsilon = np.log1p(xi)-np.log1p(-xi)
    return epsilon, 2./(1.-xi*xi)


class ExternalPPC:
    """Strict Section IV-B PPC, with Remark 5's known nullspace task.

    The dynamics/mass/inertia constructor arguments are interface placeholders;
    they are intentionally neither stored nor read. Only the local kinematic
    model, own joint position/velocity, reference and offline grasp geometry
    enter command(). ``update`` is a no-op: there is no learner or filter.
    """

    def __init__(self, kinematics, dynamics, center, payload_mass,
                 payload_inertia, hand_index, settings, nonmonotonic=False,
                 *, reference, external_settings=None):
        self.kinematics = kinematics
        self.center = np.asarray(center, dtype=float)
        self.cfg = settings
        self.hand_index = int(hand_index)
        if self.hand_index not in (0, 1):
            raise ValueError('This implementation is for two opposite CR7 grasps.')
        self.nonmonotonic = bool(nonmonotonic)
        self.reference = reference
        c = deepcopy(DEFAULT_EXTERNAL_SETTINGS)
        if external_settings is not None:
            unknown = set(external_settings)-set(c)
            if unknown:
                raise ValueError('Unknown external PPC settings: '+str(sorted(unknown)))
            c.update(deepcopy(external_settings))
        self.external_settings = c
        self.gs, self.gv = float(c['gs']), float(c['gv'])
        if not (np.isfinite(self.gs) and self.gs > 0 and np.isfinite(self.gv) and self.gv > 0):
            raise ValueError('The published scalar gs and gv must be positive.')
        self.rho_s_initial = _vector(c['rho_s_initial'], 'rho_s_initial')
        self.rho_s_final = _vector(c['rho_s_final'], 'rho_s_final')
        self.rho_s_rate = _vector(c['rho_s_rate'], 'rho_s_rate')
        self.rho_v_final = _vector(c['rho_v_final'], 'rho_v_final')
        self.rho_v_rate = _vector(c['rho_v_rate'], 'rho_v_rate')
        self.rho_v_initial = (None if c['rho_v_initial'] is None else
                              _vector(c['rho_v_initial'], 'rho_v_initial'))
        self.velocity_initial_slack = float(c['velocity_initial_slack'])
        self.desired_pitch_max = float(c['desired_pitch_max'])
        self.virtual_mass = float(c['virtual_mass'])
        self.virtual_inertia = float(c['virtual_inertia'])
        constants = (self.velocity_initial_slack, self.desired_pitch_max,
                     self.virtual_mass, self.virtual_inertia)
        if not np.all(np.isfinite(constants)) or min(constants) <= 0:
            raise ValueError('Positive finite PPC/slack/load-sharing constants required.')
        if np.any(self.rho_s_initial <= self.rho_s_final):
            raise ValueError('Every native pose funnel must decrease to a positive limit.')
        if self.rho_s_initial[4]+self.desired_pitch_max >= np.pi/2:
            raise ValueError('Paper pitch assumption theta_star+bar_theta_d < pi/2 fails.')
        self.grasp_positions_object = np.asarray(c['grasp_positions_object'], dtype=float)
        if self.grasp_positions_object.shape != (2, 3) or not np.all(np.isfinite(self.grasp_positions_object)):
            raise ValueError('Known object-to-palm offsets must have shape (2,3).')
        if not np.allclose(self.grasp_positions_object.sum(axis=0), 0., atol=1e-12):
            raise ValueError('Equal virtual masses require central opposite grasp positions.')
        # Verify offline geometry against this arm's encoder reconstruction.
        own_d_object = np.asarray(kinematics.rotation).T@np.asarray(kinematics.offset)
        if not np.allclose(own_d_object, -self.grasp_positions_object[self.hand_index], atol=1e-8):
            raise ValueError('External PPC grasp geometry does not match local kinematics.')
        internal_force = float(settings.internal_force)
        if not np.isfinite(internal_force):
            raise ValueError('Desired internal force must be finite.')
        # Runner convention: environment-on-robot Euler wrench at object CoM.
        # The actuator wrench added by Remark 5 is its NEGATIVE.
        self.fd = np.array([0., (-1. if self.hand_index == 0 else 1.)*internal_force,
                            0., 0., 0., 0.])
        self._first_time = None

    @staticmethod
    def _envelope(t, initial, final, rate):
        return (initial-final)*np.exp(-rate*t)+final

    def _grasp_matrices(self, rpy):
        """Paper (9), (20), with positive equal designer masses/inertias.

        For the central opposite pair and isotropic designer inertias the
        printed order J_O*^{-1} S(d_i) obeys G.T @ G_M_plus == I exactly.
        """
        rotation = Rotation.from_euler('xyz', rpy).as_matrix()
        d = -(rotation@self.grasp_positions_object.T).T  # palm -> object, world
        skews = [_skew(di) for di in d]
        inert = 2*self.virtual_inertia*np.eye(3)
        inert -= self.virtual_mass*sum((s@s for s in skews), start=np.zeros((3, 3)))
        inert_inv = np.linalg.inv(inert)
        grasps, shares = [], []
        for s in skews:
            grasp = np.eye(6)
            grasp[:3, 3:] = s
            share = np.zeros((6, 6))
            share[:3, :3] = .5*np.eye(3)
            share[:3, 3:] = self.virtual_mass*inert_inv@s
            share[3:, 3:] = self.virtual_inertia*inert_inv
            grasps.append(grasp)
            shares.append(share)
        return np.stack(grasps), np.stack(shares)

    def command(self, t, q, qd):
        t = float(t)
        if not np.isfinite(t) or t < 0:
            raise ValueError('External PPC requires nonnegative finite time.')
        qd = np.asarray(qd, dtype=float)
        # Only FK/Jacobian, not a finite-difference acceleration or dynamics.
        x, j = self.kinematics.pose_jacobian(q)
        xd = j@qd
        ref, _, _ = self.reference(t, self.center)
        ref = np.asarray(ref, dtype=float)
        if abs(ref[4]) > self.desired_pitch_max+1e-12:
            raise FloatingPointError('Reference exceeds the configured paper pitch bound.')
        b = _twist_map(x[3:])
        velocity = b@xd
        error = x-ref
        rho_s = self._envelope(t, self.rho_s_initial, self.rho_s_final, self.rho_s_rate)
        xi_s = error/rho_s
        epsilon_s, r_s = _log_transform(xi_s, 'pose')
        # (31), algebraic and WITHOUT desired-velocity feedforward.
        v_reference = -self.gs*(b@(r_s*epsilon_s/rho_s))
        velocity_error = velocity-v_reference
        if self._first_time is None:
            if t > 1e-12:
                raise ValueError('PPC initialization must be the t=0 encoder sample.')
            if self.rho_v_initial is None:
                # Step II-a: one norm-based bound for all six components.
                self.rho_v_initial = np.full(6, np.linalg.norm(velocity_error)+self.velocity_initial_slack)
            if np.any(self.rho_v_initial <= self.rho_v_final):
                raise ValueError('rho_v_initial must exceed rho_v_final in every component.')
            self._first_time = t
        rho_v = self._envelope(t, self.rho_v_initial, self.rho_v_final, self.rho_v_rate)
        xi_v = velocity_error/rho_v
        epsilon_v, r_v = _log_transform(xi_v, 'velocity')
        object_feedback = -self.gv*r_v*epsilon_v/rho_v
        grasps, shares = self._grasp_matrices(x[3:])
        g = np.concatenate(grasps, axis=0)
        gmplus = np.concatenate(shares, axis=0)
        # Remark 5: a known desired wrench in null(G.T), requiring no sensing.
        # Opposite Euler-CoM fd vectors sum to zero. Convert to physical palm
        # actuator wrenches before applying the paper's nullspace projector.
        # Same global left/right order in both independent local controllers.
        left_fd = np.array([0., -float(self.cfg.internal_force), 0., 0., 0., 0.])
        desired = np.concatenate([np.linalg.solve(grasps[h].T,
                                   np.linalg.solve(b.T, -(left_fd if h == 0 else -left_fd))) for h in range(2)])
        internal = ((np.eye(12)-gmplus@g.T)@desired).reshape(2, 6)
        h = self.hand_index
        palm_wrench = shares[h]@object_feedback+internal[h]
        palm_jacobian = grasps[h]@b@j
        torque = palm_jacobian.T@palm_wrench
        nan86 = np.full((8, 6), np.nan)
        info = dict(x=x, xd=xd, reference=ref, error=error, jacobian=j,
                    surface=velocity_error, xi=xi_s,
                    barrier_margin=1.-np.abs(xi_s), upper=rho_s, lower=-rho_s,
                    regressor=nan86.copy(), momentum=nan86.copy(), remainder=nan86.copy(),
                    A=np.full((6, 6), np.nan), fd=self.fd.copy(),
                    sigma_min=float(np.linalg.svd(j, compute_uv=False)[-1]),
                    rho=np.nan, robust_power=np.nan,
                    ppc_rho_s=rho_s, ppc_rho_v=rho_v,
                    ppc_xi_s=xi_s, ppc_xi_v=xi_v,
                    ppc_epsilon_s=epsilon_s, ppc_epsilon_v=epsilon_v,
                    ppc_v_reference=v_reference,
                    ppc_object_velocity=velocity, ppc_object_feedback=object_feedback,
                    ppc_load_allocation=shares[h], ppc_grasp_map=grasps[h],
                    ppc_palm_wrench=palm_wrench, ppc_internal_palm_wrench=internal[h],
                    ppc_allocation_identity_error=float(np.linalg.norm(g.T@gmplus-np.eye(6))),
                    ppc_internal_null_error=float(np.linalg.norm(g.T@internal.reshape(12))))
        return torque, info

    def update(self, t, applied, info):
        """No history, dynamic filter, force feedback or adaptation exists."""
        return None

    def configuration(self):
        c = deepcopy(self.external_settings)
        c['resolved_rho_v_initial'] = (None if self.rho_v_initial is None else self.rho_v_initial.tolist())
        return dict(controller='Verginis-Mastellaro-Dimarogonas PPC',
                    source='https://arxiv.org/html/1710.11088v4',
                    source_section='IV-B equations (25)-(36), equation (20), Remark 5',
                    has_adaptation=False, has_dynamic_filter=False,
                    measurements='own joint position and velocity only',
                    native_bounds='symmetric monotonic exponential; distinct from common evaluation bounds',
                    internal_force_adaptation='Remark 5 known nullspace task; same +/-5N target',
                    not_applicable=['JAS S', 'JAS xi', 'regressor', 'A', 'theta', 'IE'],
                    settings=c)
