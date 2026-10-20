"""Numerical integration of the paper's ORIGINAL projected composite law.

The continuous equation remains Upsilon*Pr(-Y*S+gamma*(p-G*theta)).
Upsilon is a constant positive diagonal, never an online data inverse.
Backward Euler handles a stiff history term; its box variational inequality
is solved jointly, rather than clipping an unconstrained matrix solution.
"""
import numpy as np
from scipy.linalg import solve

from jas.learning import CompositeLearner


def box_quadratic(hessian, rhs, lower, upper, start):
    """Primal active-set solution of a strictly convex small box quadratic."""
    x = np.clip(np.asarray(start, dtype=float).copy(), lower, upper)
    active = np.zeros(len(x), dtype=int)
    active[x <= lower+1e-13] = -1
    active[x >= upper-1e-13] = 1
    # Judge each face in parameter units. A global |rhs| tolerance can hide
    # a large violation in a weakly scaled coordinate of this small QP.
    diagonal = np.diag(hessian)
    tolerance = 3e-13
    for _ in range(100):
        gradient = hessian@x-rhs
        free = np.flatnonzero(active == 0)
        direction = np.zeros_like(x)
        if len(free):
            direction[free] = solve(hessian[np.ix_(free, free)], -gradient[free],
                                    assume_a='pos', check_finite=False)
        if np.max(np.abs(direction)) < 3e-13:
            violation = np.where(active == -1, -gradient/diagonal,
                                 np.where(active == 1, gradient/diagonal, -np.inf))
            release = int(np.argmax(violation))
            if violation[release] <= tolerance:
                return x
            active[release] = 0
            continue
        alpha = 1.
        blocker = None
        for j in free:
            if direction[j] > 0:
                step = (upper[j]-x[j])/direction[j]
            elif direction[j] < 0:
                step = (lower[j]-x[j])/direction[j]
            else:
                continue
            if step < alpha:
                alpha, blocker = max(0., step), int(j)
        x += alpha*direction
        if blocker is not None:
            active[blocker] = 1 if direction[blocker] > 0 else -1
            x[blocker] = upper[blocker] if active[blocker] == 1 else lower[blocker]
    raise RuntimeError('Projected original learning integration did not satisfy its box KKT conditions.')


class ProjectedCompositeLearner(CompositeLearner):
    """Original filter/history and original law, with a box-implicit step."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gain = np.broadcast_to(np.asarray(self.gain, dtype=float), self.theta.shape).copy()
        if np.any(self.gain <= 0) or self.gamma <= 0:
            raise ValueError('Original Upsilon diagonal and gamma must be positive constants.')
        self.last_fixed_point_error = 0.
        self.maximum_fixed_point_error = 0.
        self.active_projection_samples = 0
        self.last_drive = np.zeros_like(self.theta)

    def advance(self, t, momentum, remainder, known_torque, regressor, surface,
                next_known_torque=None):
        right = known_torque if next_known_torque is None else next_known_torque
        if self.initial_m is None:
            if abs(t) > 1e-12:
                raise ValueError('Zero-state filter must initialize at t=0.')
            self.initial_m = momentum.copy()
            self.previous = (t, remainder-self.rate*momentum, right.copy())
            self.previous_gram = np.zeros_like(self.gram)
            self.previous_p = np.zeros_like(self.p)
            return
        last, old_source, old_torque = self.previous
        dt = t-last
        if dt <= 0:
            raise ValueError('Learning timestamps must strictly increase.')
        source = remainder-self.rate*momentum
        self.z = self.filter_step(self.z, old_source, source, dt)
        self.tauf = self.filter_step(self.tauf, old_torque, known_torque, dt)
        yf = self.rate*(momentum-np.exp(-self.rate*t)*self.initial_m)+self.z
        self.filtered_regressor = yf
        if self.frozen_time is None:
            gt, pt = yf@yf.T, yf@self.tauf
            dg, dp = .5*dt*(gt+self.previous_gram), .5*dt*(pt+self.previous_p)
            self.history.append((t, dg, dp))
            self.gram += dg
            self.p += dp
            while self.history and self.history[0][0] <= t-self.window:
                _, out_g, out_p = self.history.popleft()
                self.gram -= out_g
                self.p -= out_p
            self.gram = .5*(self.gram+self.gram.T)
            self.minimum_eigenvalue = float(np.linalg.eigvalsh(self.gram)[0])
            if t >= self.window and self.minimum_eigenvalue >= self.omega:
                self.frozen_time = float(t)
            self.previous_gram, self.previous_p = gt, pt
        old_theta = self.theta.copy()
        self.epsilon = self.p-self.gram@old_theta
        self.last_drive = regressor@surface
        diagonal = 1./(dt*self.gain)
        hessian = np.diag(diagonal)+self.gamma*self.gram
        rhs = diagonal*old_theta+self.gamma*self.p-self.last_drive
        self.theta = box_quadratic(hessian, rhs, self.lower, self.upper, old_theta)
        raw_step = old_theta+dt*self.gain*(
            -self.last_drive+self.gamma*(self.p-self.gram@self.theta))
        fixed_point = np.clip(raw_step, self.lower, self.upper)
        self.last_fixed_point_error = float(np.max(np.abs(fixed_point-self.theta)))
        self.maximum_fixed_point_error = max(self.maximum_fixed_point_error, self.last_fixed_point_error)
        if self.last_fixed_point_error > 2e-8:
            raise RuntimeError('Original projected update fixed-point tolerance exceeded.')
        self.active_projection_samples += int(np.any(np.abs(raw_step-fixed_point) > 1e-12))
        self.previous = (t, source.copy(), right.copy())
