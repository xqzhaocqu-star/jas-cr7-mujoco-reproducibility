"""Measured-force, preconditioned historical identification extension.

This is explicitly NOT equation (gengxin): after a valid IE window it uses
only the historical prediction, with a data-derived positive definite metric.
No true parameters, measured acceleration, or guessed regression targets enter.
"""
from collections import deque

import numpy as np
from scipy.linalg import cho_factor, cho_solve


class HistoricalIdentifier:
    def __init__(self, size=8, outputs=12, filter_rate=8., window=6.,
                 normalized_threshold=1e-5, rate=1.5, lower=.85, upper=1.15):
        self.theta = np.full(size, .9)
        self.lower = np.full(size, lower)
        self.upper = np.full(size, upper)
        self.filter_rate, self.window = float(filter_rate), float(window)
        self.normalized_threshold, self.rate = float(normalized_threshold), float(rate)
        self.z = np.zeros((size, outputs))
        self.force_f = np.zeros(outputs)
        self.regressor_f = np.zeros_like(self.z)
        self.gram = np.zeros((size, size))
        self.p = np.zeros(size)
        self.history = deque()
        self.previous = None
        self.initial_momentum = None
        self.previous_integrands = None
        self.frozen_time = None
        self.theta_at_freeze = None
        self.data_target = None
        self.scale = None
        self.factor = None
        self.minimum_eigenvalue = 0.
        self.normalized_minimum_eigenvalue = 0.
        self.rejected_outside_prior = 0
        self.projection_count = 0
        self.epsilon = np.zeros(size)

    def _filter(self, state, old, new, dt):
        x = self.filter_rate*dt
        decay = np.exp(-x)
        ramp = 1+np.expm1(-x)/x
        return decay*state-np.expm1(-x)*old+ramp*(new-old)

    def advance(self, t, momentum, remainder, force_left, force_right=None):
        right = force_left if force_right is None else force_right
        source = remainder-self.filter_rate*momentum
        if self.initial_momentum is None:
            if abs(t) > 1e-12:
                raise ValueError('The zero-state filter must start at t=0.')
            self.initial_momentum = momentum.copy()
            self.previous = (float(t), source.copy(), right.copy())
            self.previous_integrands = (self.gram.copy(), self.p.copy())
            return
        old_t, old_source, old_force = self.previous
        dt = t-old_t
        if dt <= 0:
            raise ValueError('Identification timestamps must increase.')
        self.z = self._filter(self.z, old_source, source, dt)
        self.force_f = self._filter(self.force_f, old_force, force_left, dt)
        self.regressor_f = (self.filter_rate*(momentum-np.exp(-self.filter_rate*t)
                                            *self.initial_momentum)+self.z)
        yf = self.regressor_f
        integrands = (yf@yf.T, yf@self.force_f)
        was_frozen = self.frozen_time is not None
        if not was_frozen:
            dg = .5*dt*(integrands[0]+self.previous_integrands[0])
            dp = .5*dt*(integrands[1]+self.previous_integrands[1])
            self.history.append((float(t), dg, dp))
            self.gram += dg
            self.p += dp
            while self.history and self.history[0][0] <= t-self.window:
                _, first_g, first_p = self.history.popleft()
                self.gram -= first_g
                self.p -= first_p
            self.gram = .5*(self.gram+self.gram.T)
            self.minimum_eigenvalue = float(np.linalg.eigvalsh(self.gram)[0])
            diagonal = np.diag(self.gram)
            if np.all(diagonal > 1e-14):
                scale = 1/np.sqrt(diagonal)
                normalized = scale[:, None]*self.gram*scale[None, :]
                self.normalized_minimum_eigenvalue = float(np.linalg.eigvalsh(normalized)[0])
                if (t >= self.window and self.minimum_eigenvalue > 0
                        and self.normalized_minimum_eigenvalue >= self.normalized_threshold):
                    factor = cho_factor(normalized, lower=True, check_finite=True)
                    target = scale*cho_solve(factor, scale*self.p)
                    if np.all((target >= self.lower) & (target <= self.upper)):
                        self.frozen_time = float(t)
                        self.theta_at_freeze = self.theta.copy()
                        self.data_target = target.copy()
                        self.scale, self.factor = scale, factor
                        self.history.clear()
                    else:
                        self.rejected_outside_prior += 1
            self.previous_integrands = (integrands[0].copy(), integrands[1].copy())
        self.epsilon = self.p-self.gram@self.theta
        if was_frozen:
            # Exact integration of dot(theta)=rate*G^-1*(p-G theta).
            # The driving prediction comes entirely from measured history.
            direction = self.scale*cho_solve(self.factor, self.scale*self.epsilon)
            trial = self.theta-np.expm1(-self.rate*dt)*direction
            self.projection_count += int(np.any((trial < self.lower) | (trial > self.upper)))
            self.theta = np.clip(trial, self.lower, self.upper)
        self.previous = (float(t), source.copy(), right.copy())
