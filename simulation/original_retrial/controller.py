"""The paper's full torque formula and full composite learning law.

Only the prescribed task, constant gains, and numerical integration are
configurable here. There is no force feedback, new regressor, data inverse
learning gain, history-target gate, or replacement learning channel.
"""
from dataclasses import dataclass
import numpy as np

from jas.controller import JASController, Settings
from jas.barrier import transform, terminal_surface
from jas.reference import envelope
from jas.dynamics import coupled_bases
from jas.learning import CompositeLearner
from .learning import ProjectedCompositeLearner


@dataclass
class OriginalSettings(Settings):
    adaptation_gain: tuple = (.0005,)*8
    learning_integrator: str = 'projected_implicit'
    excitation_profile: str = 'moderate'
    excitation_duration: float = 20.


class OriginalController(JASController):
    def __init__(self, *args, reference, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference = reference
        c = self.cfg
        if c.robust_mode != 'paper_sign':
            raise ValueError('The original-method retrial requires the paper sign formula.')
        if len(c.adaptation_gain) != 8 or np.min(c.adaptation_gain) <= 0:
            raise ValueError('Upsilon must be an eight-dimensional constant positive diagonal.')
        classes = {'projected_implicit': ProjectedCompositeLearner, 'explicit_euler': CompositeLearner}
        cls = classes[c.learning_integrator]
        self.learner = cls(filter_rate=c.filter_rate, window=c.window, omega=c.omega,
                           gain=np.asarray(c.adaptation_gain), gamma=c.gamma,
                           lower=c.prior_lower, upper=c.prior_upper)

    def command(self, t, q, qd):
        # This is equation (kongzhiqi) with the same reference-velocity
        # construction and original local momentum/remainder as jas/controller.
        c = self.cfg
        x, j, jdot = self.kinematics.evaluate(q, qd)
        xd = j@qd
        ref, dref, ddref = self.reference(t, self.center)
        bs = transform(x-ref, xd-dref,
                       envelope(t, np.array(c.chi), np.array(c.ell), self.nonmonotonic),
                       np.array(c.chi), np.array(c.ell))
        surface, xir, ddxir = terminal_surface(bs.xi, bs.xidot, c.lam1, c.lam2,
                                             c.alpha, c.beta, c.delta)
        jinv = np.linalg.inv(j)
        a = jinv/bs.a
        adot = -jinv@jdot@a-a*(bs.adot/bs.a)
        known = bs.a*dref-bs.b
        v = a@known
        vdot = adot@known+a@(bs.adot*dref+bs.a*ddref-bs.bdot)
        qr = a@xir+v
        qrr = adot@xir+a@ddxir+vdot
        robot = self.dynamics.evaluate(q, qd)
        db, cb, wb = coupled_bases(robot, j, jdot, x, xd, self.mass, self.inertia)
        wb[6] = .05*qd
        regressor = np.stack([a.T@(db[k]@qrr+cb[k]@qr+wb[k]) for k in range(8)])
        w = regressor@surface
        robust = regressor.T@(2*self.radius*np.sign(w))
        rho = c.rho_hat*np.linalg.norm(a, ord=2)**2
        task = (regressor.T@self.learner.theta-(np.asarray(c.ka)+rho)*surface
                -np.asarray(c.k2)*np.sign(surface)*np.abs(surface)**c.upsilon1
                -np.asarray(c.k3)*np.sign(surface)*np.abs(surface)**c.upsilon2-robust)
        torque = j.T@(bs.a*task-self.fd)
        momentum = np.stack([a.T@db[k]@qd for k in range(8)])
        remainder = np.stack([a.T@(wb[k]-cb[k].T@qd)-adot.T@db[k]@qd for k in range(8)])
        return torque, dict(x=x, xd=xd, reference=ref, error=x-ref, jacobian=j,
                            surface=surface, xi=bs.xi, barrier_margin=bs.margin,
                            upper=bs.upper, lower=bs.lower, regressor=regressor,
                            momentum=momentum, remainder=remainder, A=a,
                            fd=self.fd.copy(), sigma_min=np.linalg.svd(j, compute_uv=False)[-1],
                            rho=rho, robust_power=float(w@(2*self.radius*np.sign(w))))
