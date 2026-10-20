"""Original JAS motion formula with a declared reference and identifier.

Only the source of theta_hat and reference trajectory differ from JASController.
The original implementation and its validated result manifests remain intact.
"""
from dataclasses import dataclass

import numpy as np

from jas.controller import JASController, Settings
from jas.barrier import transform, terminal_surface
from jas.reference import envelope
from jas.dynamics import coupled_bases


@dataclass
class IdentificationSettings(Settings):
    identification_rate: float = 1.5
    identification_window: float = 6.
    identification_threshold: float = 1e-5
    excitation_profile: str = 'baseline'
    excitation_duration: float = 16.


class IdentificationController(JASController):
    def __init__(self, *args, reference, identifier, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference = reference
        self.learner = identifier

    def command(self, t, q, qd):
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
        qr, qrr = a@xir+v, adot@xir+a@ddxir+vdot
        robot_bases = self.dynamics.evaluate(q, qd)
        db, cb, wb = coupled_bases(robot_bases, j, jdot, x, xd, self.mass, self.inertia)
        wb[6] = .05*qd
        regressor = np.stack([a.T@(db[k]@qrr+cb[k]@qr+wb[k]) for k in range(8)])
        w = regressor@surface
        switching = np.sign(w) if c.robust_mode == 'paper_sign' else np.clip(w/c.robust_width, -1., 1.)
        robust = regressor.T@(2*self.radius*switching)
        rho = c.rho_hat*np.linalg.norm(a, ord=2)**2
        task = (regressor.T@self.learner.theta-(np.asarray(c.ka)+rho)*surface
                -np.asarray(c.k2)*np.sign(surface)*np.abs(surface)**c.upsilon1
                -np.asarray(c.k3)*np.sign(surface)*np.abs(surface)**c.upsilon2-robust)
        torque = j.T@(bs.a*task-self.fd)
        return torque, dict(x=x, xd=xd, reference=ref, error=x-ref, jacobian=j,
                            surface=surface, xi=bs.xi, barrier_margin=bs.margin,
                            upper=bs.upper, lower=bs.lower, robot_bases=robot_bases,
                            sigma_min=np.linalg.svd(j, compute_uv=False)[-1])
