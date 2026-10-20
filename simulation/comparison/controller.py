"""One-module-at-a-time ablations of the original JAS implementation.

No unknown dynamics, constraint reactions or measured acceleration are used.
The proposed path deliberately calls OriginalController.command unchanged.
"""
from dataclasses import dataclass
import numpy as np

from jas.barrier import BarrierState, transform, terminal_surface
from jas.reference import envelope
from jas.dynamics import coupled_bases
from original_retrial.controller import OriginalController, OriginalSettings


METHODS = ('proposed', 'traditional_bf', 'linear_surface', 'linear_reaching',
           'no_history', 'external_ppc')


@dataclass
class ComparisonSettings(OriginalSettings):
    method: str = 'proposed'
    linear_surface_gain: tuple = (8.,)*6
    evaluation_initial_upper: tuple = (.08, .07, .07, .22, .20, .22)
    evaluation_initial_lower: tuple = (.09, .07, .08, .25, .20, .23)
    evaluation_transition_s: float = 1.
    log_bf_scale: tuple = ()


def common_bounds(t, settings, nonmonotonic=False):
    """Finite C2 assessment bounds; match original bounds from t=1 s.

Original JAS still uses mu(0)=ell. The first-second common finite envelope
is an additional measured requirement, not its original invariance theorem.
Returns positive U,L and their first and second time derivatives.
"""
    c = settings
    chi, ell = np.asarray(c.chi), np.asarray(c.ell)
    u, v, du, dv, ddu, ddv = envelope(t, chi, ell, nonmonotonic)
    if t < c.evaluation_transition_s:
        z = max(t, 0.)/c.evaluation_transition_s
        q = 10*z**3-15*z**4+6*z**5
        dq = (30*z**2-60*z**3+30*z**4)/c.evaluation_transition_s
        ddq = (60*z-180*z**2+120*z**3)/c.evaluation_transition_s**2
    else:
        q, dq, ddq = 1., 0., 0.
    result = []
    for mu, dmu, ddmu, initial in (
            (u, du, ddu, c.evaluation_initial_upper),
            (v, dv, ddv, c.evaluation_initial_lower)):
        k0 = np.tanh(chi*np.asarray(initial))
        k = k0+(1-k0)*q
        dk, ddk = (1-k0)*dq, (1-k0)*ddq
        s, ds = k*mu/ell, (dk*mu+k*dmu)/ell
        dds = (ddk*mu+2*dk*dmu+k*ddmu)/ell
        gap = 1-s*s
        result.append((np.arctanh(s)/chi, ds/(chi*gap),
                       dds/(chi*gap)+2*s*ds*ds/(chi*gap*gap)))
    (upper, du, ddu), (lower, dl, ddl) = result
    return upper, lower, du, dl, ddu, ddl


def conventional_log_barrier(error, velocity, bounds, scale):
    """Classical asymmetric logarithmic error transformation, exact derivatives.

xi=c*log((1+e/L)/(1-e/U)), -L<e<U. This comparator replaces
the proposed stretch/rational transform; it is not a published full method.
"""
    u, l, du, dl, ddu, ddl = bounds
    e, ed, scale = np.asarray(error), np.asarray(velocity), np.asarray(scale)
    gm, gp = e+l, u-e
    if np.any(gm <= 0) or np.any(gp <= 0):
        raise FloatingPointError('Conventional log-BF domain violated; no error clipping.')
    xi = scale*(np.log1p(e/l)-np.log1p(-e/u))
    a = scale*(1/gm+1/gp)
    b = scale*(dl/gm-du/gp-dl/l+du/u)
    adot = scale*(-(ed+dl)/gm**2-(du-ed)/gp**2)
    bdot = scale*(ddl/gm-dl*(ed+dl)/gm**2-ddu/gp+du*(du-ed)/gp**2
                  -ddl/l+(dl/l)**2+ddu/u-(du/u)**2)
    margin = np.minimum(gm/l, gp/u)
    return BarrierState(xi, a*ed+b, a, b, adot, bdot, e, margin, u, -l)


def log_scale(settings):
    if settings.log_bf_scale:
        return np.asarray(settings.log_bf_scale)
    # Fixed once at e=0,t=0, nominal task. No plant/truth/data dependence.
    u, l = np.asarray(settings.evaluation_initial_upper), np.asarray(settings.evaluation_initial_lower)
    return (np.asarray(settings.chi)/np.asarray(settings.ell))/(1/u+1/l)


class ComparisonController(OriginalController):
    def __init__(self, *args, reference, **kwargs):
        super().__init__(*args, reference=reference, **kwargs)
        c = self.cfg
        if c.method not in METHODS[:-1]:
            raise ValueError('External complete controller has a separate implementation.')
        if c.learning_integrator != 'explicit_euler':
            raise ValueError('Comparisons use the same original projected Euler discretization.')
        if np.asarray(c.linear_surface_gain).shape != (6,) or np.min(c.linear_surface_gain) <= 0 or c.evaluation_transition_s <= 0:
            raise ValueError('Positive comparison constants required.')
        if c.method == 'no_history':
            # Explicitly labelled ablation: keep filters, IE diagnostics and
            # -YS/projection identical; delete only gamma*epsilon from update.
            self.learner.gamma = 0.

    def command(self, t, q, qd):
        if self.cfg.method in ('proposed', 'no_history'):
            return super().command(t, q, qd)
        c = self.cfg
        x, j, jdot = self.kinematics.evaluate(q, qd)
        xd = j@qd
        ref, dref, ddref = self.reference(t, self.center)
        if c.method == 'traditional_bf':
            bs = conventional_log_barrier(x-ref, xd-dref,
                                           common_bounds(t, c, self.nonmonotonic), log_scale(c))
        else:
            bs = transform(x-ref, xd-dref,
                           envelope(t, np.array(c.chi), np.array(c.ell), self.nonmonotonic),
                           np.array(c.chi), np.array(c.ell))
        if c.method == 'linear_surface':
            xir = -np.asarray(c.linear_surface_gain)*bs.xi
            ddxir = -np.asarray(c.linear_surface_gain)*bs.xidot
            surface = bs.xidot-xir
        else:
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
        task = regressor.T@self.learner.theta-(np.asarray(c.ka)+rho)*surface-robust
        if c.method != 'linear_reaching':
            task -= (np.asarray(c.k2)*np.sign(surface)*np.abs(surface)**c.upsilon1
                     +np.asarray(c.k3)*np.sign(surface)*np.abs(surface)**c.upsilon2)
        torque = j.T@(bs.a*task-self.fd)
        momentum = np.stack([a.T@db[k]@qd for k in range(8)])
        remainder = np.stack([a.T@(wb[k]-cb[k].T@qd)-adot.T@db[k]@qd for k in range(8)])
        return torque, dict(x=x, xd=xd, reference=ref, error=x-ref, jacobian=j,
                            surface=surface, xi=bs.xi, barrier_margin=bs.margin,
                            upper=bs.upper, lower=bs.lower, regressor=regressor,
                            momentum=momentum, remainder=remainder, A=a,
                            fd=self.fd.copy(), sigma_min=np.linalg.svd(j, compute_uv=False)[-1],
                            rho=rho, robust_power=float(w@(2*self.radius*np.sign(w))))
