"""JAS torque law and an explicitly selected boundary-layer engineering mode."""
from dataclasses import dataclass,asdict
import numpy as np
from .barrier import transform,terminal_surface
from .reference import desired,envelope
from .dynamics import coupled_bases
from .learning import CompositeLearner


@dataclass
class Settings:
    chi: tuple=(20.,20.,20.,8.,8.,8.)
    ell: tuple=(50.,50.,50.,20.,20.,20.)
    lam1: float=.8
    lam2: float=.8
    alpha: float=.5
    beta: float=1.5
    delta: float=.0005
    ka: tuple=(6.,6.,6.,.02,.02,.02)
    k2: tuple=(3.,3.,3.,.01,.01,.01)
    k3: tuple=(1.,1.,1.,.005,.005,.005)
    upsilon1: float=1.5
    upsilon2: float=.5
    prior_lower: float=.85
    prior_upper: float=1.15
    filter_rate: float=8.
    window: float=2.
    omega: float=1e-6
    adaptation_gain: float=.0005
    gamma: float=.2
    internal_force: float=5.
    rho_hat: float=1e-8
    robust_mode: str='paper_sign'
    robust_width: float=.05


class JASController:
    def __init__(self,kinematics,dynamics,center,payload_mass,payload_inertia,
                 hand_index,settings=None,nonmonotonic=False):
        self.kinematics=kinematics; self.dynamics=dynamics
        self.center=np.asarray(center); self.mass=payload_mass; self.inertia=payload_inertia
        self.cfg=settings or Settings(); self.nonmonotonic=nonmonotonic
        c=self.cfg
        if c.rho_hat<=0:
            raise ValueError('The paper requires a positive rho_hat bound.')
        if c.robust_mode not in ('paper_sign','boundary_layer'):
            raise ValueError('Unknown robust feedback mode.')
        if c.robust_width<=0:
            raise ValueError('Boundary-layer width must be positive.')
        self.learner=CompositeLearner(filter_rate=c.filter_rate,window=c.window,
                                     omega=c.omega,gain=c.adaptation_gain,gamma=c.gamma,
                                     lower=c.prior_lower,upper=c.prior_upper)
        self.radius=np.full(8,(c.prior_upper-c.prior_lower)/2)
        self.previous_applied=None
        self.fd=np.array([0.,(-1 if hand_index==0 else 1)*c.internal_force,0.,0.,0.,0.])

    def command(self,t,q,qd):
        c=self.cfg
        x,j,jdot=self.kinematics.evaluate(q,qd)
        xd=j@qd
        ref,dref,ddref=desired(t,self.center)
        bs=transform(x-ref,xd-dref,envelope(t,np.array(c.chi),np.array(c.ell),self.nonmonotonic),
                     np.array(c.chi),np.array(c.ell))
        surface,xir,ddxir=terminal_surface(bs.xi,bs.xidot,c.lam1,c.lam2,c.alpha,c.beta,c.delta)
        jinv=np.linalg.inv(j)
        a=jinv/bs.a
        adot=-jinv@jdot@a-a*(bs.adot/bs.a)
        known=bs.a*dref-bs.b
        v=a@known
        vdot=adot@known+a@(bs.adot*dref+bs.a*ddref-bs.bdot)
        qr=a@xir+v
        qrr=adot@xir+a@ddxir+vdot
        rb=self.dynamics.evaluate(q,qd)
        db,cb,wb=coupled_bases(rb,j,jdot,x,xd,self.mass,self.inertia)
        wb[6]=.05*qd
        regressor=np.stack([a.T@(db[k]@qrr+cb[k]@qr+wb[k]) for k in range(8)])
        w=regressor@surface
        # Only paper_sign is equation (kongzhiqi). The optional saturation
        # keeps the prior box intact but changes its ideal reaching proof.
        switching=(np.sign(w) if c.robust_mode=='paper_sign'
                   else np.clip(w/c.robust_width,-1.,1.))
        robust=regressor.T@(2*self.radius*switching)
        # Property 3 takes a positive bound even for constant shares (L_i=0).
        # Retain its prescribed local K_b = rho_i I from (hatrho).
        rho=c.rho_hat*np.linalg.norm(a,ord=2)**2
        task=(regressor.T@self.learner.theta-(np.asarray(c.ka)+rho)*surface
              -np.asarray(c.k2)*np.sign(surface)*np.abs(surface)**c.upsilon1
              -np.asarray(c.k3)*np.sign(surface)*np.abs(surface)**c.upsilon2-robust)
        torque=j.T@(bs.a*task-self.fd)
        momentum=np.stack([a.T@db[k]@qd for k in range(8)])
        remainder=np.stack([a.T@(wb[k]-cb[k].T@qd)-adot.T@db[k]@qd for k in range(8)])
        info=dict(x=x,xd=xd,reference=ref,error=x-ref,jacobian=j,
                  surface=surface,xi=bs.xi,barrier_margin=bs.margin,
                  upper=bs.upper,lower=bs.lower,regressor=regressor,
                  momentum=momentum,remainder=remainder,A=a,
                  fd=self.fd.copy(),sigma_min=np.linalg.svd(j,compute_uv=False)[-1],
                  rho=rho,
                  robust_power=float(w@(2*self.radius*switching)))
        return torque,info

    def update(self,t,torque,info):
        held=torque if self.previous_applied is None else self.previous_applied
        endpoint=info['A'].T@(held+info['jacobian'].T@self.fd)
        right_limit=info['A'].T@(torque+info['jacobian'].T@self.fd)
        self.learner.advance(t,info['momentum'],info['remainder'],endpoint,
                             info['regressor'],info['surface'],
                             next_known_torque=right_limit)
        self.previous_applied=np.asarray(torque).copy()

    def configuration(self):
        return asdict(self.cfg)
