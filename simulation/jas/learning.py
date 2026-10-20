"""Local acceleration-free regression, moving window and projected update."""
from collections import deque
import numpy as np


class CompositeLearner:
    def __init__(self,size=8,filter_rate=8.,window=2.,omega=1e-6,
                 gain=.0005,gamma=.2,lower=.85,upper=1.15):
        self.theta=np.full(size,.9)
        self.lower=np.full(size,lower); self.upper=np.full(size,upper)
        self.rate=filter_rate; self.window=window; self.omega=omega
        self.gain=gain; self.gamma=gamma
        self.z=np.zeros((size,6)); self.tauf=np.zeros(6)
        self.gram=np.zeros((size,size)); self.p=np.zeros(size)
        self.history=deque()
        self.initial_m=None; self.previous=None
        self.previous_gram=None; self.previous_p=None
        self.frozen_time=None; self.minimum_eigenvalue=0.
        self.epsilon=np.zeros(size); self.filtered_regressor=np.zeros((size,6))
        self.numerical_projection_count=0

    def filter_step(self,z,old,new,dt):
        decay=np.exp(-self.rate*dt)
        ramp=1-(1-decay)/(self.rate*dt)
        return decay*z+(1-decay)*old+ramp*(new-old)

    def advance(self,t,momentum,remainder,known_torque,regressor,surface,
                next_known_torque=None):
        # known_torque is the left limit at the END of the elapsed physical
        # interval. A newly issued motor command is only the right limit for
        # the NEXT interval; do not interpolate across its instantaneous jump.
        right_torque=known_torque if next_known_torque is None else next_known_torque
        if self.initial_m is None:
            if abs(t)>1e-12:
                raise ValueError('Zero-state momentum filter must initialize at t=0.')
            self.initial_m=momentum.copy()
            self.previous=(t,remainder-self.rate*momentum,right_torque.copy())
            self.previous_gram=np.zeros_like(self.gram)
            self.previous_p=np.zeros_like(self.p)
            return
        last,old_source,old_torque=self.previous
        dt=t-last
        if dt <= 0:
            raise ValueError('Learning timestamps must strictly increase.')
        source=remainder-self.rate*momentum
        self.z=self.filter_step(self.z,old_source,source,dt)
        self.tauf=self.filter_step(self.tauf,old_torque,known_torque,dt)
        yf=self.rate*(momentum-np.exp(-self.rate*t)*self.initial_m)+self.z
        self.filtered_regressor=yf
        if self.frozen_time is None:
            gt=yf@yf.T; pt=yf@self.tauf
            dg=.5*dt*(gt+self.previous_gram)
            dp=.5*dt*(pt+self.previous_p)
            self.history.append((t,dg,dp))
            self.gram+=dg; self.p+=dp
            while self.history and self.history[0][0] <= t-self.window:
                _,out_g,out_p=self.history.popleft()
                self.gram-=out_g; self.p-=out_p
            self.gram=.5*(self.gram+self.gram.T)
            self.minimum_eigenvalue=float(np.linalg.eigvalsh(self.gram)[0])
            if t>=self.window and self.minimum_eigenvalue>=self.omega:
                self.frozen_time=float(t)
            self.previous_gram=gt; self.previous_p=pt
        self.epsilon=self.p-self.gram@self.theta
        direction=-regressor@surface+self.gamma*self.epsilon
        direction=np.where(((self.theta>=self.upper)&(direction>0))
                           |((self.theta<=self.lower)&(direction<0)),0.,direction)
        # Euclidean projected Euler is the discrete counterpart of (pr).
        trial=self.theta+dt*self.gain*direction
        self.numerical_projection_count+=int(np.any((trial<self.lower)|(trial>self.upper)))
        self.theta=np.clip(trial,self.lower,self.upper)
        self.previous=(t,source.copy(),right_torque.copy())
