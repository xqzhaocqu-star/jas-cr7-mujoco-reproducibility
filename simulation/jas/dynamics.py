"""Exact linear density bases of the nominal simulation robot.

The six body COM locations and normalized inertia shapes are known. Each
body's mass and inertia are scaled together by one uncertain coefficient.
This is a restricted, explicitly declared physical uncertainty model.
"""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
from .geometry import angular_map


def christoffel(derivative, velocity):
    # derivative[k,i,j] = partial M[i,j] / partial q[k]
    return .5*(np.einsum('kij,k->ij',derivative,velocity)
               +np.einsum('jik,k->ij',derivative,velocity)
               -np.einsum('ijk,k->ij',derivative,velocity))


class LinkDynamics:
    def __init__(self,path):
        self.model=mujoco.MjModel.from_xml_path(str(path))
        self.data=mujoco.MjData(self.model)
        self.ids=[self.model.body('link'+str(j)).id for j in range(1,7)]
        self.jp=np.zeros((3,6)); self.jr=np.zeros((3,6))

    def masses(self,q,gravity=False):
        m,d=self.model,self.data
        d.qpos[:]=q
        mujoco.mj_kinematics(m,d)
        mujoco.mj_comPos(m,d)
        matrices=np.zeros((6,6,6)); grav=np.zeros((6,6))
        for j,body in enumerate(self.ids):
            mujoco.mj_jacBodyCom(m,d,self.jp,self.jr,body)
            r=d.ximat[body].reshape(3,3)
            inertia=(r*m.body_inertia[body])@r.T
            matrices[j]=m.body_mass[body]*self.jp.T@self.jp+self.jr.T@inertia@self.jr
            if gravity:
                grav[j]=-m.body_mass[body]*self.jp.T@m.opt.gravity
        return matrices,grav

    def evaluate(self,q,qd):
        step=2e-6
        derivatives=np.empty((6,6,6,6))  # [body,coordinate,row,column]
        for k in range(6):
            shift=np.zeros(6); shift[k]=step
            plus,_=self.masses(q+shift)
            minus,_=self.masses(q-shift)
            derivatives[:,k]=(plus-minus)/(2*step)
        mass,gravity=self.masses(q,True)
        coriolis=np.stack([christoffel(derivatives[j],qd) for j in range(6)])
        return mass,coriolis,gravity


def object_mass(x,mass,inertia):
    r=Rotation.from_euler('xyz',x[3:]).as_matrix()
    w=angular_map(x[3:])
    result=np.zeros((6,6))
    result[:3,:3]=mass*np.eye(3)
    result[3:,3:]=w.T@(r@np.diag(inertia)@r.T)@w
    return result


def object_dynamics(x,xd,mass,inertia):
    matrix=object_mass(x,mass,inertia)
    dm=np.zeros((6,6,6))
    step=2e-6
    for k in range(3,6):
        shift=np.zeros(6); shift[k]=step
        dm[k]=(object_mass(x+shift,mass,inertia)-object_mass(x-shift,mass,inertia))/(2*step)
    return matrix,christoffel(dm,xd),np.array([0.,0.,9.81*mass,0.,0.,0.])


def coupled_bases(robot,j,jdot,x,xd,payload_mass,payload_inertia,share=.5):
    mr,cr,gr=robot
    mo,co,go=object_dynamics(x,xd,payload_mass,payload_inertia)
    d=np.zeros((8,6,6)); c=np.zeros_like(d); w=np.zeros((8,6))
    d[:6]=mr; c[:6]=cr; w[:6]=gr
    # Parameter 6 scales the nominal 0.05 Nm s/rad viscous joint friction.
    # Caller fills w[6] using the actual measured local joint velocity.
    d[7]=share*j.T@mo@j
    c[7]=share*j.T@(mo@jdot+co@j)
    w[7]=share*j.T@go
    return d,c,w
