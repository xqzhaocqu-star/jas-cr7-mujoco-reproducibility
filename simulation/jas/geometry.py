"""Nonsingular six-coordinate object kinematics from local encoders."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation


def skew(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def angular_map(rpy):
    _,p,y=rpy
    cp,sp,cy,sy=np.cos(p),np.sin(p),np.cos(y),np.sin(y)
    return np.array([[cy*cp,-sy,0.],[sy*cp,cy,0.],[-sp,0.,1.]])


class LocalKinematics:
    def __init__(self, model_path, base_position, base_rotation,
                 palm_to_object_position, palm_to_object_rotation):
        self.model=mujoco.MjModel.from_xml_path(str(model_path))
        self.data=mujoco.MjData(self.model)
        self.site=self.model.site('palm').id
        self.base_p=np.asarray(base_position)
        self.base_r=np.asarray(base_rotation).reshape(3,3)
        self.offset=np.asarray(palm_to_object_position)
        self.rotation=np.asarray(palm_to_object_rotation).reshape(3,3)

    def pose_jacobian(self,q):
        m,d=self.model,self.data
        d.qpos[:]=q
        mujoco.mj_kinematics(m,d)
        mujoco.mj_comPos(m,d)
        r=d.site_xmat[self.site].reshape(3,3)
        p=d.site_xpos[self.site]
        arm=np.zeros((3,6)); rot=np.zeros((3,6))
        mujoco.mj_jacSite(m,d,arm,rot,self.site)
        shift=r@self.offset
        obj_r=self.base_r@r@self.rotation
        rpy=Rotation.from_matrix(obj_r).as_euler('xyz')
        position=self.base_p+self.base_r@(p+shift)
        j=np.vstack((self.base_r@(arm-skew(shift)@rot),
                     np.linalg.solve(angular_map(rpy),self.base_r@rot)))
        return np.r_[position,rpy],j

    def evaluate(self,q,qd):
        step=1e-6/(1+np.linalg.norm(qd))
        _,jp=self.pose_jacobian(q+step*qd)
        _,jm=self.pose_jacobian(q-step*qd)
        pose,j=self.pose_jacobian(q)
        return pose,j,(jp-jm)/(2*step)
