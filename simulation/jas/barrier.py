"""Equations (errortransfor), (barrier), (a), (b), (huamohanshu).

All derivatives use position, velocity and known C2 envelope derivatives.
No measured acceleration or numerical differentiation of measured velocity.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class BarrierState:
    xi: np.ndarray
    xidot: np.ndarray
    a: np.ndarray
    b: np.ndarray
    adot: np.ndarray
    bdot: np.ndarray
    eta: np.ndarray
    margin: np.ndarray
    upper: np.ndarray
    lower: np.ndarray


def transform(error, velocity, envelope, chi, ell):
    u, v, du, dv, ddu, ddv = envelope
    z = np.tanh(chi * error)
    eta = ell * z
    eta_e = chi * ell * (1 - z * z)
    deta = eta_e * velocity
    gap_a, gap_b = u - eta, v + eta
    if np.any(gap_a <= 0) or np.any(gap_b <= 0):
        raise FloatingPointError('JAS barrier domain violated; trajectory is not clipped.')
    den = gap_a * gap_b
    dden = (du - deta) * gap_b + gap_a * (dv + deta)
    numerator = u * v + eta * eta
    a = eta_e * numerator / den**2
    adot = ((-2 * chi * z * eta_e * velocity) * numerator
            + eta_e * (du*v + u*dv + 2*eta*deta)) / den**2 - 2*a*dden/den
    n = du*v + u*dv + (du-dv)*eta
    dn = ddu*v + 2*du*dv + u*ddv + (ddu-ddv)*eta + (du-dv)*deta
    b = -eta*n/den**2
    bdot = -(deta*n + eta*dn)/den**2 - 2*b*dden/den
    with np.errstate(divide='ignore', invalid='raise'):
        upper = np.arctanh(u/ell)/chi
        lower = -np.arctanh(v/ell)/chi
    return BarrierState(eta/den, a*velocity+b, a, b, adot, bdot,
                        eta, np.minimum(gap_a,gap_b)/ell, upper, lower)


def terminal_surface(xi, xidot, lam1, lam2, alpha, beta, delta):
    p1 = (3-alpha)*delta**(alpha-1)/2
    p2 = (alpha-1)*delta**(alpha-3)/2
    inside = np.abs(xi) < delta
    h = np.where(inside, p1*xi+p2*xi**3, np.sign(xi)*np.abs(xi)**alpha)
    # Avoid evaluating a negative power at zero in np.where's unused branch.
    dh = np.empty_like(xi)
    dh[inside] = p1+3*p2*xi[inside]**2
    dh[~inside] = alpha*np.abs(xi[~inside])**(alpha-1)
    reference = -lam1*h-lam2*np.sign(xi)*np.abs(xi)**beta
    dreference = -(lam1*dh+lam2*beta*np.abs(xi)**(beta-1))*xidot
    return xidot-reference, reference, dreference
