"""Independent scalar mechanism illustrations, not robot controller baselines.

Quadrature computes entry times into the same positive layer. No polynomial
regularization is evaluated below the layer, and no finite-time-to-zero claim
is attached to the C1 sliding law. Initial magnitudes outside the robot's
kinematic domain are explicitly a scalar mathematical illustration only.
"""
import numpy as np
from scipy.integrate import quad


def scalar_entry_times(settings):
    c = settings
    magnitudes = np.logspace(-4, 18, 89)
    delta = float(c.delta)
    linear = float(np.asarray(c.linear_surface_gain)[0])
    eps = .001
    ka, k2, k3 = (float(np.asarray(getattr(c, name))[0]) for name in ('ka','k2','k3'))

    def travel(radius, threshold, speed):
        if radius <= threshold:
            return 0.
        # Log-coordinate integration avoids losing the near-layer contribution
        # when the initial magnitude spans many orders of magnitude.
        return float(quad(lambda z: np.exp(z)/speed(np.exp(z)), np.log(threshold),
                          np.log(radius), epsabs=2e-11, epsrel=2e-11,limit=200)[0])

    sliding = lambda r: c.lam1*r**c.alpha+c.lam2*r**c.beta
    reaching = lambda r: ka*r+k2*r**c.upsilon1+k3*r**c.upsilon2
    return dict(initial_magnitude=magnitudes.tolist(),
                sliding_layer=delta,
                sliding_dual_power_entry_s=[travel(r,delta,sliding) for r in magnitudes],
                sliding_linear_entry_s=[float(max(0.,np.log(r/delta))/linear) for r in magnitudes],
                sliding_uniform_comparison_bound_s=float(1/(c.lam1*(1-c.alpha))+1/(c.lam2*(c.beta-1))),
                sliding_linear_gain=linear,
                reaching_layer=eps,
                reaching_dual_power_entry_s=[travel(r,eps,reaching) for r in magnitudes],
                reaching_linear_entry_s=[float(max(0.,np.log(r/eps))/ka) for r in magnitudes],
                reaching_uniform_comparison_bound_s=float(1/(k2*(c.upsilon1-1))+1/(k3*(1-c.upsilon2))),
                interpretation='Standalone scalar autonomous ODEs. These are not full robot trajectories, and their bounds are not the robot Theorem 2 inertia-dependent bound. The robot linear-reaching ablation retains robust sign; it is not asserted to be non-fixed-time.')
