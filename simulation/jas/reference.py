"""New six-dimensional cooperative-transport task and C2 performance bounds."""
import numpy as np


def desired(t, center):
    # Independently specified six-axis excitation, followed by a smooth hold.
    # quintic transition of the phase speed gives a C2 stop at t=10 s.
    if t <= 8.0:
        phase, speed, accel = t, 1., 0.
    elif t < 10.0:
        z = (t-8.)/2.
        # speed=1-10z^3+15z^4-6z^5; its integral matches both ends.
        phase = 8.+2*(z-2.5*z**4+3*z**5-z**6)
        speed = 1-10*z**3+15*z**4-6*z**5
        accel = (-30*z**2+60*z**3-30*z**4)/2
    else:
        phase, speed, accel = 9., 0., 0.
    amplitude = np.array([.035,.025,.025,.08,.07,.08])
    frequency = np.array([.83,1.11,.67,.92,.76,1.21])
    shift = np.array([0.,.7,1.2,.4,1.,1.8])
    arg = frequency*phase+shift
    # Zero displacement and velocity at t=0; smooth startup over 1 s.
    if t < 1.:
        z=t
        gate=10*z**3-15*z**4+6*z**5
        dgate=30*z**2-60*z**3+30*z**4
        ddgate=60*z-180*z**2+120*z**3
    else:
        gate,dgate,ddgate=1.,0.,0.
    wave=amplitude*np.sin(arg)
    dwave=amplitude*frequency*np.cos(arg)*speed
    ddwave=amplitude*(-frequency**2*np.sin(arg)*speed**2+frequency*np.cos(arg)*accel)
    return (np.asarray(center)+gate*wave,
            dgate*wave+gate*dwave,
            ddgate*wave+2*dgate*dwave+gate*ddwave)


def envelope(t, chi, ell, nonmonotonic=False):
    # Steady upper/lower bounds differ by axis and sign, in SI/radian units.
    final_a=np.array([.012,.014,.010,.045,.040,.050])
    final_b=np.array([.018,.010,.016,.060,.055,.035])
    ua=ell*np.tanh(chi*final_a)
    ub=ell*np.tanh(chi*final_b)
    rate=.65
    e=np.exp(-rate*t)
    u=ua+(ell-ua)*e
    v=ub+(ell-ub)*e
    du=-rate*(ell-ua)*e
    dv=-rate*(ell-ub)*e
    ddu=rate**2*(ell-ua)*e
    ddv=rate**2*(ell-ub)*e
    if nonmonotonic and 3. < t < 7.:
        # Compact C2 relaxation bump: zero value, first/second derivative at ends.
        z=(t-3.)/4.
        bump=64*z**3*(1-z)**3
        db=48*z**2*(1-z)**2*(1-2*z)
        ddb=24*z*(1-z)*(1-5*z+5*z*z)
        extra=.23*ell
        u=u+extra*bump
        v=v+.8*extra*bump
        du=du+extra*db
        dv=dv+.8*extra*db
        ddu=ddu+extra*ddb
        ddv=ddv+.8*extra*ddb
    if np.any(u>ell*(1+1e-12)) or np.any(v>ell*(1+1e-12)):
        raise ValueError('Envelope must not exceed the stretching range.')
    return u,v,du,dv,ddu,ddv
