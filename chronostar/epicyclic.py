"""
epicyclic.py

[MZ]
A module aimed at projecting an orbit forward or backward through time.
Operates in a co-rotating, RH cartesian coordinate system centred on the
local standard of rest.
Using epicyclic approximations. To be used for ages below 100Myr.

"""
import numpy as np

def trace_cartesian_orbit_epicyclic(xyzuvw_start, times=None):
    """
    Given a star's XYZUVW relative to the LSR (at any time), project its
    orbit forward (or backward) to each of the times listed in *times*

    Positive times --> traceforward
    Negative times --> traceback

    Parameters
    ----------
    xyzuvw : [pc,pc,pc,km/s,km/s,km/s]
    times : (float) or ([ntimes] float array)
        Myr - time of 0.0 must be present in the array (for the plotting purposes).
        Times need not be spread linearly.

    Returns
    -------
    xyzuvw_tf : [ntimes, 6] array
        [pc, pc, pc, km/s, km/s, km/s] - the traced orbit with positions
        and velocities

    Notes
    -----
    Under construction.
    """

    times = np.array(times)

    # Bovy 2017. Scale factors to match MW2014.
    A0 = 0.99 * 15.3  # km/s/kpc
    B0 = 1.04 * (-11.9)  # km/s/kpc

    s = np.sqrt(-B0 / (A0 - B0))
    A = A0 * 0.0010227121650537077  # Myr-1
    B = B0 * 0.0010227121650537077  # Myr-1

    # Scale factor to match MW2014
    rho = 1.15 * 0.0889  # M0/pc3
    Grho = rho * 0.004498502151575285  # Myr-2; rho should be given in M0/pc3
    kappa = np.sqrt(-4.0 * B * (A - B))
    nu = np.sqrt(4.0 * np.pi * Grho + (A + B) * (A - B))

    x, y, z, u, v, w = xyzuvw_start

    # Velocities are in km/s, convert into pc/Myr
    u = u * 1.0227121650537077  # pc/Myr
    v = v * 1.0227121650537077  # pc/Myr
    w = w * 1.0227121650537077  # pc/Myr

    Xg = (kappa * x - s * v) / (kappa - 2.0 * A * s)
    Yg = y + u / (kappa * s)
    phi_b = np.arctan2(u, (2.0 * A * Xg - v) * s)
    b0 = (x - Xg) / (s * np.cos(phi_b))
    phi_c = np.arctan2(-w, nu * z)
    Zg = z / np.cos(phi_c)

    # New position
    k = kappa * times + phi_b
    n = nu * times + phi_c
    X = Xg + b0 * s * np.cos(k)
    Y = Yg + 2.0 * A * Xg * times + b0 * np.sin(k)
    Z = Zg * np.cos(n)
    U = -b0 * kappa * s * np.sin(k)
    V = 2.0 * A * Xg + b0 * kappa * np.cos(k)
    W = -Zg * nu * np.sin(n)

    # Convert velocities from pc/Myr to km/s
    U = U / 1.0227121650537077  # km/s
    V = V / 1.0227121650537077  # km/s
    W = W / 1.0227121650537077  # km/s

    xyzuvw = np.array([X, Y, Z, U, V, W])

    return xyzuvw
