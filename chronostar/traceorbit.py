"""
traceorbit.py

A module aimed at projecting an orbit forward or backward through time.
Operates in a co-rotating, RH cartesian coordinate system centred on the
local standard of rest.
"""
import logging
import numpy as np

# from astropy.io import fits
from galpy.orbit import Orbit
from galpy.potential import MWPotential2014 #, MiyamotoNagaiPotential
from galpy.util import bovy_conversion

mp = MWPotential2014
# mp = MiyamotoNagaiPotential(a=0.5,b=0.0375,amp=1.,normalize=1.) # Params from the example webpage. No idea if that's good or not.

def convert_myr2bovytime(times):
    """
    Convert times provided in Myr into times in bovy internal units.

    Galpy parametrises time based on the natural initialising values
    (r_0 and v_0) such that after 1 unit of time, a particle in a
    circular orbit at r_0, with circular velocity of v_0 will travel
    1 radian, azimuthally.

    Parameters
    ---------
    times : [ntimes] float array
        Times in Myr

    Return
    ------
    bovy_times : [ntimes] float array
        Times in bovy internal units
    """
    bovy_times = times*1e-3 / bovy_conversion.time_in_Gyr(220., 8.)
    return bovy_times


def convert_bovytime2myr(times):
    chron_times = times/1e-3 * bovy_conversion.time_in_Gyr(220., 8.)
    return chron_times


def convert_cart2galpycoords(data, ts=None, ro=8., vo=220., debug=False,
                             bovy_times=None, lsr_centered=True):
    """
    To build: construct this function so the treatment of galpy
    orbits can be debugged more easily.

    Parameters
    ----------
    data: [npoints, 6] float array
        Phase-space positions in standard Chronostar coordinates:
        right-handed cartesian system centred on the local standard
        of rest (sun's radius and azimuthal position, projected onto
        the galactic plane, with velocities as given by Schoenrich
        2012(?) ).
        [X, Y, Z, U, V, W]
        [pc, pc, pc, km/s, km/s, km/s]
    ts: [npoints] float array
        The time [in Myr] of each phase-space position
    ro: float [8.]
        galpy coordinate system set up term
    vo: float [220.]
        galpy coordinate system set up term
    lsr_centered: boolean {True}
        If True, behaves as normal: each point in the star's orbit is rotated
        backwards by the LSR's azimuthal change, with velocity vectors
        rotated accordingly.
        New addition. If false, orbits azimuthal positions aren't modified.
        If this is false, you should also calculate the LSR's orbit, and
        take the difference.

    Returns
    -------
    galpy_coords: [npoints, 6] float array
        The phase-space positions in galpy cylindrical,
        normalised coordinates
        [R, vR, vT, z, vz, phi]
        where distances are in units `ro` and velocities in units
        `vo`.
    """
    # Tidy input
    if ts is None:
        ts = 0.0
    if bovy_times is None:
        bovy_times = convert_myr2bovytime(ts)
    data = np.array(data)

    if lsr_centered:
        phi_lsr = np.copy(bovy_times)

    Xs, Ys, Zs, Us, Vs, Ws = data.T

    # Simply scale vertical height and velocities
    zs = Zs / 1000. / ro
    vzs = Ws / vo

    # Get Rs from X and Y
    Rs = np.sqrt((ro - Xs/1000.)**2 + (Ys/1000.)**2) / ro

    # Get azimuthal angle with respect to Chronostar origin
    if debug:
        import pdb; pdb.set_trace()
    phis = np.arctan2(Ys/1000., ro - Xs/1000.)

    # Calculate planar velocities. Note that we need to incorporate
    # The velocity of the LSR in V
    vTs = ((Vs+220) * np.cos(phis) + Us*np.sin(phis))/vo
    vRs = ((Vs+220) * np.sin(phis) - Us * np.cos(phis))/vo

    # Finally, we offset the azimuthal position angle by the amount
    # travelled by the lsr
    if lsr_centered:
        phis += phi_lsr

    galpy_coords = np.vstack((Rs, vRs, vTs, zs, vzs, phis)).T

    # If only one coord, get rid of unnecessary dimension
    if galpy_coords.shape == (1,6):
        galpy_coords = galpy_coords[0]
    return galpy_coords


def convert_galpycoords2cart(data, ts=None, ro=8., vo=220., rc=True, lsr_centered=True):
    """
    Converts orbits from galpy internal coords to chronostar coords

    Data should be raw galpy data (i.e. output from o.getOrbit()).
    Chronostar coordinate frame is a corotating reference frame centred on
    the LSR as defined by the Schoenrich solar motion of
    XYZUVW = 0, 0, 25pc, 11.1 km/s, 12.24 km/s, 7.25 km/s
    Galpy coordinates are [R, vR, vT, z, vz, phi]
    By default, positions are scaled by LSR distance from galactic centre,
    ro=8kpc, and velocities scaled by the LSR circular velocity,
    vo = 220km/s. Time is scaled such that after 1 time unit has passed,
    the LSR has travelled 1 radian about the galactic centre. The values are
    returned in a [ntimes, 6]

    array:
        R : galactic radial distance /ro
        vR : galactic radial velocity /vo
        vT : circular velocity /vo
        z  : vertical distance from plane / ro
        vz : vertical velocity / vo
        phi : angle about the galaxy (anticlockwise from LSR's location at
             t=0)
    For example, the LSR at t=0.0 and t=1.0 as values:
    [1., 0., 1., 0., 0., 0.]
    [1., 0., 1., 0., 0., 1.]

    Parameters
    ----------
    data : [ntimes, 6] float array
        output from o.getOrbit. Data is encoded as:
        [R, vR, vT, z, vz, phi]
        R : galactic radial distance /ro
        vR : galactic radial velocity /vo
        vT : circular velocity /vo
        z  : vertical distance from plane / ro
        vz : vertical velocity / vo
        phi : angle about the galaxy (anticlockwise from LSR's location
              at t=0)
    ts : [ntimes] float array [galpy time units]
        times used to generate orbit. Ensure the units are in galpy time
        units
    ro : float
        a conversion factor that takes units from galpy units to
        physical units. If left as default, output will be in kpc
    vo : float
        a conversion factor that takes units form galpy units to
        physical units. If left as default, output will be in km/s
        This is also the circular velocity of a circular orbit with X,Y
        equal to that of the sun.
    rc : boolean
        whether to calculate XYZUVW in a right handed coordinate system
        (X, U positive towards galactic centre)
    lsr_centered: boolean {True}
        New addition. If false, orbits azimuthal positions aren't modified.
        If True, behaves as normal: each point in the star's orbit is rotated
        backwards by the LSR's azimuthal change, with velocity vectors
        rotated accordingly.

    Returns
    -------
    xyzuvw : [ntimes, 6] float array
        [pc, pc, pc, km/s, km/s, km/s] - traced orbit in chronostar
        coordinates (cartesian coordinate system with LSR as the origin
        and coordinate frame co-rotates such that the X axis points
        towards the galactic centre.

    TODO: This works for t=0, but not for other times
    """
    if ts is not None and lsr_centered:
        phi_lsr = ts
    else:
        phi_lsr = 0.0
    R, vR, vT, z, vz, phi_s = data.T

    # This is the angular distance between the LSR and our star
    if lsr_centered:
        phi = (phi_s.T - phi_lsr).T
    else:
        phi = phi_s

    # Can convert to XYZUVW coordinate frame. See thesis for derivation
    # Need to scale values back into physical units with ro and vo.
    # 1. in X and V are the LSR R and vT respectively (which are unitary
    # due to the normalisation of units inside galpy
    X = 1000 * ro * (1. - R * np.cos(phi))
    Y = 1000 * ro * R * np.sin(phi)
    Z = 1000 * ro * z

    # Ensure that -at the origin- U points towards the galactic centre (like X),
    # and V points in direction of circular rotation (like Y).
    U = vo * (-vR*np.cos(phi) + vT*np.sin(phi))
    V = vo * ( vT*np.cos(phi) + vR*np.sin(phi) - 1.)
    W = vo * vz

    if not rc:
        print("BUT EVERYONE IS USING RHC!!!")
        X = -X
        U = -U

    xyzuvw = np.vstack((X,Y,Z,U,V,W)).T
    # included for compatability with single data point
    if xyzuvw.shape == (1,6):
        xyzuvw = xyzuvw[0]
    return xyzuvw

def convert_cart2curvilin(data, ro=8., vo=220.,
                             lsr_centered=True):
    """
    MZ (2020 - 01 - 17)

    Converts cartesian coordinates XYZUVW (given with respect to the LSR) to
    the curvilinear system.
    Curvilinear system is corotating so its radial component xi is always pointing
    towards the galactic center.
    Coordinates in the curvilinear system are [xi, eta, zeta, xidot, etadot, zetadot]

    Parameters
    ----------
    data: [6, (npoints)] float np.array
        [pc, pc, pc, km/s,km/s,km/s]
        [X,  Y,  Z,  U,   V,   W]

    Returns
    -------
    curvilin_coordinates: [6, (npoints)] float np.array
        xi     : radial distance from the origin in LSR
        eta    :
        zeta   : vertical distance from plane
        xidot  :
        etadot :
        zetadot:

    """
    data = np.array(data)

    X, Y, Z, U, V, W = data.T

    if not lsr_centered:
        todo=True
        logging.log('LSR in convert cart2curvilin is not implemented yet.')


    R0 = ro*1000.0 # pc
    Omega0 = vo/R0 # km/s / pc

    # Place the velocities in a rotating frame
    U = U - Y*Omega0
    V = V + X*Omega0

    R = np.sqrt(Y**2 + (R0-X)**2)
    phi = np.arctan2(Y, R0-X)

    xi = R0-R
    eta = phi*R0
    zeta = Z
    xidot = U*np.cos(phi) - V*np.sin(phi)
    etadot = R0/R * (V*np.cos(phi) + U*np.sin(phi))
    zetadot = W

    curvilin_coordinates = np.array([xi, eta, zeta, xidot, etadot, zetadot])

    return curvilin_coordinates.T


def convert_curvilin2cart(data, ro=8., vo=220.,
                          lsr_centered=True):
    """
    MZ (2020 - 01 - 17)

    Returns
    -------

    """

    xi, eta, zeta, xidot, etadot, zetadot = data.T

    R0 = ro*1000.0

    R = R0 - xi
    phi = eta/R0

    X = xi*np.cos(phi) + R0*(1.0-np.cos(phi)) #R0 - R*np.cos(phi)
    Y = R*np.sin(phi)
    Z = zeta

    U = xidot*np.cos(phi) + R/R0*etadot*np.sin(phi)
    V = - xidot*np.sin(phi) + R/R0*etadot*np.cos(phi)
    W = zetadot

    # Convert to a non-rotating observed frame
    Omega0 = vo/R0 # km/s / pc
    U = U + Y*Omega0
    V = V - X*Omega0

    cart_coordinates = np.array([X, Y, Z, U, V, W])

    return cart_coordinates.T

def epicyclic_approx(data, times=None, sA=0.89, sB=1.15, sR=1.21):
    """
    MZ (2020 - 01 - 17)

    Epicyclic approximation following the Makarov et al. 2004 paper
    in the curvilinear coordinate system:
    The radial component xi is pointing towards the Galactic center
    at all times and equals 0 at R0.
    The circular component eta circles around the Galaxy; eta = phi*R.
    The vertical component is defined as a displacement from the Galactic plane.

    This approximation works close to the LSR.

    Parameters
    ------------
    data : [pc, pc*, pc, km/s, km/s, km/s] # *parsecs in the eta component are scales parsecs...
           xi, eta, zeta, xidot, etadot, zetadot
    """
    xi0, eta0, zeta0, xidot0, etadot0, zetadot0 = data.T

    # Bovy 2017
    A0 = 15.3  # km/s/kpc
    B0 = -11.9  # km/s/kpc

    # Bovy 2017. Scale factors to match MW2014.
    # Mike's scale factors
    #A0 = 0.97*15.3 # km/s/kpc
    #B0 = 1.12*(-11.9) # km/s/kpc

    # Marusa's factors
    A0 = A0*sA # km/s/kpc
    B0 = B0*sB # km/s/kpc


    # Unit conversion: convert from km/s/kpc to Myr-1
    A = A0 * 0.0010227121650537077  # Myr-1
    B = B0 * 0.0010227121650537077  # Myr-1

    # Fine tuning rho
    rho_scale_factor = sR #1.36
    rho = rho_scale_factor * 0.0889  # M0/pc3
    Grho = rho * 0.004498502151575285  # Myr-2; rho should be given in M0/pc3
    kappa = np.sqrt(-4.0 * B * (A - B)) # Myr-1
    nu = np.sqrt(4.0 * np.pi * Grho + (A + B) * (A - B)) # Myr-1

    t=times

    kt=kappa*t
    nt=nu*t

    xi = xi0 + xidot0/kappa*np.sin(kt) + (etadot0 - 2.0*A*xi0) * (1.0 - np.cos(kt)) / (2.0*B)
    eta = eta0 - xidot0 * (1.0 - np.cos(kt)) / (2.0*B) + etadot0 * (A*kt - (A-B)*np.sin(kt)) / (kappa*B) - xi0 * 2.0*A*(A-B)*(kt-np.sin(kt)) / (kappa*B)
    zeta = zeta0*np.cos(nt) + zetadot0/nu*np.sin(nt)

    xidot = xidot0*np.cos(kt) + (etadot0 - 2.0*A*xi0)*kappa*np.sin(kt) / (2.0*B)
    etadot = -xidot0*kappa/(2.0*B)*np.sin(kt) + etadot0/B*(A-(A-B)*np.cos(kt)) - 2.0*A*xi0*(A-B)*(1.0-np.cos(kt))/B
    zetadot = -zeta0*nu*np.sin(nt) + zetadot0*np.cos(nt)

    new_position = np.array([xi, eta, zeta, xidot, etadot, zetadot])
    new_position = new_position.T
    return new_position

def trace_epicyclic_orbit(xyzuvw_start, times=None, sA=0.89, sB=1.15, sR=1.21, single_age=True,
                          ro=8., vo=220.):
    """
    MZ (2020 - 01 - 17)

    Given a star's XYZUVW relative to the LSR (at any time), project its
    orbit forward (or backward) to each of the times listed in *times*
    using epicyclic approximation. This only works close to the LSR.

    Positive times --> traceforward
    Negative times --> traceback

    Parameters
    ----------
    xyzuvw : [pc,pc,pc,km/s,km/s,km/s]
    times : (float) or ([ntimes] float array)
        Myr - time of 0.0 must be present in the array. Times need not be #TODO: time 0.0 really? [TC: this was true for galpy]
        spread linearly.
    single_age: (bool) {True}
        Set this flag if only providing a single age to trace to
        This is there for the plotting purposes.

    Returns
    -------
    xyzuvw_tf : [ntimes, 6] array
        [pc, pc, pc, km/s, km/s, km/s] - the traced orbit with positions
        and velocities
    """
    if single_age:
        # replace 0 with some tiny number
        try:
            if times == 0.:
                times = 1e-15
            # times = np.array([0., times])
        except ValueError as err:
            if not err.args:
                err.args = ('',)
            err.args = err.args + ('WARNING: comparing array to float? '
                                   'Did you leave single_age as True?',)
            raise

    else:
        raise UserWarning('Multi age orbit integation no longer supported')
        times = np.array(times)

    # Make sure numbers are floats!
    xyzuvw_start = np.atleast_2d(xyzuvw_start.astype(np.float))

    # Units: Velocities are in km/s, convert into pc/Myr
    # try:
        # xyzuvw_start[3:] = xyzuvw_start[3:] * 1.0227121650537077 # pc/Myr
    # except:
    xyzuvw_start[:,3:] = xyzuvw_start[:,3:] * 1.0227121650537077 # pc/Myr

    # Transform to curvilinear
    curvilin = convert_cart2curvilin(xyzuvw_start, ro=ro, vo=vo)

    # Trace orbit with epicyclic approx.
    new_position = epicyclic_approx(curvilin, times=times, sA=sA, sB=sB, sR=sR)

    #print(curvilin)
    #print(new_position[-1])

    # Transform back to cartesian
    xyzuvw_new = convert_curvilin2cart(new_position, ro=ro, vo=vo)

    # Units: Transform velocities from pc/Myr back to km/s
    # xyzuvw_new[3:] /= 1.0227121650537077
    xyzuvw_new[:,3:] /= 1.0227121650537077

    # Remove any excess dimensions with squeeze
    return np.squeeze(xyzuvw_new)

def trace_galpy_orbit(galpy_start, times=None, single_age=True,
                      potential=MWPotential2014, ro=8, vo=220.,
                      method='dopr54_c'):
    """
    An extra (potentially superfluous) function, currently only used
    for testing.

    Given a star's initial phase-space position in galpy coordinates,
    project its orbit forward (or backward) to each of the times listed
    in `times`.
    Note: times should be in Myr
    """
    if single_age:
        # replace 0 with some tiny number
        try:
            if times == 0.:
                times = 1e-15
            times = np.array([0., times])
        except ValueError as err:
            if not err.args:
                err.args = ('',)
            err.args = err.args + ('WARNING: comparing array to float? '
                                   'Did you leave single_age as True?',)
            raise

    else:
        times = np.array(times)

    #Make sure we have a float array.
    #MJI: Not sure why this is needed, as this isn't changed in-place anywhere.
    #TC: Cause I kept passing in a lists of integers by accident, which would
    # the crash things when converted to arrays.
    galpy_start = np.copy(galpy_start).astype(np.float)

    #Convert to to Galpy times, which go from 0 to 2\pi around the LSR orbit.
    bovy_times = convert_myr2bovytime(times)

    # since the LSR is constant in chron coordinates, the starting point
    # is always treated as time 0
    # galpy_coords = convert_cart2galpycoords(xyzuvw_start, ts=0.,
    #                                         ro=ro, vo=vo)

    o = Orbit(vxvv=galpy_start, ro=ro, vo=vo)
    o.integrate(bovy_times, potential, method=method)
    galpy_coords = o.getOrbit()

    # galpy_coords = convert_galpycoords2cart(o.getOrbit(), bovy_times,
                                            # ro=ro, vo=vo)
    #import pdb; pdb.set_trace()
    if single_age:
        return galpy_coords[-1]
    return galpy_coords


def trace_cartesian_orbit(xyzuvw_start, times=None, single_age=True,
                          potential=MWPotential2014, ro=8., vo=220.,
                          method='dopr54_c'):
    """
    Given a star's XYZUVW relative to the LSR (at any time), project its
    orbit forward (or backward) to each of the times listed in *times*

    Positive times --> traceforward
    Negative times --> traceback

    TODO: Primary source of inefficiencies, 1366.2 (s)

    Parameters
    ----------
    xyzuvw : [pc,pc,pc,km/s,km/s,km/s]
    times : (float) or ([ntimes] float array)
        Myr - time of 0.0 must be present in the array. Times need not be
        spread linearly.
    single_age: (bool) {True}
        Set this flag if only providing a single age to trace to
    method: str {'odeint'}
        The integration method used by the galpy orbit integrator. An
        alternative is 'dopr54_c' which is a fast, high-order Dormand-Prince
        method.
        Options (listed in improving speed):
            'odeint',
            'symplec4_c',
            'rk4_c',
            'dopr54_c',
            'rk6_c', [!!! this one has a bug !!!]
        Our tests show dopr54_c to be fastest with no bugs.

    Returns
    -------
    xyzuvw_tf : [ntimes, 6] array
        [pc, pc, pc, km/s, km/s, km/s] - the traced orbit with positions
        and velocities

    Notes
    -----
    Profiling comments have been left in for future reference, but note
    that the profiling was done with previous versions of coordinate
    functions - ones that utilised astropy.units (and thus symbolic algebra)
    """
    if single_age:
        # replace 0 with some tiny number
        try:
            if times == 0.:
                times = 1e-15
            times = np.array([0., times])
        except ValueError as err:
            if not err.args:
                err.args = ('',)
            err.args = err.args + ('WARNING: comparing array to float? '
                                   'Did you leave single_age as True?',)
            raise

    else:
        raise UserWarning('Multi age orbit integation no longer supported')
        times = np.array(times)

    #Make sure we have a float array.
    xyzuvw_starts = np.array(xyzuvw_start).astype(np.float)

    # Check if we are doing multiple orbits in one call
    if len(xyzuvw_starts.shape) == 1:
        xyzuvw_starts = [xyzuvw_starts]

    #Convert to to Galpy times, which go from 0 to 2\pi around the LSR orbit. 
    bovy_times = convert_myr2bovytime(times)

    # since the LSR is constant in chron coordinates, the starting point
    # is always treated as time 0
    xyzuvw_ends = []
    for xyzuvw_start in xyzuvw_starts:
        galpy_coords = convert_cart2galpycoords(xyzuvw_start, ts=0.,
                                                ro=ro, vo=vo)
        o = Orbit(vxvv=galpy_coords, ro=ro, vo=vo)
        o.integrate(bovy_times, potential, method=method)

        xyzuvw_ends.append(convert_galpycoords2cart(o.getOrbit(), bovy_times,
                                          ro=ro, vo=vo)[-1])
    xyzuvw_ends = np.squeeze(np.array(xyzuvw_ends))

   #  if single_age:
   #      return xyzuvw[-1]
    return xyzuvw_ends


def trace_many_cartesian_orbit(xyzuvw_starts, times=None, single_age=True,
                               savefile=''):
    """
    (This function is not used by Chronostar (yet). It is currently here
    purely for testing reasons.)

    Given a star's XYZUVW relative to the LSR (at any time), project its
    orbit forward (or backward) to each of the times listed in *times*

    Positive times --> traceforward
    Negative times --> traceback

    Parameters
    ----------
    xyzuvw_starts : [nstars, 6] array (pc,pc,pc,km/s,km/s,km/s)
    times : [ntimes] float array
        Myr - time of 0.0 must be present in the array. Times need not be
        spread linearly.
    single_age : (Boolean {False})
        If set to true, times must be given a single non-zero float

    Returns
    -------
    xyzuvw_to : [nstars, ntimes, 6] array
        [pc, pc, pc, km/s, km/s, km/s] - the traced orbit with positions
        and velocities
        If single_age is set, output is [nstars, 6] array
    """
    if single_age:
        ntimes = 1
    else:
        times = np.array(times)
        ntimes = times.shape[0]

    nstars = xyzuvw_starts.shape[0]
    logging.debug("Nstars: {}".format(nstars))

    if single_age:
        xyzuvw_to = np.zeros((nstars, 6))
    else:
        xyzuvw_to = np.zeros((nstars, ntimes, 6))
    for st_ix in range(nstars):
        xyzuvw_to[st_ix] = trace_cartesian_orbit(xyzuvw_starts[st_ix], times,
                                                 single_age=single_age)
    #TODO: test this
    if savefile:
        np.save(savefile, xyzuvw_to)
    return xyzuvw_to


def trace_orbit_builder(potential):
    """
    Build a replica of trace_cartesian_orbit but with custom
    potential. e.g. MiyamotoNagaiPotential
    With parameters (from website):
    MiyamotoNagaiPotential(a=0.5,b=0.0375,amp=1.,normalize=1.)
    """
    def f_(xyzuvw_start, times=None, single_age=True):
        return trace_cartesian_orbit(xyzuvw_start=xyzuvw_start, times=times,
                                     single_age=single_age,
                                     potential=potential)
    return f_


# def generateTracebackFile(star_pars_now, times, savefile=''):
#     """
#     Take XYZUVW of the stars at the current time and trace back for
#     timesteps
#
#     Parameters
#     ----------
#     star_pars_now: dict
#         'xyzuvw': [nstars, 6] numpy array
#             the mean XYZUVW for each star at t=0
#         'xyzuvw_cov': [nstars, 6, 6] numpy array
#             the covariance for each star at t=0
#     times: [ntimes] array
#         the times at which to be traced back to
#     """
#     times = np.array(times)
#     ntimes = times.shape[0]
#     nstars = star_pars_now['xyzuvw'].shape[0]
#     logging.debug("Attempting traced means")
#     means = traceManyOrbitXYZUVW(star_pars_now['xyzuvw'], times,
#                                  single_age=False)
#     logging.debug("Successfully traced means")
#
#     covs = np.zeros((nstars, ntimes, 6, 6))
#     for star_ix in range(nstars):
#         for time_ix in range(ntimes):
#             if times[time_ix] == 0.0:
#                 covs[star_ix, time_ix] = star_pars_now['xyzuvw_cov'][star_ix]
#             else:
#                 covs[star_ix, time_ix] =\
#                     tf.transformCovMat(cov=star_pars_now['xyzuvw_cov'][star_ix],
#                                        trans_func=traceOrbitXYZUVW,
#                                        loc=star_pars_now['xyzuvw'][star_ix],
#                                        args=(times[time_ix],)
#                                        )
#     logging.debug("Successfully traced covs")
#     star_pars_all = {'xyzuvw':means,
#                      'xyzuvw_cov':covs,
#                      'times':times}
#     if savefile:
#         if (savefile[-3:] != 'fit') and (savefile[-4:] != 'fits'):
#             savefile = savefile + ".fits"
#         hl = fits.HDUList()
#         hl.append(fits.PrimaryHDU())
#         hl.append(fits.ImageHDU(star_pars_all['xyzuvw']))
#         hl.append(fits.ImageHDU(star_pars_all['xyzuvw_cov']))
#         hl.append(fits.ImageHDU(star_pars_all['times']))
#         hl.writeto(savefile, overwrite=True)
#
#     return star_pars_all
#
#

