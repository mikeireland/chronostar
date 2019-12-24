"""
Module to handle various coordinate transformations.

TODO: Update docstrings to reflect the removal of astropy
"""

import logging
import numpy as np

# a_o = 192.8595 * un.degree
# b_ncp = d_o = 27.1283 * un.degree
# l_ncp = l_o = 122.9319 * un.degree
#
# old_a_ngp = 192.25 * un.degree
# old_d_ngp = 27.4 * un.degree
# old_th = 123 * un.degree
#
# a_ngp = 192.25 * un.degree
# d_ngp = 27.4 * un.degree

eq_to_gc = np.array([
    [-0.06699, -0.87276, -0.48354],
    [ 0.49273, -0.45035,  0.74458],
    [-0.86760, -0.18837,  0.46020],
])

modern_eq_to_gc = np.array([
    [-0.05487549, -0.87343736, -0.48383454],
    [ 0.49411024, -0.44482901,  0.74698208],
    [-0.86766569, -0.19807659,  0.45598455]
])

modern_gc_to_eq = np.linalg.inv(modern_eq_to_gc)

gc_to_eq = np.linalg.inv(eq_to_gc)


def convert_ra2deg(hh, mm, ss):
    """
    Convert measurement of right ascension in hh:mm:ss to
    decimal degrees

    Parameters
    ----------
    hh : int -or- string -or- float
        hours component of right ascension
    mm : int -or- string -or- float
        minutes component of right ascension
    ss : int -or- string -or- float
        seconds component of right ascension

    Returns
    -------
    result : float
        Right ascension in degrees
    """
    if isinstance(hh, str):
        hh = float(hh)
    if isinstance(mm, str):
        mm = float(mm)
    if isinstance(ss, str):
        ss = float(ss)
    rahh = hh + mm / 60. + ss / 3600.
    return rahh * 360. / 24.


def convert_dec2deg(deg, arcm, arcs):
    """
    Convert measurement of declination in deg arcmin, arcsec to
    decimal degrees

    Parameters
    ----------
    deg : int -or- string -or- float
        degrees value
    arcm : int -or- string -or- float
        arc minute value
    arcs : int -or- string -or- float
        arc second value

    Returns
    -------
    result : float
        declination in degrees
    """

    if isinstance(deg, str):
        sign=1.0
        if '-' in deg:
            sign=-1.0
        deg = float(deg)
    else:
        sign = 1.0
        if deg < 0:  # But make sure that in case deg=-0 you still get this minus!
            sign = -1.0
    if isinstance(arcm, str):
        arcm = float(arcm)
    if isinstance(arcs, str):
        arcs = float(arcs)

    deg=np.abs(deg) # Multiply with the sign a few lines below
    result = deg + arcm / 60. + arcs / 3600.
    result = sign * result

    return result


def calc_eq2gc_matrix(a_deg=192.8595, d_deg=27.1283, th_deg=122.9319):
    """
    Generate matrix to transform cartesian points determined by equatorial
    coordinates to cartesian points determined by Galactic coordinates

    Using the RA (a) DEC (d) of Galactic north, and theta, generate matrix
    Default values are from J2000

    Parameters
    ----------
    a_deg : float (deg)
        right ascension of the north galactic pole
    d_deg : float (deg)
        declination of the north galactic pole
    th_deg : float (deg)
        position angle of the North Celestial Pole relative to the great
        semicircle passing through the NGP and zero Galactic longitude

    Returns
    -------
    res : [3x3] array
    """
    assert isinstance(a_deg, (int, float, np.float32, np.float64))
    a_rad = a_deg*np.pi/180
    d_rad = d_deg*np.pi/180
    th_rad = th_deg*np.pi/180
    first_t = np.array([
        [np.cos(a_rad),  np.sin(a_rad), 0],
        [np.sin(a_rad), -np.cos(a_rad), 0],
        [        0,          0, 1] 
    ])

    second_t = np.array([
        [-np.sin(d_rad),  0, np.cos(d_rad)],
        [         0, -1,         0],
        [np.cos(d_rad),   0, np.sin(d_rad)]
    ])
    third_t = np.array([
        [np.cos(th_rad),  np.sin(th_rad), 0],
        [np.sin(th_rad), -np.cos(th_rad), 0],
        [         0,           0, 1]
    ])
    return np.dot(third_t, np.dot(second_t, first_t))


def calc_gc2eq_matrix(a_deg=192.8595, d_deg=27.1283, th_deg=122.9319):
    """
    Generate a matrix that takes Galactic coordinates to equatorial

    This is simply the inverse of the EQ to GC matrix

    Parameters
    ----------
    a_deg : float (deg)
        right ascension of the north galactic pole
    d_deg : float (deg)
        declination of the north galactic pole
    th_deg : float (deg)
        position angle of the North Celestial Pole relative to the great
        semicircle passing through the NGP and zero Galactic longitude

    Returns
    -------
    result : [6x6] array
    """
    return np.linalg.inv(calc_eq2gc_matrix(a_deg, d_deg, th_deg))


def convert_angles2cartesian(theta_deg, phi_deg, radius=1.0):
    """
    theta   : angle (as astropy degrees) about the north pole (longitude, RA)
    phi : angle (as astropy degrees) from the plane (lattitude, dec))

    Tested
    """
    assert isinstance(theta_deg, (int, float, np.float32, np.float64))
    theta_rad = theta_deg*np.pi/180.
    phi_rad = phi_deg*np.pi/180
    x = radius * np.cos(phi_rad)*np.cos(theta_rad)
    y = radius * np.cos(phi_rad)*np.sin(theta_rad)
    z = radius * np.sin(phi_rad)
    return np.array((x,y,z))


def convert_cartesian2angles(x, y, z, return_dist=False):
    """Tested

    TODO: This takes up 10% of a run
    Uses astropy angles which slows everything down
    """
    dist = np.sqrt(x**2 + y**2 + z**2)
    if dist == 0.0:
        z += 1e-10 #HACK allowing sun (who has dist=0) to be inserted
        dist = np.sqrt(x**2 + y**2 + z**2)
    phi_deg = np.arcsin(z/dist)*180./np.pi
    theta_deg = np.mod((np.arctan2(y/dist,x/dist))*180./np.pi, 360.)
    if return_dist:
        return np.array([theta_deg, phi_deg, dist])
    else:
        return np.array([theta_deg, phi_deg])


def convert_equatorial2galactic(theta_deg, phi_deg):
    """
    Convert equatorial (ra, dec) to galactic (longitude, latitude)

    Parameters
    ----------
    theta: (float) right ascension in degrees
    phi:   (float) declination in degrees
    value: (bool) {True} Set flag if output desired as raw float (as opposed
                         to an astropy unit object)

    Output
    ------
    pos_gc: (float, float) Galactic coordinates l and b, in degrees
    """
    # logging.debug("Converting eq ({}, {}) to gc: ".format(theta_deg, phi_deg))
    assert isinstance(theta_deg, (int, float, np.float32, np.float64))
    cart_eq = convert_angles2cartesian(theta_deg, phi_deg)
    # logging.debug("Cartesian eq coords: {}".format(cart_eq))
    eq_to_gc = calc_eq2gc_matrix()
    cart_gc = np.dot(eq_to_gc, cart_eq)
    # logging.debug("Cartesian gc coords: {}".format(cart_gc))
    pos_gc_deg = convert_cartesian2angles(*cart_gc)
    return pos_gc_deg


def convert_galactic2equatorial(theta_deg, phi_deg, value=True):
    """
    Convert galactic (longitude, latitude) to equatorial (ra, dec)

    Parameters
    ----------
    theta: (float) galactic l in degrees
    phi:   (float) galactic b in degrees
    value: (bool) {True} Set flag if output desired as raw float (as opposed
                         to an astropy unit object)

    Output
    ------
    pos_gc: (float, float) Equatorial coordinates RA and DEC, in degrees
    """
    # logging.debug("Converting gc ({}, {}) to eq:".format(theta_deg, phi_deg))
    try:
        assert isinstance(theta_deg, (int, float, np.float32, np.float64))
    except AssertionError:
        print(type(theta_deg))
        AssertionError
    cart_gc = convert_angles2cartesian(theta_deg, phi_deg)
    # logging.debug("Cartesian eq coords: {}".format(cart_gc))
    gc_to_eq = calc_gc2eq_matrix()
    cart_eq = np.dot(gc_to_eq, cart_gc)
    # logging.debug("Cartesian gc coords: {}".format(cart_eq))
    pos_eq_deg = convert_cartesian2angles(*cart_eq)
    return pos_eq_deg


def calc_pm_coord_matrix(a_deg, d_deg):
    """
    Generate a coordinate matrix for calculating proper motions

    This is matrix `A` in Johnson & Soderblom (1987)
    """
    a_rad = a_deg*np.pi/180.
    d_rad = d_deg*np.pi/180.

    first_t = np.array([
        [ np.cos(d_rad),  0, -np.sin(d_rad)],
        [         0, -1,          0],
        [-np.sin(d_rad),  0, -np.cos(d_rad)]
    ])
    second_t = np.array([
        [np.cos(a_rad),  np.sin(a_rad), 0],
        [np.sin(a_rad), -np.cos(a_rad), 0],
        [        0,         0, -1],
    ])
    return np.dot(second_t, first_t)


def convert_pm2heliospacevelocity(a_deg, d_deg, pi, mu_a, mu_d, rv):
    """
    Convert proper motions to space velocities

    Paramters
    ---------
    a_deg : (deg) right ascension in equatorial coordinates
    d_deg : (deg) declination in equatorial coordinates
    pi : (arcsec) parallax
    mu_a : (arcsec/yr) proper motion in right ascension
    mu_d : (arcsec/yr) proper motion in declination
    rv : (km/s) radial velocity

    Returns
    -------
    UVW : [3] array
    """
    assert isinstance(a_deg, (int, float, np.float32, np.float64))

    B = np.dot(
        calc_eq2gc_matrix(),
        calc_pm_coord_matrix(a_deg, d_deg),
    )
    K = 4.74057 #(km/s) / (1AU/yr)
    astr_vels = np.array([
        rv,
        K * mu_a / pi,
        K * mu_d / pi,
    ])
    space_vels = np.dot(B, astr_vels)
    return space_vels


def convert_heliospacevelocity2pm(a_deg, d_deg, pi, u, v, w):
    """Take the position and space velocities, return proper motions and rv

    Paramters
    ---------
    a_deg : (deg) right ascension
    d_deg : (deg) declination
    pi : (as) parallax
    u : (km/s) heliocentric velocity towards galactic centre
    v : (km/s) heliocentric velocity towards in direction of circular orbit
    w : (km/s) heliocentric velocity towards galactic north

    Returns
    -------
    mu_a : (as/yr) proper motion in right ascension
    mu_d : (as/yr) proper motion in declination
    rv : (km/s) line of sight velocity
    """
    assert isinstance(a_deg, (int, float, np.float32, np.float64))
    # logging.debug("Parallax is {} as which is a distance of {} pc".format(
    #     pi, 1./pi
    # ))

    space_vels = np.array([u,v,w])

    B_inv = np.linalg.inv(np.dot(
        calc_eq2gc_matrix(),
        calc_pm_coord_matrix(a_deg, d_deg)
    ))
    sky_vels = np.dot(B_inv, space_vels) # now in km/s
    K = 4.74057 #(km/s) / (AU/yr)
    rv = sky_vels[0]
    mu_a = pi * sky_vels[1] / K
    mu_d = pi * sky_vels[2] / K
    return np.array([mu_a, mu_d, rv])


def convert_helioxyzuvw2astrometry(xyzuvw_helio):
    """
    Takes as input heliocentric XYZUVW values, returns astrometry

    Parameters
    ----------
    xyzuvw_helio : (pc, pc, pc, km/s, km/s, km/s) array
        The position and velocity of a star in a right handed cartesian system
        centred on the sun

    Returns
    -------
    a_deg : (deg) right ascention
    d_deg : (deg) declination
    pi : (as) parallax
    mu_a : (as/yr) proper motion in right ascension
    mu_d : (as/yr) proper motion in declination
    rv : (km/s) line of sight velocity
    """
    x, y, z, u, v, w = xyzuvw_helio
    l_deg, b_deg, dist = convert_cartesian2angles(x, y, z, return_dist=True)
    # logging.debug("l,b,Distance is {}, {}, {} pc".format(l_deg, b_deg, dist))
    a_deg, d_deg = convert_galactic2equatorial(l_deg, b_deg)
    pi = 1./dist
    mu_a, mu_d, rv = convert_heliospacevelocity2pm(a_deg, d_deg, pi, u, v, w)
    return np.array([a_deg, d_deg, pi, mu_a, mu_d, rv])


def convert_astrometry2helioxyzuvw(a_deg, d_deg, pi, mu_a, mu_d, rv):
    """
    Converts astrometry to heliocentric XYZUVW values

    Parameters
    ----------
    a_deg : (deg) right ascention
    d_deg : (deg) declination
    pi : (as) parallax
    mu_a : (as/yr) proper motion in right ascension
    mu_d : (as/yr) proper motion in declination
    rv : (km/s) line of sight velocity
    """
    # logging.debug("Input:\nra {}\ndec {}\nparallax {}\nmu_ra {}\nmu_de {}\n"
    #               "rv {}".format(a_deg, d_deg, pi, mu_a, mu_d, rv))
    dist = 1/pi #pc
    l_deg, b_deg = convert_equatorial2galactic(a_deg, d_deg)
    x, y, z = convert_angles2cartesian(l_deg, b_deg, radius=dist)
    u, v, w = convert_pm2heliospacevelocity(a_deg, d_deg, pi, mu_a, mu_d, rv)
    xyzuvw_helio = np.array([x,y,z,u,v,w])
    # logging.debug("XYZUVW heliocentric is : {}".format(xyzuvw_helio))
    return xyzuvw_helio


def convert_lsr2helio(xyzuvw_lsr, kpc=False):
    """
    Convert cartesian xyzuvw position from LSR-centred to helio-centred

    Assumes position is in pc unless stated otherwise

    Parameters
    ----------
    xyzuvw_lsr : [6] array
        3 position [pc] and 3 velocity with respect to the local standard
        of rest
    kpc : bool {False}
        If set, then treats input units of position as kpc (as opposed to pc)
        and outputs in kpc.
        Use this flag if inputing position into Galpy!!!

    Returns
    -------
    result : [6] array
        3 position [pc] (or [kpc] if `kpc` set to True)
        and 3 velocity with respect to the sun
    """
    XYZUVWSOLARNOW = np.array([0., 0., 25., 11.1, 12.24, 7.25])
    XYZUVWSOLARNOW_KPC = np.array([0., 0., 0.025, 11.1, 12.24, 7.25])
    if kpc:
        res = xyzuvw_lsr - XYZUVWSOLARNOW_KPC
        return res

    return xyzuvw_lsr - XYZUVWSOLARNOW


def convert_helio2lsr(xyzuvw_helio, kpc=False):
    """
    Convert cartesian xyzuvw position from helio-centred to LSR-centred

    Assumes position is in pc unless stated otherwise

    Parameters
    ----------
    xyzuvw_helio : [6] array
        3 position [pc] and 3 velocity with respect to the the sun
    kpc : bool {False}
        If set, then treats input units of position as kpc (as opposed to pc)

    Returns
    -------
    result : [6] array
        3 position [pc] (or [kpc] if `kpc` set to True)
        and 3 velocity with respect to the local standard of rest
    """
    XYZUVWSOLARNOW = np.array([0., 0., 25., 11.1, 12.24, 7.25])
    XYZUVWSOLARNOW_KPC = np.array([0., 0., 0.025, 11.1, 12.24, 7.25])
    if kpc:
        return xyzuvw_helio + XYZUVWSOLARNOW_KPC

    return xyzuvw_helio + XYZUVWSOLARNOW


#def convertAstrometryToLSRXYZUVW(a, d, pi, mu_a, mu_d, rv, mas=True):
def convert_astrometry2lsrxyzuvw(astro, mas=True):
    """
    Take a point straight from a catalogue, return it as XYZUVW

    This function takes astrometry in conventional units, and converts them
    into internal units for convenience.

    Parameters
    ----------
    a : (deg) right ascention
    d : (deg) declination
    pi : (mas) parallax
    mu_a : (mas/yr) proper motion in right ascension
    mu_d : (mas/yr) proper motion in declination
    rv : (km/s) line of sight velocity

    mas : Boolean {True}
        set if input parallax and proper motions are in mas

    Returns
    -------
    XYZUVW : (pc, pc, pc, km/s, km/s, km/s)
    """
    astro = np.copy(astro)
    # convert to as for internal use
    if mas:
        astro[2:5] *= 1e-3
    # logging.debug("Input (after conversion) is: {}".format(astro))
    xyzuvw_helio = convert_astrometry2helioxyzuvw(*astro)
    # logging.debug("Heliocentric XYZUVW is : {}".format(xyzuvw_helio))
    xyzuvw_lsr = convert_helio2lsr(xyzuvw_helio)

    # logging.debug("LSR XYZUVW (pc) is : {}".format(xyzuvw_lsr))
    return xyzuvw_lsr

def convert_many_astrometry2lsrxyzuvw(astr_arr, mas=True):
    """
    Take a point straight from a catalogue, return it as XYZUVW

    This function takes astrometry in conventional units, and converts them
    into internal units for convenience.

    Parameters
    ----------
    a : (deg) right ascention
    d : (deg) declination
    pi : (mas) parallax
    mu_a : (mas/yr) proper motion in right ascension
    mu_d : (mas/yr) proper motion in declination
    rv : (km/s) line of sight velocity

    mas : Boolean {True}
        set if input parallax and proper motions are in mas
    """
    logging.info("converting to LSRXYZUVW")
    xyzuvws = np.zeros(astr_arr.shape)
    for i, astr in enumerate(astr_arr):
        if (i % 1000 == 0):
            logging.info("{} of {} done".format(i, xyzuvws.shape[0]))
        xyzuvws[i] = convert_astrometry2lsrxyzuvw(astr, mas=mas)
    return xyzuvws


def convert_lsrxyzuvw2astrometry(xyzuvw_lsr):
    """
    Takes as input heliocentric XYZUVW values, returns astrometry

    Parameters
    ----------
    xyzuvw_lsr : (pc, pc, pc, km/s, km/s, km/s) array
        The position and velocity of a star in a right handed cartesian system
        corotating with and centred on the local standard of rest

    Returns
    -------
    a : (deg) right ascention
    d : (deg) declination
    pi : (mas) parallax
    mu_a : (mas/yr) proper motion in right ascension
    mu_d : (mas/yr) proper motion in declination
    rv : (km/s) line of sight velocity
    """
    xyzuvw_lsr = np.copy(xyzuvw_lsr)

    xyzuvw_helio = convert_lsr2helio(xyzuvw_lsr)
    astr = np.array(convert_helioxyzuvw2astrometry(xyzuvw_helio))

    # Finally converts angles to mas for external use
    astr[2:5] *= 1e3
    return astr


def convert_many_lsrxyzuvw2astrometry(xyzuvw_lsrs):
    """
    Takes as input heliocentric XYZUVW values, returns astrometry

    Parameters
    ----------
    xyzuvw_lsr : (pc, pc, pc, km/s, km/s, km/s) array
        The position and velocity of a star in a right handed cartesian system
        corotating with and centred on the local standard of rest

    Returns
    -------
    a : (deg) right ascention
    d : (deg) declination
    pi : (mas) parallax
    mu_a : (mas/yr) proper motion in right ascension
    mu_d : (mas/yr) proper motion in declination
    rv : (km/s) line of sight velocity
    """
    astros = np.zeros(xyzuvw_lsrs.shape)
    for i, xyzuvw_lsr in enumerate(xyzuvw_lsrs):
        astros[i] = convert_lsrxyzuvw2astrometry(xyzuvw_lsr)
    return astros
