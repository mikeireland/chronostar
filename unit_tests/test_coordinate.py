"""
Testing the coordinates.py moduel - a utility module for converting
between various coordinate systems
"""
import numpy as np
import sys

from astropy.coordinates import SkyCoord

#from galpy.util import bovy_coords
import logging

sys.path.insert(0,'..')

import chronostar.coordinate as cc

# Cartesian position and velocity of the sun with respect to the LSR
# as reported by Schoenrich et al.
XYZUVWSOLARNOW_pc = np.array([0., 0., 25., 11.1, 12.24, 7.25])

def test_calcEQToGCMatrix():
    """
    Compare generated matrices with those reported by
    Johnson and Soderblom (1987)
    """
    old_a_ngp = 192.25
    old_d_ngp = 27.4
    old_th = 123.

    # Matrix copied from paper to take sky positions and parallax
    # to heliocentric Galactic cartesian coordinates
    js1987 = np.array([
        [-0.06699, -0.87276, -0.48354],
        [0.49273, -0.45035, 0.74458],
        [-0.86760, -0.18837, 0.46020],
    ])

    assert np.allclose(
        js1987,
        cc.calc_eq2gc_matrix(old_a_ngp, old_d_ngp, old_th),
        rtol=1e-4
    )

    js1987_inv = np.linalg.inv(js1987)
    assert np.allclose(
        js1987_inv,
        cc.calc_gc2eq_matrix(old_a_ngp, old_d_ngp, old_th),
        rtol=1e-4
    )

    assert np.allclose(
        np.dot(
            cc.calc_eq2gc_matrix(old_a_ngp, old_d_ngp, old_th),
            cc.calc_gc2eq_matrix(old_a_ngp, old_d_ngp, old_th),
        ), np.eye(3),
        rtol=1e-4
    )


def test_CartesianAngleConversions():
    """Take position angles to cartesian vectors (on unit sphere) and back"""
    sphere_pos_list = [
        (0,0),
        (45,0),
        (315,0),
        (0,45),
        (0,90),
    ]

    cart_list = [
        (1,0,0),
        (np.sqrt(0.5), np.sqrt(0.5), 0),
        (np.sqrt(0.5), -np.sqrt(0.5), 0),
        (np.sqrt(0.5), 0, np.sqrt(0.5)),
        (0, 0, 1)
    ]

    for sphere_pos, cart in zip(sphere_pos_list, cart_list):
        cart = cc.convert_angles2cartesian(*sphere_pos)
        sphere_pos2 = cc.convert_cartesian2angles(*cart)
        assert np.allclose(sphere_pos, sphere_pos2)


def test_convertEquatorialToGalactic():
    """Take equatorial coords to galactic and back"""
    pos_ncp_list = [
        (20,85),
        (30,20),
        (260, -30),
        (100, -60)
    ]
    for pos_ncp in pos_ncp_list:
        pos_ncp_gc = cc.convert_equatorial2galactic(*pos_ncp)
        pos_ncp2 = cc.convert_galactic2equatorial(*pos_ncp_gc)
        assert np.allclose(pos_ncp, pos_ncp2)

def test_famousPositions():
    # galactic north pole in two coordinate systems
    gnp_eq = (192.8595, 27.1283)
    gnp_gc = (0,90)

    # testing gnp_eq --> gnp_gc is tricky, since there is a degeneracy,
    # so only compare the second coordinate
    assert np.allclose(
        cc.convert_equatorial2galactic(*gnp_eq)[1],
        gnp_gc[1], rtol=1e-4
    )
    assert np.allclose(
        cc.convert_galactic2equatorial(*gnp_gc),
        gnp_eq, rtol=1e-4
    )

    # beta pic
    bp_eq = [86.82125, -51.0664]
    bp_gc = [258.3638, -30.6117]
    assert np.allclose(cc.convert_equatorial2galactic(*bp_eq), bp_gc)
    assert np.allclose(cc.convert_galactic2equatorial(*bp_gc), bp_eq)

def test_convertPMToSpaceVelocity():
    """
    Check conversion of proper motion + radial velocity to UVW.

    Use famous star Beta Pictoris
    """
    # astrometry from wikiepdia
    astr_bp_ref = [
        86.82125, #deg
        -51.0664, #deg
        0.05144,  #as
        0.00465,  #as/yr
        0.0831,   #as/yr
        20.0      #km/s
    ]
    # kinematics (from Mamajek & Bell 2014)
    xyzuvw_bp_ref = np.array([-3.4, -16.4, -9.9, -11.0, -16.0, -9.1])

    # check UVW
    uvw = cc.convert_pm2heliospacevelocity(*astr_bp_ref)
    assert np.allclose(xyzuvw_bp_ref[3:], uvw, rtol=1e-2)

    # check pmrv conversion given position, parallax and (heliocentric) UVW
    pos_uvw_bp = np.hstack((astr_bp_ref[:3], xyzuvw_bp_ref[3:]))
    logging.debug("Input: {}".format(pos_uvw_bp))
    logging.debug("Input: {} {} {} {} {} {}".format(*pos_uvw_bp))
    pms_bp = cc.convert_heliospacevelocity2pm(*pos_uvw_bp)
    assert np.allclose(astr_bp_ref[3:], pms_bp, rtol=1e-1)

def test_convertHelioXYZUVWToAstrometry():
    astr_bp = [ # astrometry from wikiepdia
        86.82125, #deg
        -51.0664, #deg
        0.05144,  #as
        0.00465,  #as/yr
        0.0831,   #as/yr
        20.0      #km/s
    ]
    xyzuvw_bp_helio = np.array([-3.4, -16.4, -9.9, -11.0, -16.0, -9.1])
    calculated_astr_bp = cc.convert_helioxyzuvw2astrometry(xyzuvw_bp_helio)
    assert np.allclose(astr_bp, calculated_astr_bp, rtol=1e-2)

    calculated_xyzuvw_bp_helio = cc.convert_astrometry2helioxyzuvw(
        *astr_bp
    )
    assert np.allclose(calculated_xyzuvw_bp_helio, xyzuvw_bp_helio, rtol=1e-2)

def test_convertAstrometryToLSRXYZUVW():
    """Check edge case of nearby sun, also compare with output of astropy"""
    # pick an astrometry which should be right near the sun
    sun_astro = (0., 90., 1e15, 0.,0.,0.)
    sun_xyzuvw_lsr = (0., 0., 25., 11.1, 12.24, 7.25)
    assert np.allclose(
        cc.convert_astrometry2lsrxyzuvw(sun_astro), sun_xyzuvw_lsr
    )

    # TODO: compare astropy coordinates
    # will do this when upgraded to python 3
    star_astros = np.array([
        [86.82, -51.067, 51.44, 4.65, 83.1, 20],        # beta Pic
        [165.466, -34.705, 18.62, -66.19, -13.9, 13.4], # TW Hya
        [82.187, -65.45, 65.93, 33.16, 150.83, 32.4],   # AB Dor
        [100.94, -71.977, 17.17, 6.17, 61.15, 20.7]     # HIP 32235
    ])


def test_origin():
    """Checks that the sun can be put in without breaking angle calculations"""
    xyzuvw_solar_helio = np.zeros(6) # = [0.,0.,25.,11.1,12.24,7.25]
    astro = cc.convert_helioxyzuvw2astrometry(xyzuvw_solar_helio)
    assert not np.any(np.isnan(astro))


def test_convertLSRXYZUVWToAstrometry():
    """Checks Beta Pictoris is accurately handled"""
    astr_bp = [ # astrometry from wikiepdia
        86.82125, #deg
        -51.0664, #deg
        51.44,  #mas
        4.65,  #mas/yr
        83.1,   #mas/yr
        20.0      #km/s
    ]
    xyzuvw_bp_helio = np.array([-3.4, -16.4, -9.9, -11.0, -16.0, -9.1])
    xyzuvw_bp_lsr = cc.convert_helio2lsr(xyzuvw_bp_helio)

    calculated_astr_bp = cc.convert_lsrxyzuvw2astrometry(xyzuvw_bp_lsr)
    assert np.allclose(astr_bp, calculated_astr_bp, rtol=1e-2)

    calculated_xyzuvw_bp_lsr = cc.convert_astrometry2lsrxyzuvw(astr_bp)
    assert np.allclose(calculated_xyzuvw_bp_lsr, xyzuvw_bp_lsr, rtol=0.15)

    calculated_astr_bp2 = cc.convert_lsrxyzuvw2astrometry(xyzuvw_bp_lsr)
    calculated_xyzuvw_bp_lsr2 = cc.convert_astrometry2lsrxyzuvw(
        calculated_astr_bp2
    )
    assert np.allclose(calculated_xyzuvw_bp_lsr2, xyzuvw_bp_lsr)

    xyzuvw_bp_lsr_many = np.array([xyzuvw_bp_lsr for _ in range(10)])
    astr_bp_many = cc.convert_lsrxyzuvw2astrometry(xyzuvw_bp_lsr_many)

    assert np.allclose(calculated_astr_bp, astr_bp_many[0])


def test_convertManyLSRXYZUVWToAstrometry():
    return
    xyzuvw_bp_helio = np.array([-3.4, -16.4, -9.9, -11.0, -16.0, -9.1])
    xyzuvw_bp_lsr =  xyzuvw_bp_helio + XYZUVWSOLARNOW_pc
    astr_bp = [ # astrometry from wikiepdia
        86.82125, #deg
        -51.0664, #deg
        51.44,  #mas
        4.65,  #mas/yr
        83.1,   #mas/yr
        20.0      #km/s
    ]

    xyzuvw_lsrs = np.array([
        xyzuvw_bp_lsr,
        xyzuvw_bp_lsr,
        xyzuvw_bp_lsr,
    ])
    astros = np.array([
        astr_bp,
        astr_bp,
        astr_bp,
    ])
    calculated_astros = cc.convert_many_lsrxyzuvw2astrometry(xyzuvw_lsrs)
    assert np.allclose(calculated_astros, astros, rtol=1e-2)

    recalculated_xyzuvws = cc.convert_many_astrometry2lsrxyzuvw(calculated_astros)
    assert np.allclose(recalculated_xyzuvws, xyzuvw_lsrs)



def test_internalConsistency():
    '''
    Take a starting LSR cartesian mean, convert to RA, DEC directly and
    indirectly.
    '''
    start_position_lsr = np.array([120.7, -18.5, 74.9, 5.4, -4., 0.5])

    # Convert directly to equatorial
    direct_ra, direct_dec =\
        cc.convert_lsrxyzuvw2astrometry(start_position_lsr)[:2]

    # Convert to helio
    start_position_helio = cc.convert_lsr2helio(start_position_lsr)[:3]
    # Convert to galactic coordiantes
    theta_gal, phi_gal = cc.convert_cartesian2angles(*start_position_helio)
    # Convert to equatorial

    indirect_ra, indirect_dec = cc.convert_galactic2equatorial(theta_gal,
                                                               phi_gal)

    assert np.isclose(direct_ra, indirect_ra)
    assert np.isclose(direct_dec, indirect_dec)

if __name__ == '__main__':
    #test_convertPMToSpaceVelocity()
    test_convertLSRXYZUVWToAstrometry()
