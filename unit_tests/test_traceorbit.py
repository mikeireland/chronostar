"""
Tests traceorbit module. This module makes great use of Galpy so for
convenience a brief summary of Galpy coordinates is provided:

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
"""

import logging
import numpy as np
import sys

sys.path.insert(0, '..')
import chronostar.traceorbit as torb
from galpy.potential import MiyamotoNagaiPotential

LOGGINGLEVEL = logging.DEBUG
method_atols = {
    'odeint':1e-10,
    'symplec4_c':1e-5,
    'rk4_c':1e-5,
    'dopr54_c':1e-6,
    'rk6_c':1e-6,
}


def test_galpy2chron2galpy_stationary():
    """
    Check that converting from Galpy to Chronostar is internally
    consistent (you can convert back and forth)

    Time is fixed at 0
    """
    for i in range(100):
        xyzuvw_start = np.random.rand(6)
        galpy_start = torb.convert_cart2galpycoords(xyzuvw_start)
        xyzuvw_res = torb.convert_galpycoords2cart(galpy_start)
        assert np.allclose(xyzuvw_start, xyzuvw_res)

def test_galpy2chron2galpy_moving():
    """
    Check that converting from Galpy to Chronostar is internally
    consistent (you can convert back and forth)

    Time is allowed to vary
    """

    # Test first LSR
    xyzuvw_start = np.zeros(6)
    time = 1.
    galpy_start = torb.convert_cart2galpycoords(xyzuvw_start, bovy_times=time)
    # import pdb; pdb.set_trace()
    xyzuvw_res = torb.convert_galpycoords2cart(galpy_start, ts=time)
    # import pdb; pdb.set_trace()
    assert np.allclose(xyzuvw_start, xyzuvw_res)

    # Now test slightly rotated LSR
    time = np.pi/4
    galpy_stat_start = np.array([1., 0., 1., 0., 0., time])
    xyzuvw_start = torb.convert_galpycoords2cart(galpy_stat_start)
    galpy_start = torb.convert_cart2galpycoords(xyzuvw_start, bovy_times=time)
    xyzuvw_res = torb.convert_galpycoords2cart(galpy_start, ts=time)
    assert np.allclose(xyzuvw_start, xyzuvw_res)

    # Now test points randomly dispersed near the LSR
    # but at a time range corresponding to LSR rotation of [0-1] rad
    for i in range(100):
        xyzuvw_start = np.random.rand(6)
        # Time is in galpy units
        time = np.random.rand()
        galpy_start = torb.convert_cart2galpycoords(xyzuvw_start, bovy_times=time)
        # import pdb; pdb.set_trace()
        xyzuvw_res = torb.convert_galpycoords2cart(galpy_start, ts=time)
        # import pdb; pdb.set_trace()
        assert np.allclose(xyzuvw_start, xyzuvw_res)


def test_LSR():
    """
    Check that LSR remains constant in our frame of reference.

    Since our frame of reference is **centred** on the LSR, then the LSR
    should remain at the origin.
    """
    xyzuvw_lsr = [0.,0.,0.,0.,0.,0.]
    times = np.linspace(0,100,101)

    for method, atol in method_atols.items():

        xyzuvws = torb.trace_cartesian_orbit(xyzuvw_lsr, times,
                                             single_age=False,
                                             method=method)
        assert np.allclose(xyzuvws[0,:5],xyzuvws[-1,:5], atol=atol), 'Method {}'.format(method)


def test_rotatedLSR():
    """
    Check that LSRs with different azimuthal positions also remain constant
    """
    # method_atols = {'odeint':1e-11, 'dopr54_c':1e-6}
    for method, atol in method_atols.items():
        rot_lsr_gp_coords = np.array([1., 0., 1., 0., 0., np.pi])
        xyzuvw_rot_lsr = torb.convert_galpycoords2cart(rot_lsr_gp_coords)
        ### xyzuvw_rot_lsr = [16000., 0, 0, 0, 0, 0,]
        times = np.linspace(0,100,101)
        xyzuvws = torb.trace_cartesian_orbit(xyzuvw_rot_lsr, times,
                                             single_age=False,
                                             method=method)

        # On a circular orbit, same radius as LSR, so shouldn't vary at all
        # assert np.allclose(xyzuvws[0,:5],xyzuvws[-1,:5])
        assert np.allclose(xyzuvws[0],xyzuvws[-1], atol=atol), 'Method {}'.format(method)

        # Should be initialised on opposite side of galaxy (X = 16kpc)
        assert np.allclose(16000., xyzuvws[0,0])

def test_singleTime():
    """Test usage where we only provide the desired age, and not an array

    Good demo of how to traceback 2 stars forward through time, either
    with an array of time steps, or a single age
    """
    xyzuvw_1 = [0.,0.,25.,0.,0.,0.]
    xyzuvw_2 = [0.,0.,0.,0.,-10.,0.]
    xyzuvws = np.vstack((xyzuvw_1, xyzuvw_2))
    age = 10.
    times = np.linspace(0., age, 2)

    # get position for each time in times
    xyzuvws_both = torb.trace_many_cartesian_orbit(xyzuvws, times, single_age=False)

    # get position for *age* only
    xyzuvws_now = torb.trace_many_cartesian_orbit(xyzuvws, age, single_age=True)
    assert np.allclose(xyzuvws_both[:,1], xyzuvws_now)

    xyzuvw_both = torb.trace_cartesian_orbit(xyzuvws[0], times, single_age=False)
    xyzuvw_now = torb.trace_cartesian_orbit(xyzuvws[0], age, single_age=True)
    assert np.allclose(xyzuvw_both[1], xyzuvw_now)


def test_invertedAngle():
    times = np.array([2*np.pi, 0., -2*np.pi])
    lsr_galpy_plus_cycle = np.array([1,0,1,0,0,2*np.pi])
    lsr_galpy = np.array([1.,0,1,0,0,0])
    lsr_galpy_minus_cycle = np.array([1,0,1,0,0,-2*np.pi])
    galpy_pos = np.vstack((lsr_galpy_minus_cycle, lsr_galpy, lsr_galpy_plus_cycle))

    my_xyzuvw = torb.convert_galpycoords2cart(galpy_pos, times)
    assert np.allclose(my_xyzuvw, np.zeros((len(times), 6)))

    ntimes = 9
    semi_times = np.linspace(-2*np.pi, 2*np.pi, ntimes)

    galpy_pos = np.repeat(lsr_galpy, ntimes).reshape(6,ntimes).T
    galpy_pos[:,-1] = semi_times
    my_xyzuvw = torb.convert_galpycoords2cart(galpy_pos, semi_times)
    assert np.allclose(my_xyzuvw, np.zeros((ntimes, 6)))

def test_traceforwardThenBack():
    """Check that tracing a point forward then back for the same time step
    returns initial position
    """
    ABS_TOLERANCE = 1e-3
    xyzuvws = np.array([
        [0.,0.,25.,0.,0.,0.],
        # [10.,0.,-50.,0.,0.,0.],
        # [0.,0.,0.,10.,25.,30.,],
    ])
    age = 100.
    times = np.linspace(0,100,1001)
    for xyzuvw_start in xyzuvws:
        xyzuvw_end = torb.trace_cartesian_orbit(xyzuvw_start,
                                                times=age,
                                                single_age=True,
                                                )
        xyzuvw_start_again = torb.trace_cartesian_orbit(xyzuvw_end,
                                                        times=-age,
                                                        single_age=True,
                                                        )
        assert np.allclose(xyzuvw_start, xyzuvw_start_again,
                           atol=ABS_TOLERANCE)

def test_different_potential():
    miya_pot = MiyamotoNagaiPotential(a=0.5, b=0.0375, amp=1., normalize=1.)
    miya_trace_cartesian_orbit = torb.trace_orbit_builder(miya_pot)

    ABS_TOLERANCE = 1e-3
    xyzuvws = np.array([
        [0., 0., 25., 0., 0., 0.],
        [10., 0., -50., 0., 0., 0.],
        [10., 0., -50., 0., 0., -5.],
        [0., 0., 0., 10., 25., 30.,],
    ])
    age = 100.
    times = np.linspace(0, 100, 1001)
    for xyzuvw_start in xyzuvws:
        xyzuvw_end = miya_trace_cartesian_orbit(xyzuvw_start,
                                                times=age,
                                                single_age=True,
                                                )
        xyzuvw_start_again = miya_trace_cartesian_orbit(xyzuvw_end,
                                                        times=-age,
                                                        single_age=True,
                                                        )

        assert np.allclose(xyzuvw_start, xyzuvw_start_again,
                           atol=ABS_TOLERANCE)

        # Confirm that tracing forward with one potential but back with another
        # gives different starting position
        xyzuvw_start_but_with_diff_pot = torb.trace_cartesian_orbit(xyzuvw_end,
                                                                    times=-age,)
        assert not np.allclose(xyzuvw_start, xyzuvw_start_but_with_diff_pot)

    def test_traceforwardThenBack():
        """Check that tracing a point forward then back for the same time step
        returns initial position
        """
        return
        ABS_TOLERANCE = 1e-3
        xyzuvws = np.array([
            [0., 0., 25., 0., 0., 0.],
            # [10.,0.,-50.,0.,0.,0.],
            # [0.,0.,0.,10.,25.,30.,],
        ])
        age = 100.
        times = np.linspace(0, 100, 1001)
        for xyzuvw_start in xyzuvws:
            galpy_start = None
            xyzuvw_end = torb.trace_cartesian_orbit(xyzuvw_start,
                                                    times=age,
                                                    single_age=True,
                                                    )
            xyzuvw_start_again = torb.trace_cartesian_orbit(xyzuvw_end,
                                                            times=-age,
                                                            single_age=True,
                                                            )
            assert np.allclose(xyzuvw_start, xyzuvw_start_again,
                               atol=ABS_TOLERANCE)

    def test_galpy_stationary_conversions():
        """Check if gaply conversions behave as expected where everything
        is at time 0"""

        # Test LSR
        lsr_chron = np.zeros(6)
        lsr_galpy = np.array([1., 0, 1, 0, 0, 0])

        assert np.allclose(lsr_chron,
                           torb.convert_galpycoords2cart(lsr_galpy, ts=0.))
        assert np.allclose(lsr_galpy,
                           torb.convert_cart2galpycoords(lsr_chron, ts=0.))

        # Test galactic centre
        gc_chron = np.array([8000., 0, 0, 0, -220., 0, ])
        gc_galpy = np.ones(6) * 1e-15

        assert np.allclose(gc_chron,
                           torb.convert_galpycoords2cart(gc_galpy, ts=0.))
        assert np.allclose(gc_galpy,
                           torb.convert_cart2galpycoords(gc_chron, ts=0.))

        # Test simple, off origin point
        off_chron = np.array([4000, 8000. * np.sqrt(3) / 2, 0,
                              np.sin(np.pi / 3) * 220.,
                              -np.cos(np.pi / 3) * 220.,
                              0])
        off_galpy = np.array([1., 0, 1, 0, 0, np.pi / 3.])

        assert np.allclose(off_galpy,
                           torb.convert_cart2galpycoords(off_chron, ts=0.))
        assert np.allclose(off_chron,
                           torb.convert_galpycoords2cart(off_galpy, ts=0.))

        # Test random positions
        SPREAD = 100000
        NSAMPLES = int(1e6)
        many_pos_chron = (np.random.rand(NSAMPLES, 6) - 0.5) * SPREAD  # uniform between -10 and 10
        many_pos_galpy = torb.convert_cart2galpycoords(many_pos_chron, ts=0.)

        assert np.allclose(many_pos_chron,
                           torb.convert_galpycoords2cart(many_pos_galpy, ts=0.),
                           atol=1e-2)

    def test_galpy_moving_conversions():
        """Check if gaply conversions behave as expected where time
        is allowed to vary."""
        lsr_chron = np.zeros(6)
        lsr_galpy = np.array([1., 0, 1, 0, 0, 0])
        # Incorporate positive time into lsr position checks
        NSTEPS = 10
        galpy_times = np.linspace(0., 2 * np.pi, NSTEPS)
        lsrs_chron = np.repeat(lsr_chron, NSTEPS).reshape(6, -1).T
        lsrs_galpy = np.repeat(lsr_galpy, NSTEPS).reshape(6, -1).T
        lsrs_galpy[:, -1] = galpy_times
        chron_times = torb.convert_bovytime2myr(galpy_times)

        assert np.allclose(
            lsrs_chron,
            torb.convert_galpycoords2cart(lsrs_galpy, ts=galpy_times))
        assert np.allclose(
            lsrs_galpy,
            torb.convert_cart2galpycoords(lsrs_chron, ts=chron_times)
        )

        # Incorporate negative time into lsr position checks
        galpy_times = np.linspace(0., -2 * np.pi, NSTEPS)
        lsrs_chron = np.repeat(lsr_chron, NSTEPS).reshape(6, -1).T
        lsrs_galpy = np.repeat(lsr_galpy, NSTEPS).reshape(6, -1).T
        lsrs_galpy[:, -1] = galpy_times
        chron_times = torb.convert_bovytime2myr(galpy_times)

        assert np.allclose(
            lsrs_chron,
            torb.convert_galpycoords2cart(lsrs_galpy, ts=galpy_times))
        assert np.allclose(
            lsrs_galpy,
            torb.convert_cart2galpycoords(lsrs_chron, ts=chron_times)
        )

        # Test random positions with random times
        SPREAD = int(1e4)  # pc
        NSAMPLES = 100
        many_pos_chron = (np.random.rand(NSAMPLES, 6) - 0.5) * SPREAD  # uniform between -10 and 10
        many_chron_times = np.random.rand(NSAMPLES) * 100  # Myr
        many_pos_galpy = torb.convert_cart2galpycoords(
            many_pos_chron, ts=many_chron_times
        )
        many_galpy_times = torb.convert_myr2bovytime(many_chron_times)

        for i in range(NSAMPLES):
            assert np.allclose(many_pos_chron[i],
                               torb.convert_galpycoords2cart(
                                   many_pos_galpy[i], ts=many_galpy_times[i]
                               ),
                               atol=1e-2)

    def test_careful_traceback_and_forward():
        """Step by step, project orbit forward, then backward"""
        bovy_times = np.array([0., np.pi / 3.])
        chron_times = torb.convert_bovytime2myr(bovy_times)

        init_pos_chron = np.array([
            4000, 8000. * np.sqrt(3) / 2, 0,
                  np.sin(np.pi / 3) * 220.,
                  -np.cos(np.pi / 3) * 220.,
            0
        ])
        init_pos_galpy = torb.convert_cart2galpycoords(init_pos_chron, ts=0.)

        assert np.allclose(np.array([1., 0, 1, 0, 0, np.pi / 3.]),
                           init_pos_galpy)

        o = Orbit(vxvv=init_pos_galpy, ro=8., vo=220.)
        o.integrate(bovy_times, MWPotential2014, method='odeint')

        orbit_galpy = o.getOrbit()
        assert np.allclose(init_pos_galpy, orbit_galpy[0])
        assert np.allclose(init_pos_galpy
                           + np.array([0., 0., 0., 0., 0., bovy_times[-1]]),
                           orbit_galpy[-1])

        orbit_chron = torb.convert_galpycoords2cart(orbit_galpy,
                                                    ts=bovy_times)
        assert np.allclose(init_pos_chron, orbit_chron[0])
        assert np.allclose(init_pos_chron,
                           orbit_chron[-1])

        # Setup for backwards time integration
        # Currently at time of PI/3
        back_init_pos_chron = orbit_chron[-1]
        back_init_pos_galpy = torb.convert_cart2galpycoords(
            back_init_pos_chron,
            bovy_times=bovy_times[-1],
        )

        assert np.allclose(back_init_pos_galpy,
                           torb.convert_cart2galpycoords(
                               back_init_pos_chron,
                               bovy_times=bovy_times[-1]
                           ))

        back_o = Orbit(vxvv=back_init_pos_galpy, ro=8., vo=220.)
        back_o.integrate(-1 * bovy_times, MWPotential2014, method='odeint')

        back_orbit_galpy = back_o.getOrbit()

        assert np.allclose(back_init_pos_galpy, back_orbit_galpy[0])
        assert np.allclose(back_init_pos_galpy
                           - np.array([0., 0., 0., 0., 0., bovy_times[-1]]),
                           back_orbit_galpy[-1])
        assert np.allclose(init_pos_galpy, back_orbit_galpy[-1])

        back_orbit_chron = torb.convert_galpycoords2cart(
            back_orbit_galpy,
            ts=bovy_times[::-1],
        )

        assert np.allclose(init_pos_chron, back_orbit_chron[-1])

    def test_traceback_and_forward():
        """The test that shows things are broken"""
        time = 10.  # Myr
        times = np.array([0., 10.])
        init_pos_chron = np.array([10., 0., 30., 0., 0., 0.])
        # init_pos_chron = np.zeros(6)
        init_pos_chron = np.array([4000, 8000. * np.sqrt(3) / 2, 0,
                                   np.sin(np.pi / 3) * 220.,
                                   -np.cos(np.pi / 3) * 220.,
                                   0])
        final_pos_chron = torb.trace_cartesian_orbit(
            init_pos_chron, times=times, single_age=False)[-1]
        final_pos_chron = torb.traceforward_from_now(
            init_pos_chron, time=time,
        )
        assert np.allclose(
            init_pos_chron,
            torb.traceback_to_now(
                final_pos_chron, time,
            ),
            atol=1e-5
        )

    def test_multi_traceback_and_forward():
        np.random.seed(0)
        NPOSITIONS = 10
        init_positions = np.random.rand(NPOSITIONS, 6) * 20 - 10
        time_spans = np.random.rand(NPOSITIONS) * 30
        for pos, time in zip(init_positions, time_spans):
            final_pos = torb.traceforward_from_now(pos, time)
            init_pos = torb.traceback_to_now(final_pos, time)
            assert np.allclose(pos, init_pos, atol=1e-3)

        for pos, time in zip(init_positions, time_spans):
            print('time: {}'.format(time))
            final_pos = torb.traceback_from_now(pos, time)
            init_pos = torb.traceforward_to_now(final_pos, time)
            assert np.allclose(pos, init_pos, atol=1e-3)

    def test_interval_tracing():
        np.random.seed(0)
        start = np.random.rand(6) * 20 - 10

        time_steps = [3., -10., -3., 10]

        current_pos = start
        for time_step in time_steps:
            current_pos = torb.base_trace_cartesian_orbit(
                current_pos,
                end_time=time_step,
            )
        assert np.allclose(start, current_pos, atol=1e-3)

    def test_interval_tracing_orig():
        np.random.seed(0)
        start = np.random.rand(6) * 20 - 10

        time_steps = [3., -10., -3., 10]

        current_pos = start
        for time_step in time_steps:
            current_pos = torb.trace_cartesian_orbit(
                current_pos,
                times=time_step,
                single_age=True,
            )
        assert np.allclose(start, current_pos, 1e-3)

if __name__ == '__main__':
    test_different_potential()
