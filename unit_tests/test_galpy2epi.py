"""
REWRITE BECAUSE USING "RETIRED" CODE!
"""

import numpy as np

import sys
sys.path.insert(0, '..')
import chronostar.traceorbit as torb
# import chronostar.retired2.epicyclic_via_galpy as eg

torb.MIKES_IMP = False

### def test_x_offset():
###     x_offset = torb.convert_cart2galpycoords([10.,0.,0.,0.,0.,0.])
###     assert np.allclose(x_offset, eg.convert_epi2galpy(eg.convert_galpy2epi(x_offset)))
###
### def test_y_offset():
###     y_offset = torb.convert_cart2galpycoords([0.,10.,0.,0.,0.,0.])
###     y_res = eg.convert_galpy2epi(y_offset)
###     assert np.allclose(y_offset, eg.convert_epi2galpy(eg.convert_galpy2epi(y_offset)))
###
### def test_z_offset():
###     z_offset = torb.convert_cart2galpycoords([0.,0.,10.,0.,0.,0.])
###     assert np.allclose(z_offset, eg.convert_epi2galpy(eg.convert_galpy2epi(z_offset)))
###
### def test_u_offset():
###     u_offset = torb.convert_cart2galpycoords([0.,0.,0.,1.,0.,0.])
###     assert np.allclose(u_offset, eg.convert_epi2galpy(eg.convert_galpy2epi(u_offset)))
###
### def test_v_offset():
###     rtol = 1e-4
###     v_offset = torb.convert_cart2galpycoords([0.,0.,0.,0.,1.,0.])
###     assert np.allclose(v_offset, eg.convert_epi2galpy(eg.convert_galpy2epi(v_offset)),
###                        rtol=rtol)
###
### def test_w_offset():
###     w_offset = torb.convert_cart2galpycoords([0.,0.,0.,0.,0.,1.])
###     assert np.allclose(w_offset, eg.convert_epi2galpy(eg.convert_galpy2epi(w_offset)))
###
### def test_misc():
###     POS_SPAN = 10
###     VEL_SPAN = 10
###     for i in range(100):
###         # Generate random stars with chron coords between [SPAN, SPAN]
###         rand_xyz = 2*POS_SPAN*np.random.rand(3) - POS_SPAN
###         rand_uvw = 2*VEL_SPAN*np.random.rand(3) - VEL_SPAN
###         rand_xyzuvw = np.hstack((rand_xyz, rand_uvw))
###         rand_galpy  = torb.convert_cart2galpycoords(rand_xyzuvw)
###         rand_epi    = eg.convert_galpy2epi(rand_galpy)
###         galpy_res   = eg.convert_epi2galpy(rand_epi)
###         xyzuvw_res  = torb.convert_galpycoords2cart(galpy_res)
###         try:
###             # Positions shouldn't vary by more than 3 pc
###             pos_atol = 2
###             # Velocities shouldn't vary by more than 0.05 km/s
###             vel_atol = 0.03

###            assert np.allclose(rand_xyzuvw[:3], xyzuvw_res[:3], atol=pos_atol)
###            assert np.allclose(rand_xyzuvw[3:], xyzuvw_res[3:], atol=vel_atol)
###            # assert np.allclose(rand_galpy, eg.convert_epi2galpy(rand_epi), rtol=rtol)
###        except AssertionError:
###            print(i)
###            import pdb; pdb.set_trace()

### def test_evolve_epispace():
###     """
###     Compare orbits evolved with both galpy and epicyclic, comparing in epi space
###     """
###     pass
###
### def test_evolve_chronspace():
###     """
###     Compare orbits evolved with both galpy and epicyclic, comparing in chron space
###     """
###     time = 50. #Myr
###     btime = torb.convert_myr2bovytime(time)
###     chron_start = np.array([0.,0.,10.,-2.,0.,0.])
###     galpy_start = torb.convert_cart2galpycoords(chron_start)
###     epi_start = eg.convert_galpy2epi(galpy_start)
###
###     # Just make sure the starting point can be transformed back and forth
###     assert np.allclose(galpy_start, eg.convert_epi2galpy(epi_start))
###     assert np.allclose(chron_start, torb.convert_galpycoords2cart(
###                                         eg.convert_epi2galpy(epi_start)))
###
###     epi_end = eg.evolve_epi(epi_start, time)[0]
###     galpy_end = torb.trace_galpy_orbit(galpy_start, times=time, single_age=True)
###     chron_end = torb.trace_cartesian_orbit(chron_start, times=time, single_age=True)
###
###     # Chronostar orbit end point, in galpy units
###     chron_end_gu = torb.convert_cart2galpycoords(chron_end, ts=time)
###
###     # This should be exact, because algorithmically it's the same thing.
###     assert np.allclose(galpy_end, chron_end_gu)
###
### #     # Epicyclic orbit end point, in chronostar units
### #     epi_end_chron = torb.convert_galpycoords2cart(eg.convert_epi2galpy(epi_end),
### #                                                   ts=btime)
#     # assert position accurate within 15 pc
#     # NOTE this is quite large... offset mainly in X. Maybe something to
#     # investigate.
#     assert np.allclose(chron_end[:3], epi_end_chron[:3], atol=15.)
#     # assert velocity accurate within 0.5 km/s
#     assert np.allclose(chron_end[3:], epi_end_chron[3:], atol=.5)


