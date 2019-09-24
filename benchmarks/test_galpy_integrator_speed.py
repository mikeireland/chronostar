'''Checks speed and accuracy of various integration methods implemented
by Galpy. On Tim's macbook, 4k_6 is fastest (6 times that of odeint, the
previous default) and has position errors of 1e-7 and velocity errors of
3e-9 km/s when calculating the motion of the LSR after 100 Myr'''
import numpy as np
import sys
import time

if sys.version[0] == '2':
    timer = time.clock
elif sys.version[0] == '3':
    timer = time.perf_counter

sys.path.insert(0, '..')
from chronostar.traceorbit import trace_cartesian_orbit

integ_methods = [
    'odeint',
    'symplec4_c',
    'rk4_c',
    'dopr54_c',
    'rk6_c',
]


xyzuvw_start = [0.,0.,25.,0.,0.,0.]
max_time = 1000
orbit_times = np.linspace(0,max_time,100)

niters = 100

print('Integrating up to {} Myr'.format(max_time))
print('Iterating {} times'.format(niters))

print('----------- Check accuracy ----------')
xyzuvw_lsr = [0.,0.,0.,0.,0.,0.]
traceback_age = 100. #Myr

for method in integ_methods:
    print('_____ Using {} _____'.format(method))
    xyzuvw_final = trace_cartesian_orbit(xyzuvw_lsr, traceback_age,
                                         single_age=True,
                                         method=method)
    diff = xyzuvw_final - xyzuvw_lsr
    pos_error = np.sqrt(np.sum(np.square(diff[:3])))
    vel_error = np.sqrt(np.sum(np.square(diff[3:])))
    print('Position error: {:.3} pc'.format(pos_error))
    print('Velocity error: {:.3} km/s'.format(vel_error))

    print('')



print('----------- Check timings ----------')
for method in integ_methods:
    print('_____ Using {} _____'.format(method))
    duration_times = []
    for i in range(niters):
        start = timer()

        trace_cartesian_orbit(xyzuvw_start, orbit_times, single_age=False,
                              method=method)

        end = timer()
        duration_times.append(end-start)
    print('Average time taken: {:.1f} ms'.format(1000*np.mean(duration_times)))
    print('Best time:          {:.1f} ms'.format(1000*np.min(duration_times)))
    print('Worst time:         {:.1f} ms'.format(1000*np.max(duration_times)))
    print('')






