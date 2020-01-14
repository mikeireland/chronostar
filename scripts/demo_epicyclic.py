"""
Trace a bunch of orbits with mutliple methods,
then plot orbits in different coordinate spaces.

NOTE: the following subdirectories will be created automatically for plots:
- chron_orbits
- epicyclic_orbits
- galpy_orbits
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '..')
from chronostar import epicyclic_via_galpy as eg
from chronostar import traceorbit as torb


def plot_galpy_orbit(galpy_start, space='chron'):
    """
    This function is a flaming pile of trash. It does the job, it\s not pretty,
    and it's 11:19pm so Tim is very tired. @_@

    Plot a galpy orbit and an epicyclic orbit, in the coordinate space
    specificed by `space`.

    Note that I've introduced brand new behavior for the 'chron' (cartesian)
    for this funciton only. Orbits are here not co-rotating, but rather the
    LSR's orbit is also calculated, and we plot the difference.

    This is so when plotting things in cartesian space, the small oscillations
    aren't swamped by the star travelling 16000 kpc after 100 Myr.
    """
    plot_dir = '{}_orbits'.format(space)
    if not os.path.exists(plot_dir):
        os.mkdir(plot_dir)

    LSR_CENT = True
    galpy_start = np.array(galpy_start)

    endtime = 150
    times = np.linspace(0, endtime, 101)  # Myr

    epi_start = eg.convert_galpy2epi(galpy_start)
    galpy_orbit = torb.trace_galpy_orbit(galpy_start, times=times,
                                         single_age=False)

    ### Left over from when trialling not rotating Chronostar. Needed a reference
#     galpy_ref = torb.trace_galpy_orbit([1., 0, 1, 0, 0, 0], times=times,
#                                        single_age=False)
#     galpy_ref = np.array(
#             [torb.convert_galpycoords2cart(gr, lsr_centered=LSR_CENT)
#              for gr in galpy_ref])

    epi_orbit = eg.evolve_epi(epi_start, time=times)

    if space == 'epicyclic':
        epi_points = np.array(epi_orbit)
        galpy_points = []
        for i, t in enumerate(times):
            galpy_points.append(eg.convert_galpy2epi(galpy_orbit[i]))
        galpy_points = np.array(galpy_points)

    elif space == 'galpy':
        galpy_points = np.array(galpy_orbit)
        epi_points = []
        for i, t in enumerate(times):
            epi_points.append(eg.convert_epi2galpy(epi_orbit[i]))
        epi_points = np.array(epi_points)

        # scale up galpy velocities
        galpy_points[:, 1:3] *= 220
        galpy_points[:, 4] *= 220
        # scale up galpy positions
        galpy_points[:, 0] *= 8000
        galpy_points[:, 3] *= 8000
        # scale up epi velocities
        epi_points[:, 1:3] *= 220
        epi_points[:, 4] *= 220
        # scale up epi positions
        epi_points[:, 0] *= 8000
        epi_points[:, 3] *= 8000
    else:  # 'chron'
        epi_points = []
        galpy_points = []
        galpy_points = torb.convert_galpycoords2cart(
                galpy_orbit, ts=torb.convert_myr2bovytime(times),
                lsr_centered=LSR_CENT)
        # chron_start = torb.convert_galpycoords2cart(galpy_start)

        for i, t in enumerate(times):
            # Original: evolve through epi, then convert back to cart
            temp_epi = eg.convert_epi2galpy(epi_orbit[i])
            epi_points.append(torb.convert_galpycoords2cart(
                    temp_epi, ts=torb.convert_myr2bovytime(t),
                    lsr_centered=LSR_CENT))

            # FUCHS 2006, calculate result directly in cart space
            # epi_points.append(trace_cartesian_makarov(chron_start, t))

            #            temp_galpy = convert_galpy2epi(galpy_orbit[i])
            #            temp_galpy = convert_epi2galpy(temp_galpy)
            # Convert galpy orbit to chron space
            temp_galpy = torb.convert_galpycoords2cart(galpy_orbit[i],
                                                       ts=torb.convert_myr2bovytime(
                                                           t),
                                                       lsr_centered=LSR_CENT)
            galpy_points[i] = temp_galpy
        epi_points = np.array(epi_points)
        galpy_points = np.array(galpy_points)

    fig=plt.figure(figsize=[18,5])

    if space == 'epicyclic':
        e_labels = ['$R_g$', '$T_g$', '$b_0$', '$\Phi_b$', '$Z$', '$\Phi_z$']
        epi_comb = [(0, 4), (0, 2), (3,2), (3, 1), (3, 5)]
        labels = e_labels
        comb = epi_comb
    elif space == 'galpy':
        g_labels = ['$R$', '$V_R$', '$V_T$', '$Z$', '$V_Z$', '$\phi$', ]
        galpy_comb = [(0, 1), (0, 2), (1, 2), (3, 4), (0,5)]
        labels = g_labels
        comb = galpy_comb
    else:
        chron_comb = [(0, 1), (0, 3), (1, 4), (0, 2), (3,4), (2, 5)]
        chron_labels = 'XYZUVW'
        labels = chron_labels
        comb = chron_comb
    nplots = len(comb) + 1

    for i, c in enumerate(comb):
        ax=fig.add_subplot(1, nplots, i+1)

        ax.plot(galpy_points[:, c[0]], galpy_points[:, c[1]], c='y', label='T\'s galpy')
        ax.plot(epi_points[:, c[0]], epi_points[:, c[1]], c='purple', label='T\'s epi',
                ls='--')
        ax.set_xlabel(labels[c[0]])
        ax.set_ylabel(labels[c[1]])

        ax.set_title('{} vs {}'.format(labels[c[0]], labels[c[1]]))

        if i == nplots - 2:
            ax.legend(loc='best')

    ax=fig.add_subplot(1, nplots, i+2)
    ax.plot(galpy_points[:,-1], epi_points[:,-1], c='black')
    ax.set_xlabel('{} galpy'.format(labels[-1]))
    ax.set_ylabel('{} epi'.format(labels[-1]))

    fig.suptitle(plt_title)
    plt.savefig('{}_orbits/{}.pdf'.format(space, plt_title))

    return galpy_points, epi_points

if __name__ == '__main__':
    torb.MIKES_IMP = False
    # Bunch of starting points in galpy coordinates [R, vR, vT, Z, vZ, phi]
    # where R and Z are normalised such that 1 = 8000 pc
    # (so R = 8050/8000 means a starting point of X = -50)
    # and vR, vT and vZ are normalised such that 1 = 220 km/s,
    # (so vT 218./220 is V = -2)
    galpy_starts = {
        'y_offset':[1,0,1+1e-4,10./8000,0,1e-3*np.pi],
        'x_offset':[8050./8000.,0,1,10./8000,0,0.],
        'wild':    [8100./8000, 2./220, 218./220, 15./8000, -2./220, np.pi/8],
        'wilder':   [8200./8000, 5./220, 210./220, 45./8000, -12./220, np.pi/8],
        'u_offset':[1,2./220,1,10./8000,0.,0],
    }


    for plt_title, galpy_start in galpy_starts.items():
        galpy_points, epi_points = plot_galpy_orbit(galpy_start, 'epicyclic')
        galpy_points, epi_points = plot_galpy_orbit(galpy_start, 'galpy')
        galpy_points, epi_points = plot_galpy_orbit(galpy_start, 'chron')



