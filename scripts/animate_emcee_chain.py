"""
For efficiency, if a video already exists, this script will
skip over it. So if you wish to update a video, delete it 
first, and then run this script.

usage:
python animate_results_save.py path_to_your_data.fits [N_MAX_COMPS] [dim1] [dim2]
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

sys.path.insert(0, '/Users/tcrundall/code/neelesh_chronostar/')

from chronostar.component import EllipComponent
from chronostar import tabletool

print(EllipComponent.PARAMETER_FORMAT)

Component = EllipComponent

true_comp_file = 'true_and_best_comp.npy'
datafile = 'synth_data.fits'

# Load and join emcee chains:
i = 0
all_chains = []
while True:
    try:
        all_chains.append(np.load('burnin_chain%02i.npy'%i))
        i += 1
    except:
        break
all_chains = np.concatenate(all_chains, axis=1)
# all_chains ahs shape [nwalkers, nsteps, npars]

i=0
all_lnprobs = []
while True:
    try:
        all_lnprobs.append(np.load('burnin_lnprob%02i.npy'%i))
        i += 1
    except:
        break
all_lnprobs = np.concatenate(all_lnprobs,axis=1)
# all_lnprobs has shape [nwalkers, nsteps]

TRUE_COMP = Component.load_raw_components(true_comp_file)[0]

emcee_leap = 100
N_MAX_ITERS = int(all_lnprobs.shape[1]/emcee_leap)
# N_MAX_ITERS = 20

LABELS='XYZUVW'
UNITS=3*['pc'] + 3*['km/s']
dims = [(0,1), (3,4), (0,4), (1,3)] #!!! Change these if you wnat different dims


def animate(i):
    """
    This is how matplotlib aniamtes stuff.
    You write a function that is responsible for generating the 
    `i`th plot.

    So in this one, we extract the `i`th subchain of length `emcee_leap`,
    find the best component sample, then plot it against the stars and
    the true component
    """
    print('animating step %i from %i steps'%(i,N_MAX_ITERS))

    # extract emcee subchain
    subchain  = all_chains[:, emcee_leap*i:emcee_leap*(i+1)]
    sublnprob = all_lnprobs[:, emcee_leap*i:emcee_leap*(i+1)]

    # get best component index
    # (only looks complicated because we need to index a 2d array and 
    #  np.argmax only returns a 1d value)
    best_comp_2dix = np.unravel_index(sublnprob.argmax(), sublnprob.shape)

    # get best copmonent pars and build component
    best_comp_pars = subchain[best_comp_2dix]
    best_comp = Component(emcee_pars=best_comp_pars)

    # Put in some annotation (doesn't work for some reason....)
    title = 'Fitted age {:5.1f} Myr'.format(best_comp.get_age())
    first_ax = axes[0,1]
    first_ax.text(0.95, 0.9, title,
                  horizontalalignment='right',
                  transform=first_ax.transAxes)

    for ax, (dim1, dim2) in zip(axes.flatten(), dims):
        ax.tick_params(direction='in')
        print(dim1, dim2)
        ax.clear()

        ax.set_xlabel('{} [{}]'.format(LABELS[dim1], UNITS[dim1]))
        ax.set_ylabel('{} [{}]'.format(LABELS[dim2], UNITS[dim2]))

        ax.scatter(data['means'][:,dim1], data['means'][:,dim2])

        TRUE_COMP.plot(ax=ax, dim1=dim1, dim2=dim2, comp_now=True, comp_then=True, color='blue')
        best_comp.plot(ax=ax, dim1=dim1, dim2=dim2, comp_now=True, comp_then=True, color='red')


# Fancy pancy animation type stuff
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

data = tabletool.build_data_dict_from_table(datafile)

figsize = 10
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(figsize,figsize))
fig.set_tight_layout(True)

ani = matplotlib.animation.FuncAnimation(fig, animate, frames=N_MAX_ITERS, repeat=True)

save_filename = 'convergence_movie.mp4'
ani.save(save_filename, writer=writer)


