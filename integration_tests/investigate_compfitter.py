print('in investigate')
import numpy as np
from chronostar import tabletool
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from chronostar.component import SphereComponent

labels = 'XYZUVW'
units = 3*['pc'] + 3*['km/s']

def animate(i):
    plt.clf()
    axes = fig.subplots(nrows=2, ncols=2)
    fig.set_tight_layout(True)
    dims = [(0,1), (0,3), (1,4), (2,5)]
    comp = SphereComponent(emcee_pars=best_comp_per_step[i])
    for ax, (dim1, dim2) in zip(axes.flatten(), dims):
        ax.tick_params(direction='in')

        ax.set_xlabel('{} [{}]'.format(labels[dim1], units[dim1]))
        ax.set_ylabel('{} [{}]'.format(labels[dim2], units[dim2]))

        means_all = data_dict['means']  # XYZUVW

        # for i, best_fit_comp in enumerate(best_fit_comps):
        comp.plot(dim1, dim2, comp_now=True, comp_then=True,
                           color='red', ax=ax,
                           comp_orbit=True, orbit_color='red')

        ax.scatter(means_all[:, dim1], means_all[:, dim2], c='k', s=1)


# Load and prepare data
chains = {}
lnprobs = {}
i=0
while True:
    try:
        print('{}, '.format(i), end='')
        chains[i] = np.load('burnin_chain{:02}.npy'.format(i))
        lnprobs[i] = np.load('burnin_lnprob{:02}.npy'.format(i))
        i += 1
    except FileNotFoundError:
        break

# Build one giant burnin chain
burnin_chain = np.concatenate(list(chains.values()), axis=1)
burnin_chain = np.swapaxes(burnin_chain, 0, 1)
burnin_lnprobs = np.concatenate(list(lnprobs.values()), axis=1)
burnin_lnprobs = np.swapaxes(burnin_lnprobs, 0, 1)

plt.clf()
plt.plot(burnin_chain[:,:,-1])
plt.savefig('ages')

best_comp_per_step = np.argmax(burnin_lnprobs, axis=0)

best_comp_per_step = np.array([chain[np.argmax(lnprob)] for (chain, lnprob) in zip(burnin_chain[::100], burnin_lnprobs[::100]) ])

data_dict = tabletool.build_data_dict_from_table('../../temp_data/3_compfitter_lcc_like.fits')

Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

# fig = plt.figure(figsize=(10,6))
figsize = 10
fig = plt.figure(figsize=(figsize, figsize))
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(figsize, figsize))
fig.set_tight_layout(True)


nframes = len(best_comp_per_step)
# nframes = 10
ani = matplotlib.animation.FuncAnimation(fig, animate, frames=nframes, repeat=True)# , fargs=(axes,))

ani.save('burnin_comps.mp4')
