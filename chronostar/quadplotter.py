from __future__ import division, print_function

import pdb

import numpy as np
import pickle

import groupfitter as gf
import analyser as an
import error_ellipse as ee
from chronostar import utils
import matplotlib.pyplot as plt
from investigator import SynthFit

def calc_area(ys):
    """Calculate the area under the curve provided

    Parameters
    ----------
    ys : [n] array

    Output
    ------
    Approximate area under curve
    """
    assert(np.min(ys) >= 0)
    nrecs = len(ys) - 1

    total_area = 0
    for i in range(nrecs):
        total_area += 0.5 * (ys[i] + ys[i+1]) / nrecs

    return total_area

def plot_sub_traceback(xyzuvw, xyzuvw_cov, times, dim1, dim2, ax):
    """Plot the 2D traceback of stars with their error ellipses

    Parameters
    ----------
    xyzuvw : [nstars, ntimes, 6] array
        exact orbital position in XYZ (pc) and UVW (km/s)
    xyzuvw_cov : [nstars, ntimes, 6, 6] array
        covariance matrix for each star at each time step
    times : [ntimes] array
        times corresponding to each traceback step
    dim1, dim2 : ints
        Denotes which phase dimensions will be plotted [0,1,2,3,4,5] -> [XYZUVW]
    ax : pyplot axes object
        the axes to be plotted in

    Notes
    -----
    TODO: Maybe adjust so it takes in "fixed times" as its time parameter
        This way I can map fits from the fixed time fits onto the traceback plot
    """
    labels = ['X [pc]', 'Y [pc]', 'Z [pc]', 'U [km/s]', 'V [km/s]', 'W [km/s]']
    # X and U are negative as per convention
    axis_ranges = [-300, 300, 300, -50, 50, 50]

    nstars = len(xyzuvw)

    cov_ix1 = [[dim1, dim2], [dim1, dim2]]
    cov_ix2 = [[dim1, dim1], [dim2, dim2]]
    for i in range(nstars):
        ax.plot(xyzuvw[i, :, dim1], xyzuvw[i, :, dim2], 'b-')

        cov_start = xyzuvw_cov[i,  0, cov_ix1, cov_ix2]
        cov_end = xyzuvw_cov[i, -1, cov_ix1, cov_ix2]
        ee.plot_cov_ellipse(
            cov_start,
            [xyzuvw[i, 0, dim1], xyzuvw[i, 0, dim2]],
            color='g', alpha=0.1, ax=ax
        )
        ee.plot_cov_ellipse(
            cov_end,
            [xyzuvw[i, -1, dim1], xyzuvw[i, -1, dim2]],
            color='r', alpha=0.1, ax=ax
        )

    ax.set(aspect='equal')
    ax.set_xlabel(labels[dim1])
    ax.set_ylabel(labels[dim2])
    ax.set_xlim(-axis_ranges[dim1], axis_ranges[dim1]) # note inverse X axis
    ax.set_ylim(-axis_ranges[dim2], axis_ranges[dim2])

def plot_sub_spreads(fixed_times, bayes_spreads, naive_spreads, init_conditions, ax):
    """Plot the bayesian and naive fits to the spread of stars

    Parameters
    ----------
    fixed_times : [ntimes] array
        Discrete time steps corresponding to the exact traceback steps
    bayes_spreads : [ntimes] array
        The idealised spherical radius of the position component of the
        bayesian fit
    naive_spreads : [ntimes] array
        The idealised spherical radius of the gaussian fit to the "exact" orbits
    init_conditions : [14] array
        Set of parameters used to construct synthetic data initially
    ax : pyplot axes object
        The axes on which to be plotted
    """
    ax.plot(fixed_times, naive_spreads, label="Naive fit")
    ax.plot(fixed_times, bayes_spreads, label="Bayes fit")
    ax.set_xlim(fixed_times[0], fixed_times[-1])
    ax.set_ylim(
        bottom=0.0, top=max(np.max(naive_spreads), np.max(bayes_spreads))
    )
    if init_conditions is not None:
        init_age = init_conditions[13]
        ax.axvline(
            init_age, ax.get_ylim()[0], ax.get_ylim()[1], color='r', ls='--'
        )
    ax.legend(loc=1)
    ax.set_xlabel("Traceback Time [Myr]")
    ax.set_ylabel("Radius of average spread in XYZ [pc]")

def plot_age_hist(chain, ax, init_conditions=None):
    """Plot a histogram of the ages

    A histogram marginalises over all the parameters yielding a
    bayesian statistical description of the best fitting age

    Parameters
    -----------
    chain : [nsteps, nwalkers, npars] array
        the chain of samples
    ax : pyplot axes object
        the axes on which to plot
    init_conditions : [15] array {None}
        group parameters that initialised the data - external encoding
    """
    print("In plot_age_hist")
    ax.hist(chain[:,:,-1].flatten(), bins=20)

    ax.set_xlabel("Ages [Myr]")
    ax.set_ylabel("Number of samples")

    if init_conditions is not None:
        init_age = init_conditions[13]
        ax.axvline(
            init_age, ax.get_ylim()[0], ax.get_ylim()[1], color='r', ls='--'
        )

    print("Done most of plot_age_hist")


def plot_age_radius_hist(chain, ax, init_conditions=None):
    """Plot a 2D histogram of effective position radius and age

    Parameters
    ----------
    chain : [nsteps, nwalkers, npars] array
        the chain of samples
    ax : pypot axes object
        the axes on which to plot
    init_conditions : [15] array {None}
        group parameters that initialised the data - external encoding
    """
    print("In plot_age_radius_hist")
    npars = chain.shape[-1]
    nsamples = chain.shape[0] * chain.shape[1]
    flatchain = chain.reshape((nsamples, npars))
    radii = np.zeros(nsamples)

    # OMG SO FKN SLOW, maybe swig up a determinant calculator
    for i, sample in enumerate(flatchain):
        if i % 1000 == 0:
            print("{} of {} done".format(i, len(flatchain)))
        radii[i] = utils.approx_spread_from_sample(sample)
    #data = zip(radii, flatchain[:,-1])
    #pdb.set_trace()
    ax.hist2d(flatchain[:,-1], radii, bins=30)
    ax.set_xlabel("Traceback age [Myr]")
    ax.set_ylabel("Radius of spread in XYZ [pc]")
    #pdb.set_trace()

    if init_conditions is not None:
        init_age = init_conditions[13]
        ax.axvline(
            #init_age, ax.get_ylim()[0], ax.get_ylim()[1], color='r', ls='--'
            init_age, 0, 10, color='r', ls='--'
        )


def plot_sub_age_pdf(times, time_probs, init_conditions, ax):
    """
    Normalise and the plot the age pdf

    Parameters
    ----------
    times : [ntimes] array
        Discrete time steps corresponding to the exact traceback steps
    time_probs : [ntimes] array
        scaled average likelihoods of the samples for each fixed time fit
    init_conditions : [14] array
        Set of parameters used to construct synthetic data initially
    ax : pyplot axes object
        axes on which to be plotted
    """
    normalised_time_probs = time_probs / calc_area(time_probs)
    ax.plot(times, normalised_time_probs)
    ax.set_xlim(times[0],times[-1])
    ax.set_ylim(bottom=0.0, top=1.1*max(normalised_time_probs))
    if init_conditions is not None:
        init_age = init_conditions[13]
        ax.axvline(
            init_age, ax.get_ylim()[0], ax.get_ylim()[1], color='r', ls='--'
        )
    ax.set_xlabel("Traceback Time [Myr]")
    ax.set_ylabel("Age probability")

def plot_quadplots(infile, fixed_times,
                   bayes_spreads=None, naive_spreads=None, #time_probs=None,
                   init_conditions=None, plot_it=False, save_dir=''):
    """
    Generates many quad plots in the provided directory

    Parameters
    ----------
    infile : str
        The traceback file of a set of stars
    fixed_times : [nfixed_ages] array
        The times at which fixed age fits were performed. Not necessarily
        the same times as the traceback timesteps.
    bayes_spreads : [nfixed_ages] array
        The radius of an idealised sphere corresponding to the volume
        in XYZ space of the bayesian fit at each timestep
    naive_spreads : [nfixed_ages] array
        The radius of an idealised sphere corresponding to the volume
        in XYZ space of the each star's mean position (fitting a cov matrix
        to the stellar postional distribution)
    init_conditions : [15] array
        Group pars (external incoding) which initialised the synthesis
    plot_it : boolean {False}
        Generates temp plots of fitting process
    save_dir : str
        directory to save plots etc

    Returns
    -------
    -

    Notes
    -----
    TODO: take in a dir argument to investigate that directory

    TODO: incorporate the bayesian fit to the traceback
    """

    #pdb.set_trace()
    stars, trace_times, xyzuvw, xyzuvw_cov = pickle.load(open(infile, 'r'))
    nstars = len(xyzuvw)

    best_fit_free, chain_free, lnprob_free = \
        gf.fit_group(
            infile, burnin_steps=300, sampling_steps=1000, plot_it=plot_it
        )

    #pdb.set_trace()
    # Gather data of spread fits
    if naive_spreads is None:
        naive_spreads = an.get_naive_spreads(xyzuvw)
    if bayes_spreads is None: # or time_probs is None:
        bayes_spreads, _ = gf.get_bayes_spreads(infile, plot_it=plot_it)
    assert(len(fixed_times) == len(naive_spreads))
    assert(len(fixed_times) == len(bayes_spreads))

    # Plot spread fits
    plt.clf()
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    f.set_size_inches(20, 20)

    plot_sub_traceback(xyzuvw, xyzuvw_cov, trace_times, 0, 1, ax1)
    plot_sub_spreads(fixed_times, bayes_spreads, naive_spreads, init_conditions, ax2)
    plot_age_hist(chain_free, ax3, init_conditions=init_conditions)
    plot_age_radius_hist(chain_free, ax4, init_conditions=init_conditions)
    #plot_sub_traceback(xyzuvw, xyzuvw_cov, trace_times, 3, 4, ax3)
    #plot_sub_age_pdf(trace_times, time_probs, init_conditions, ax4)
    f.savefig(save_dir+'temp_plot.png')

def quadplot_synth_res(synthfit, save_dir='', maxtime=None):
    """Generate a quadplot from a synthfit result

    Parameters
    ----------
    synthfit: investigator.SynthFit instance
        encapsulates the synthesis, fitting and analysis of a group

    maxtime: float
        determines the highest traceback age to be calculated
    """
    if maxtime is None:
        maxtime = synthfit.fixed_ages[-1]
    plot_quadplots(
        synthfit.gaia_tb_file,
        synthfit.fixed_ages,
        bayes_spreads=synthfit.bayes_spreads,
        naive_spreads=synthfit.naive_spreads,
        init_conditions=synthfit.init_group_pars_ex,
        save_dir=save_dir,
    )


