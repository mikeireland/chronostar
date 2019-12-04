from __future__ import division, print_function

import logging
import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits

import chronostar.component
import chronostar.retired.fitplotter
import chronostar.traceorbit as torb
import chronostar.transform as tf
import chronostar.retired2.datatool as dt

COLORS = ['xkcd:neon purple','xkcd:orange', 'xkcd:cyan',
          'xkcd:sun yellow', 'xkcd:shit', 'xkcd:bright pink']*12
#COLORS = ['xkcd:cyan'] * 60
color_codes = ['0xF97306','0x00FFFF', '0xFFDF22',   '0x7F5F00',
                    '0xFE01B1',    '0xBC13FE',]
color_names = ['orange',  'cyan',     'sun yellow', 'shit',
                    'bright pink', 'neon purple',]

HATCHES = ['|', '/',  '+', '.', '*'] * 10


def plot_age_hist(ages, ax, init_conditions=None):
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
    logging.info("In plot_age_hist")
    ngroups = ages.shape[0]
    max_age = np.max(ages)
    min_age = np.min(ages)
    age_range = max_age - min_age
    # have one bin per 10kyr. Lets me plot many (or one) age histograms
    # on same axes, neatly
    nbins = max(int(age_range/0.5), 20)
    bins = np.linspace(min_age,max_age,nbins)

    for i in range(ngroups):
        ax.hist(ages[i], bins=bins, label="Group {}".format(i),
                color=COLORS[i], histtype='step')
    ax.set_xlabel("Ages [Myr]")
    ax.set_ylabel("Number of samples")
    ax.legend()

#    if init_conditions is not None:
#        init_age = init_conditions[13]
#        ax.axvline(
#            init_age, ax.get_ylim()[0], ax.get_ylim()[1], color='r',
#            ls='--'
#        )

def plot_then(star_pars, means, covs, ngroups, iter_count, ax, dim1, dim2):
    try:
        means['origin_then']
        origins_inc = True
    except KeyError:
        origins_inc = False

    dim_label = 'XYZUVW'
    units = ['pc']*3 + ['km/s']*3

    for i in range(ngroups):
        if origins_inc:
            #import pdb; pdb.set_trace()
            try:
                chronostar.retired.fitplotter.plotCovEllipse(
                    covs['origin_then'][i][np.ix_([dim1,dim2],[dim1,dim2])],
                    means['origin_then'][i][np.ix_([dim1,dim2])],
                    with_line=True,
                    ax=ax, color="xkcd:grey", alpha=0.3, ls='--',
                    #hatch='|',
                )
                ax.plot(means['origin_then'][i][dim1],
                        means['origin_then'][i][dim2],
                        color="xkcd:grey", marker='+', alpha=1)
            except:
                logging.info("No origins provided")
        # I plot a marker in the middle for scenarios where the volume
        # collapses to a point
        try:
            ax.plot(means['fitted_then'][i][dim1],
                    means['fitted_then'][i][dim2],
                    color=COLORS[i], marker='x', alpha=1)
            chronostar.retired.fitplotter.plotCovEllipse(
                covs['fitted_then'][i][np.ix_([dim1,dim2],[dim1,dim2])],
                means['fitted_then'][i][np.ix_([dim1,dim2])],
                with_line=True,
                ax=ax, color=COLORS[i], alpha=0.3, ls='-.', #hatch='/',
            )
        except IndexError:
            pass

    ax.set_xlabel("{} [{}]".format(dim_label[dim1], units[dim1]))
    ax.set_ylabel("{} [{}]".format(dim_label[dim2], units[dim2]))

    logging.info("Iteration {}: {}{} plot plotted".\
                 format(iter_count, dim_label[dim1], dim_label[dim2]))


def plot_now(star_pars, means, covs, ngroups, iter_count, ax, dim1=0,
             dim2=1, z=None):
    try:
        means['origin_then']
        origins_inc = True
    except KeyError:
        origins_inc = False

    dim_label = 'XYZUVW'
    units = ['pc']*3 + ['km/s']*3
    xyzuvw = star_pars['xyzuvw']
    xyzuvw_cov = star_pars['xyzuvw_cov']
    if z is not None:
        for i in range(ngroups):
            mask = np.where(z[:,i] > 0.5)
            ax.plot(xyzuvw[mask][:, dim1], xyzuvw[mask][:, dim2], '.',
                    color=COLORS[i], alpha=0.6)
            for mn, cov in zip(xyzuvw[mask], xyzuvw_cov[mask]):
                try:
                    chronostar.retired.fitplotter.plotCovEllipse(cov[np.ix_([dim1, dim2], [dim1, dim2])],
                                                                 mn[np.ix_([dim1,dim2])],
                                                                 ax=ax, color=COLORS[i], alpha=0.1)
                except IndexError:
                    pass
    else:
        ax.plot(xyzuvw[:, dim1], xyzuvw[:, dim2], 'b.')
        for mn, cov in zip(xyzuvw, xyzuvw_cov):
            chronostar.retired.fitplotter.plotCovEllipse(cov[np.ix_([dim1, dim2], [dim1, dim2])],
                                                         mn[np.ix_([dim1,dim2])],
                                                         ax=ax, color='b', alpha=0.1)
    for i in range(ngroups):
        try:
            #pdb.set_trace()
            chronostar.retired.fitplotter.plotCovEllipse(
                covs['fitted_now'][i][np.ix_([dim1,dim2],[dim1,dim2])],
                means['fitted_now'][i][np.ix_([dim1,dim2])],
                with_line=True,
                ax=ax, color=COLORS[i], ec=COLORS[i],
                fill=False, alpha=0.3, hatch=HATCHES[i], ls='-.',
            )
        except IndexError:
            #pdb.set_trace()
            pass

    ax.set_xlabel("{} [{}]".format(dim_label[dim1], units[dim1]))
    #ax.set_ylabel("{} [{}]".format(dim_label[dim2], units[dim2]))

    logging.info("Iteration {}: {}{} plot plotted".\
                 format(iter_count, dim_label[dim1], dim_label[dim2]))



def plot_fit(star_pars, means, covs, ngroups, iter_count, ax, dim1=0,
             dim2=1):
    try:
        means['origin_then']
        origins_inc = True
    except KeyError:
        origins_inc = False

    dim_label = 'XYZUVW'
    units = ['pc']*3 + ['km/s']*3
    xyzuvw = star_pars['xyzuvw']
    xyzuvw_cov = star_pars['xyzuvw_cov']
    ax.plot(xyzuvw[:, dim1], xyzuvw[:, dim2], 'b.')
    for mn, cov in zip(xyzuvw, xyzuvw_cov):
        chronostar.retired.fitplotter.plotCovEllipse(cov[np.ix_([dim1, dim2], [dim1, dim2])],
                                                     mn[np.ix_([dim1,dim2])],
                                                     ax=ax, color='b', alpha=0.1)
    for i in range(ngroups):
        if origins_inc:
            #import pdb; pdb.set_trace()
            chronostar.retired.fitplotter.plotCovEllipse(
                covs['origin_then'][i][np.ix_([dim1,dim2],[dim1,dim2])],
                means['origin_then'][i][np.ix_([dim1,dim2])],
                with_line=True,
                ax=ax, color="xkcd:grey", alpha=0.3, ls='--',
                #hatch='|',
            )
            ax.plot(means['origin_then'][i][dim1],
                    means['origin_then'][i][dim2],
                    color="xkcd:grey", marker='x', alpha=1)
        # I plot a marker in the middle for scenarios where the volume
        # collapses to a point
        ax.plot(means['fitted_then'][i][dim1],
                means['fitted_then'][i][dim2],
                color=COLORS[i], marker='x', alpha=1)
        chronostar.retired.fitplotter.plotCovEllipse(
            covs['fitted_then'][i][np.ix_([dim1,dim2],[dim1,dim2])],
            means['fitted_then'][i][np.ix_([dim1,dim2])],
            with_line=True,
            ax=ax, color=COLORS[i], alpha=0.3, ls='-.', #hatch='/',
        )
        chronostar.retired.fitplotter.plotCovEllipse(
            covs['fitted_now'][i][np.ix_([dim1,dim2],[dim1,dim2])],
            means['fitted_now'][i][np.ix_([dim1,dim2])],
            with_line=True,
            ax=ax, color=COLORS[i], ec=COLORS[i],
            fill=False, alpha=0.3, hatch=HATCHES[i], ls='-.',
        )

    ax.set_xlabel("{} [{}]".format(dim_label[dim1], units[dim1]))
    ax.set_ylabel("{} [{}]".format(dim_label[dim2], units[dim2]))

    logging.info("Iteration {}: {}{} plot plotted".\
                 format(iter_count, dim_label[dim1], dim_label[dim2]))


def get_age_samples(ngroups, final_chain):
    """
    Uses the number of groups to explore (and hopefully find) final chains

    Returns None if any of the chains aren't there.

    Returns
    -------
    age_samples : [ngroups, nsteps*nwalkers] array
    """
    logging.info("In get_age_samples")
    age_samples = []
    for group_ix in range(ngroups):
        # extract final burnin chain through trial and error XD
        burnin_cnt = 0
        try:
            # tfgroupfitter stores in this format
            age_samples.append(final_chain[group_ix][:,:,-1].flatten())
        except IOError:
            return None
    return np.array(age_samples)

def plot_hexplot(star_pars, means, covs, chain, iter_count, prec=None,
                  save_dir='', file_stem='', title=''):
    """
    Generates hex plot in the provided directory

    Paramters
    ---------
    star_pars : dict
        'xyzuvw'
        'xyzuvw_cov'
        'times'
        'something else...'
    means : dict
        'fitted_now'
        'fitted_then'
        'origin_now'  - optional (currently not in use)
        'origin_then' - optional
    covs : dict
        'fitted_now'
        'fitted_then'
        'origin_now'  - optional (currently not in use)
        'origin_then' - optional
    chain:
    iter_count : integer
    """
    logging.info("In plot_hexplot, iter {}".format(iter_count))
    ngroups = covs['fitted_then'].shape[0]

    # INITIALISE PLOT
    plt.clf()
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    f.set_size_inches(30, 20)
    f.suptitle(title)

    # PLOT THE OVAL PLOTS
    plot_fit(star_pars, means, covs, ngroups, iter_count, ax1, 0, 1)
    plot_fit(star_pars, means, covs, ngroups, iter_count, ax2, 3, 4)
    plot_fit(star_pars, means, covs, ngroups, iter_count, ax4, 0, 3)
    plot_fit(star_pars, means, covs, ngroups, iter_count, ax5, 1, 4)
    plot_fit(star_pars, means, covs, ngroups, iter_count, ax6, 2, 5)

    # PLOT THE HISTOGRAMS
    age_samples = get_age_samples(ngroups, chain)
    if age_samples is not None:
        plot_age_hist(age_samples, ax3)

    f.savefig(save_dir+"hexplot_"+file_stem+"{:02}.pdf".format(iter_count),
              bbox_inches='tight', format='pdf')
    f.clear()

def dataGatherer(res_dir='', save_dir='', data_dir='', xyzuvw_file='',
                 title='', file_stem=''):
    """
    Provided with a results directory, tries to find all she needs, then
    plots

    Parameters
    ----------
    """
    covs = {}
    means = {}
    star_pars = {}

    chain_file = res_dir + "final_chain.npy"
    lnprob_file = res_dir + "final_lnprob.npy"
    origin_file = res_dir + "origins.npy"
    if not xyzuvw_file:
        logging.info("No xyzuvw filename provided. Must be synth fit yes?")
        xyzuvw_file = res_dir + "xyzuvw_now.fits"


    chain = np.load(chain_file)
    chain = np.array([chain])
    lnprob = np.load(lnprob_file)
    best_sample = dt.getBestSample(chain, lnprob)
    best_group = chronostar.component.Component(best_sample,
                                                form=len(best_sample) == 9,
                                                internal=True)

    star_pars['xyzuvw'] = fits.getdata(xyzuvw_file, 1)
    star_pars['xyzuvw_cov'] = fits.getdata(xyzuvw_file, 2)

    try:
        origins = np.load(origin_file).item()
        means['origin_then'] = np.array([origins.mean])
        covs['origin_then'] = np.array([origins.generateCovMatrix()])
    except IOError:
        logging.info("No origins file: {}".format(origin_file))

    means['fitted_then'] = np.array([best_group.mean])
    means['fitted_now']  =\
        np.array([torb.trace_cartesian_orbit(best_group.mean, best_group.age)])

    covs['fitted_then'] = np.array([best_group.generateCovMatrix()])
    covs['fitted_now']  =\
        np.array([
            tf.transform_covmatrix(covs['fitted_then'][0], torb.trace_cartesian_orbit,
                                   means['fitted_then'][0],
                                   args=(best_group.age,True)
                                   )
        ])

    plot_hexplot(star_pars, means, covs, chain, iter_count=0,
                 save_dir=save_dir, file_stem=file_stem, title=title)
    plotXYandZW(star_pars, means, covs, chain, iter_count=0,
                 save_dir=save_dir, file_stem=file_stem, title=title)

def plotNewHex(star_pars, means, covs, chain, iter_count, prec=None,
               save_dir='', file_stem='', title='', z=None):
    logging.info("In plotXY., iter {}".format(iter_count))
    ngroups = covs['fitted_then'].shape[0]

    # INITIALISE PLOT
    plt.clf()
    #f, ax1 = plt.subplots(1, 1)
    f, axs = plt.subplots(3, 2, sharey='row', sharex='row')
    f.set_size_inches(10, 15)
    # f.suptitle(title)
    f.set_tight_layout(tight=True)
    plot_then(star_pars, means, covs, ngroups, iter_count, axs[0,0], 0, 3)
    plot_then(star_pars, means, covs, ngroups, iter_count, axs[1,0], 1, 4)
    plot_then(star_pars, means, covs, ngroups, iter_count, axs[2,0], 2, 5)
    plot_now(star_pars, means, covs, ngroups, iter_count, axs[0,1], 0, 3, z)
    plot_now(star_pars, means, covs, ngroups, iter_count, axs[1,1], 1, 4, z)
    plot_now(star_pars, means, covs, ngroups, iter_count, axs[2,1], 2, 5, z)

    f.savefig(
        save_dir + "newsixplot_" + file_stem + "{:02}.pdf".format(iter_count),
        bbox_inches='tight', format='pdf')
    f.clear()

def plotXYandZW(star_pars, means, covs, chain, iter_count, prec=None,
                  save_dir='', file_stem='', title=''):
    logging.info("In plotXY., iter {}".format(iter_count))
    ngroups = covs['fitted_then'].shape[0]

    # INITIALISE PLOT
    plt.clf()
    f, ax1 = plt.subplots(1, 1)
    f, ((ax1), (ax2)) = plt.subplots(2, 1)
    f.set_size_inches(5, 10)
    #f.suptitle(title)
    f.set_tight_layout(tight=True)

    # PLOT THE OVAL PLOTS
    plot_fit(star_pars, means, covs, ngroups, iter_count, ax1, 0, 1)
    plot_fit(star_pars, means, covs, ngroups, iter_count, ax2, 2, 5)
#    plot_fit(star_pars, means, covs, ngroups, iter_count, ax4, 0, 3)
#    plot_fit(star_pars, means, covs, ngroups, iter_count, ax5, 1, 4)
#    plot_fit(star_pars, means, covs, ngroups, iter_count, ax6, 2, 5)

#    # PLOT THE HISTOGRAMS
#    age_samples = get_age_samples(ngroups, chain)
#    if age_samples is not None:
#        plot_age_hist(age_samples, ax3)

    f.savefig(
        save_dir + "duoplot_" + file_stem + "{:02}.pdf".format(iter_count),
        bbox_inches='tight', format='pdf')
    f.clear()

def plotNewQuad(star_pars, means, covs, chain, iter_count, prec=None,
               save_dir='', file_stem='', title='', z=None, top_dims=(0,1),
               bot_dims=(3,4)):
    logging.info("In plotXY., iter {}".format(iter_count))
    try:
        ngroups = z.shape[1]
    except:
        ngroups = covs['fitted_then'].shape[0]

    # INITIALISE PLOT
    plt.clf()
    #f, ax1 = plt.subplots(1, 1)
    f, axs = plt.subplots(2, 2, sharey='row', sharex='row')
    f.set_size_inches(10, 10)
    # f.suptitle(title)
    f.set_tight_layout(tight=True)
    plot_then(star_pars, means, covs, ngroups, iter_count, axs[0,0],
              top_dims[0], top_dims[1])
    plot_then(star_pars, means, covs, ngroups, iter_count, axs[1,0],
              bot_dims[0], bot_dims[1])
    plot_now(star_pars, means, covs, ngroups, iter_count, axs[0,1],
              top_dims[0], top_dims[1], z)
    plot_now(star_pars, means, covs, ngroups, iter_count, axs[1,1],
              bot_dims[0], bot_dims[1], z)
    f.suptitle(title)

    if type(iter_count) == int:
        iter_stamp = "{:02}".format(iter_count)
    else:
        iter_stamp = str(iter_count)
    f.savefig(
        save_dir + "newquadplot_" + file_stem + "{}.pdf".format(iter_stamp),
        bbox_inches='tight', format='pdf'
    )
    f.clear()

def dataGathererEM(ngroups, iter_count, res_dir='', save_dir='', data_dir='',
                   xyzuvw_file='', title='', file_stem='', groups_file=''):
    """
    Provided with a results directory, tries to find all she needs, then
    plots

    Parameters
    ----------
    ngroups: int
        number of groups
    """
    covs = {}
    means = {}
    star_pars = {}

    if not groups_file:
        groups_file = "best_group_fit.npy"

    chain_file = "final_chain.npy"
   # lnprob_file =  "final_lnprob.npy"
   # origin_file = res_dir + "origins.npy"
    if not xyzuvw_file:
        logging.info("No xyzuvw filename provided. Must be synth fit yes?")
        xyzuvw_file = res_dir + "../xyzuvw_now.fits"
    try:
        print("Trying to load from {}".format(xyzuvw_file))
        star_pars['xyzuvw'] = fits.getdata(xyzuvw_file, 1)
        star_pars['xyzuvw_cov'] = fits.getdata(xyzuvw_file, 2)
        print("Successfully loaded from xyzuvw_file")
    except:
        import chronostar.retired.groupfitter as rgf
        old_star_pars = rgf.read_stars(res_dir + "../perf_tb_file.pkl")
        star_pars = {'xyzuvw':old_star_pars['xyzuvw'][:,0],
                     'xyzuvw_cov':old_star_pars['xyzuvw_cov'][:,0]}

    try:
        origins = np.load(res_dir + '../origins.npy')
    except:
        origins = None
    z = np.load(res_dir + '../memberships.npy')

    fitted_then_mns = []
    fitted_then_covs = []
    fitted_now_mns = []
    fitted_now_covs = []
    origin_then_mns = []
    origin_then_covs = []
    all_chains = []
    for group_ix in range(ngroups):
        gdir = res_dir + "group{}/".format(group_ix)

        chain = np.load(gdir + chain_file)
        all_chains.append(chain)

        best_group = np.load(gdir + groups_file).item()
        fitted_then_mns.append(best_group.mean)
        fitted_then_covs.append(best_group.generateCovMatrix())

        fitted_now_mn = torb.trace_cartesian_orbit(fitted_then_mns[group_ix],
                                                   best_group.age,
                                                   single_age=True)
        fitted_now_cov =\
            tf.transform_covmatrix(fitted_then_covs[group_ix],
                                   torb.trace_cartesian_orbit,
                                   fitted_then_mns[group_ix],
                                   args=(best_group.age,))
        fitted_now_mns.append(fitted_now_mn)
        fitted_now_covs.append(fitted_now_cov)

        if origins is not None:
            origin_then_mns.append(origins[group_ix].mean)
            origin_then_covs.append(origins[group_ix].generateCovMatrix())

    means = {
       'origin_then':origin_then_mns,
       'fitted_then':fitted_then_mns,
       'fitted_now':fitted_now_mns,
     }
    covs = {
        'origin_then':np.array(origin_then_covs),
        'fitted_then':np.array(fitted_then_covs),
        'fitted_now':np.array(fitted_now_covs),
    }

    plotNewQuad(star_pars, means, covs, all_chains, iter_count=iter_count,
               save_dir=save_dir, file_stem=file_stem, title=title,
               z=z)
    plotNewHex(star_pars, means, covs, all_chains, iter_count=iter_count,
               save_dir=save_dir, file_stem=file_stem, title=title,
               z=z)

