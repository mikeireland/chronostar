"""
Provides many functions that aid plotting of stellar data sets and their fits

"""

import matplotlib as mpl
from matplotlib.patches import Ellipse

mpl.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import traceorbit as torb
import transform as tf
from chronostar.retired2 import datatool as dt

COLORS = ['xkcd:blue','xkcd:red', 'xkcd:tangerine', 'xkcd:shit', 'xkcd:cyan',
          'xkcd:sun yellow', 'xkcd:neon purple', 'xkcd:bright pink']
# COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
MARKERS = ['^', 's', 'd', '*', 'x']
# MARKERS = ('s', 'p', 'D', 'X', 'H', 'D')
HATCHES = ['|', '/',  '+', '\\', 'o', '*', 'o', '0'] * 10 #'.' just look like stars, so does '*'
HATCHES = ['0'] * 100 # removed hatching for now...
# '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
MARK_SIZE = 140. #80.
BG_MARK_SIZE = 50. #20.
PT_ALPHA = 0.4
COV_ALPHA = 0.2
BG_ALPHA = 0.3
FONTSIZE = 12

MARKER_LABELS = np.array(['True {}'.format(ch) for ch in 'ABCD'])


def set_size(w,h, ax=None):
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)


def add_arrow(line, position=None, indices=None, direction='right',
              size=15, color=None):
    """
    Add an arrow along a plotted line.

    Parameters
    ----------
    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.

    -- credit to some forgotten contributor to stackoverflow --
    https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot-with-matplotlib
    thomas - https://stackoverflow.com/users/5543796/thomas
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if indices is None:
        if position is None:
            position = xdata.mean()
        # find closest index
        indices = [np.argmin(np.absolute(xdata - position))]

    for start_ind in indices:
        end_ind = start_ind + 1 if direction == 'right' else start_ind - 1

        line.axes.annotate('',
            xytext=(xdata[start_ind], ydata[start_ind]),
            xy=(xdata[end_ind], ydata[end_ind]),
            arrowprops=dict(arrowstyle="->", color=color),
            size=size
        )


def plotOrbit(pos_now, dim1, dim2, ax, end_age, ntimes=50, group_ix=None,
              with_arrow=False, annotate=False, color=None):
    """
    For traceback use negative age

    Parameters
    ----------
    pos_now: [6] array, known position of object
    dim1: integer, x-axis dimension
    dim2: integer, y-axis dimension
    ax: axes object, axes on which to plot line
    end_age: non-zero number, time to orbit till.
        Negative value --> traceback
        Positive value --> trace forward
    ntimes: integer {50], number of timesteps to calculate
    group_ix: index of group being plotted (for coloring reasons)
    with_arrow: (bool) {False}, whether to include arrows along orbit
    annotate: (bool) {False}, whether to include text
    """
    if color is None:
        if group_ix is None:
            color = COLORS[0]
        else:
            color = COLORS[group_ix]

    # orb_alpha = 0.1
    gorb = torb.trace_cartesian_orbit(pos_now,
                                      times=np.linspace(0, end_age, ntimes),
                                      single_age=False)
    line_obj = ax.plot(gorb[:, dim1], gorb[:, dim2], ls='-',
                       alpha=0.1,
                       color=color)
    indices = [int(ntimes / 3), int(2 * ntimes / 3)]
    if with_arrow:
        # make sure arrow is always pointing forwards through time
        direction = 'right' if end_age > 0 else 'left'
        add_arrow(line_obj[0], indices=indices, direction=direction,
                  color=color)
    if annotate:
        ax.annotate("Orbital trajectory",
                    (gorb[int(ntimes / 2), dim1],
                     gorb[int(ntimes / 2), dim2]),
                    color=color)


def plotCovEllipse(cov, pos, nstd=2, ax=None, with_line=False, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    if ax is None:
        ax = plt.gca()

    # largest eigenvalue is first
    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    if 'alpha' not in kwargs.keys():
        ellip.set_alpha(0.3)
    if 'color' not in kwargs.keys():# and 'c' not in kwargs.keys():
        ellip.set_facecolor('red')

    ax.add_patch(ellip)

    # THEN just f***ing plot an invisible line across the ellipse.
    if with_line:
        # brute forcing axes limits so they contain ellipse patch
        # maybe a cleaner way of doing this, but I couldn't work it out
        x_extent = 0.5*(abs(width*np.cos(np.radians(theta))) +
                        abs(height*np.sin(np.radians(theta))))
        y_extent = 0.5*(abs(width*np.sin(np.radians(theta))) +
                        abs(height*np.cos(np.radians(theta))))

        lx = pos[0] - x_extent
        ux = pos[0] + x_extent
        ly = pos[1] - y_extent
        uy = pos[1] + y_extent
        ax.plot((lx, ux), (ly, uy), alpha=0.)

    return ellip


def plotPane(dim1=0, dim2=1, ax=None, groups=(), star_pars=None,
             origin_star_pars=None,
             star_orbits=False, origins=None,
             group_then=False, group_now=False, group_orbit=False,
             annotate=False, membership=None, true_memb=None,
             savefile='', with_bg=False, markers=None, group_bg=False,
             marker_labels=None, color_labels=None,
             marker_style=None,
             marker_legend=None, color_legend=None,
             star_pars_label=None, origin_star_pars_label=None,
             range_1=None, range_2=None, isotropic=False,
             ordering=None, no_bg_covs=False):
    """
    Plots a single pane capturing kinematic info in any desired 2D plane

    Uses global constants COLORS and HATCHES to inform consistent colour
    scheme.
    Can use this to plot different panes of one whole figure

    Parameters
    ----------
    dim1: x-axis, can either be integer 0-5 (inclusive) or a letter form
          'xyzuvw' (either case)
    dim2: y-axis, same conditions as dim1
    ax:   the axes object on which to plot (defaults to pyplots currnet axes)
    groups: a list of (or just one) synthesiser.Group objects, corresponding
            to the fit of the origin(s)
    star_pars:  dict object with keys 'xyzuvw' ([nstars,6] array of current
                star means) and 'xyzuvw_cov' ([nstars,6,6] array of current
                star covariance matrices)
    star_orbits: (bool) plot the calculated stellar traceback orbits of
                        central estimate of measurements
    group_then: (bool) plot the group's origin
    group_now:  (bool) plot the group's current day distribution
    group_orbit: (bool) plot the trajectory of the group's mean
    annotate: (bool) add text describing the figure's contents
    with_bg: (bool) treat the last column in Z as members of background, and
            color accordingly
    no_bg_covs: (bool) ignore covariance matrices of stars fitted to background

    Returns
    -------
    (nothing returned)
    """
    labels = 'XYZUVW'
    units = 3 * ['pc'] + 3 * ['km/s']

    if savefile:
        plt.clf()

    # Tidying up inputs
    if ax is None:
        ax = plt.gca()
    if type(dim1) is not int:
        dim1 = labels.index(dim1.upper())
    if type(dim2) is not int:
        dim2 = labels.index(dim2.upper())
    if type(star_pars) is str:
        star_pars = dt.loadXYZUVW(star_pars)
    if type(membership) is str:
        membership = np.load(membership)
    if type(groups) is str:
        groups = dt.loadGroups(groups)
    if marker_style is None:
        marker_style = MARKERS[:]
    # if type(origin_star_pars) is str:
    #     origin_star_pars = dt.loadXYZUVW(origin_star_pars)

    legend_pts = []
    legend_labels = []

    # ensure groups is iterable
    try:
        len(groups)
    except:
        groups = [groups]
    ngroups = len(groups)
    if ordering is None:
        ordering = range(len(marker_style))

    # plot stellar data (positions with errors and optionally traceback
    # orbits back to some ill-defined age
    if star_pars:
        nstars = star_pars['xyzuvw'].shape[0]

        # apply default color and markers, to be overwritten if needed
        pt_colors = np.array(nstars * [COLORS[0]])
        if markers is None:
            markers = np.array(nstars * ['.'])

        # Incorporate fitted membership into colors of the pts
        if membership is not None:
            best_mship = np.argmax(membership[:,:ngroups+with_bg], axis=1)
            pt_colors = np.array(COLORS[:ngroups] + with_bg*['xkcd:grey'])[best_mship]
            # Incoporate "True" membership into pt markers
            if true_memb is not None:
                markers = np.array(MARKERS)[np.argmax(true_memb,
                                                      axis=1)]
                if with_bg:
                    true_bg_mask = np.where(true_memb[:,-1] == 1.)
                    markers[true_bg_mask] = '.'
        all_mark_size = np.array(nstars * [MARK_SIZE])

        # group_bg handles case where background is fitted to by final component
        if with_bg:
            all_mark_size[np.where(np.argmax(membership, axis=1) == ngroups-group_bg)] = BG_MARK_SIZE

        mns = star_pars['xyzuvw']
        try:
            covs = np.copy(star_pars['xyzuvw_cov'])
            # replace background cov matrices with None so as to avoid plotting
            if with_bg and no_bg_covs:
                print("Discarding background cov-mats")
                # import pdb; pdb.set_trace()
                covs[np.where(np.argmax(membership, axis=1) == ngroups-group_bg)] = None
        except KeyError:
            covs = len(mns) * [None]
            star_pars['xyzuvw_cov'] = covs
        st_count = 0
        for star_mn, star_cov, marker, pt_color, m_size in zip(mns, covs, markers, pt_colors,
                                                               all_mark_size):
            pt = ax.scatter(star_mn[dim1], star_mn[dim2], s=m_size, #s=MARK_SIZE,
                            color=pt_color, marker=marker, alpha=PT_ALPHA,
                            linewidth=0.0,
                            )
            # plot uncertainties
            if star_cov is not None:
                plotCovEllipse(star_cov[np.ix_([dim1, dim2], [dim1, dim2])],
                               star_mn[np.ix_([dim1, dim2])],
                               ax=ax, alpha=COV_ALPHA, linewidth='0.1',
                               color=pt_color,
                               )
            # plot traceback orbits for as long as oldest group (if known)
            # else, 30 Myr
            if star_orbits and st_count%3==0:
                try:
                    tb_limit = max([g.age for g in groups])
                except:
                    tb_limit = 30
                plotOrbit(star_mn, dim1, dim2, ax, end_age=-tb_limit,
                          color='xkcd:grey')
            st_count += 1
        if star_pars_label:
            # ax.legend(numpoints=1)
            legend_pts.append(pt)
            legend_labels.append(star_pars_label)

        if origin_star_pars is not None:
            for star_mn, marker, pt_color, m_size in\
                    zip(origin_star_pars['xyzuvw'],
                        # origin_star_pars['xyzuvw_cov'],
                        markers, pt_colors, all_mark_size):
                pt = ax.scatter(star_mn[dim1], star_mn[dim2], s=0.5*m_size,
                           # s=MARK_SIZE,
                           color=pt_color, marker='s', alpha=PT_ALPHA,
                           linewidth=0.0, #label=origin_star_pars_label,
                           )
                # # plot uncertainties
                # if star_cov is not None:
                #     ee.plotCovEllipse(
                #         star_cov[np.ix_([dim1, dim2], [dim1, dim2])],
                #         star_mn[np.ix_([dim1, dim2])],
                #         ax=ax, alpha=0.05, linewidth='0.1',
                #         color=pt_color,
                #         )
            if origin_star_pars_label:
                legend_pts.append(pt)
                legend_labels.append(origin_star_pars_label)


    # plot info for each group (fitted, or true synthetic origin)
    for i, group in enumerate(groups):
        cov_then = group.get_covmatrix()
        mean_then = group.get_mean()
        # plot group initial distribution
        if group_then:
            ax.plot(mean_then[dim1], mean_then[dim2], marker='+', alpha=0.3,
                    color=COLORS[i])
            plotCovEllipse(cov_then[np.ix_([dim1, dim2], [dim1, dim2])],
                           mean_then[np.ix_([dim1,dim2])],
                           with_line=True,
                           ax=ax, alpha=0.3, ls='--',
                           color=COLORS[i])
            if annotate:
                ax.annotate(r'$\mathbf{\mu}_0, \mathbf{\Sigma}_0$',
                            (mean_then[dim1],
                             mean_then[dim2]),
                             color=COLORS[i])

        # plot group current day distribution (should match well with stars)
        if group_now:
            mean_now = torb.trace_cartesian_orbit(mean_then, group.get_age(),
                                                  single_age=True)
            cov_now = tf.transform_covmatrix(cov_then, torb.trace_cartesian_orbit,
                                             mean_then, args=[group.get_age()])
            ax.plot(mean_now[dim1], mean_now[dim2], marker='+', alpha=0.3,
                   color=COLORS[i])
            plotCovEllipse(cov_now[np.ix_([dim1, dim2], [dim1, dim2])],
                           mean_now[np.ix_([dim1,dim2])],
                           # with_line=True,
                           ax=ax, alpha=0.4, ls='-.',
                           ec=COLORS[i], fill=False, hatch=HATCHES[i],
                           color=COLORS[i])
            if annotate:
                ax.annotate(r'$\mathbf{\mu}_c, \mathbf{\Sigma}_c$',
                            (mean_now[dim1],mean_now[dim2]),
                            color=COLORS[i])

        # plot orbit of mean of group
        if group_orbit:
            plotOrbit(mean_now, dim1, dim2, ax, -group.age, group_ix=i,
                      with_arrow=True, annotate=annotate)
    if origins:
        for origin in origins:
            cov_then = origin.generateSphericalCovMatrix()
            mean_then = origin.mean
            # plot origin initial distribution
            ax.plot(mean_then[dim1], mean_then[dim2], marker='+',
                    color='xkcd:grey')
            plotCovEllipse(
                cov_then[np.ix_([dim1, dim2], [dim1, dim2])],
                mean_then[np.ix_([dim1, dim2])],
                with_line=True,
                ax=ax, alpha=0.1, ls='--',
                color='xkcd:grey')

    ax.set_xlabel("{} [{}]".format(labels[dim1], units[dim1]))
    ax.set_ylabel("{} [{}]".format(labels[dim2], units[dim2]))

    # NOT QUITE....
    # if marker_legend is not None and color_legend is not None:
    #     x_loc = np.mean(star_pars['xyzuvw'][:,dim1])
    #     y_loc = np.mean(star_pars['xyzuvw'][:,dim2])
    #     for label in marker_legend.keys():
    #         ax.plot(x_loc, y_loc, color=color_legend[label],
    #                 marker=marker_legend[label], alpha=0, label=label)
    #     ax.legend(loc='best')

    # if star_pars_label is not None:
    #     ax.legend(numpoints=1, loc='best')
        # ax.legend(loc='best')

    # if marker_order is not None:
    #     for label_ix, marker_ix in enumerate(marker_order):
    #         axleg.scatter(0,0,color='black',marker=MARKERS[marker_ix],
    #                       label=MARKER_LABELS[label_ix])
    # #
    # if len(legend_pts) > 0:
    #     ax.legend(legend_pts, legend_labels)

    # update fontsize
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(FONTSIZE)
    if ax.get_legend() is not None:
        for item in ax.get_legend().get_texts():
            item.set_fontsize(FONTSIZE)

    # ensure we have some handle on the ranges
    # if range_1 is None:
    #     range_1 = ax.get_xlim()
    # if range_2 is None:
    #     range_2 = ax.get_ylim()

    if range_2:
        ax.set_ylim(range_2)

    if isotropic:
        print("Setting isotropic for dims {} and {}".format(dim1, dim2))
        # plt.gca().set_aspect('equal', adjustable='box')
        # import pdb; pdb.set_trace()
        plt.gca().set_aspect('equal', adjustable='datalim')

        # manually calculate what the new xaxis must be...
        figW, figH = ax.get_figure().get_size_inches()
        xmid = (ax.get_xlim()[1] + ax.get_xlim()[0]) * 0.5
        yspan = ax.get_ylim()[1] - ax.get_ylim()[0]
        xspan = figW * yspan / figH

        # check if this increases span
        if xspan > ax.get_xlim()[1] - ax.get_xlim()[0]:
            ax.set_xlim(xmid - 0.5 * xspan, xmid + 0.5 * xspan)
        # if not, need to increase yspan
        else:
            ymid = (ax.get_ylim()[1] + ax.get_ylim()[0]) * 0.5
            xspan = ax.get_xlim()[1] - ax.get_xlim()[0]
            yspan = figH * xspan / figW
            ax.set_ylim(ymid - 0.5*yspan, ymid + 0.5*yspan)

        # import pdb; pdb.set_trace()
    elif range_1:
        ax.set_xlim(range_1)

    if color_labels is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        for i, color_label in enumerate(color_labels):
            ax.plot(1e10, 1e10, color=COLORS[i], label=color_label)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(loc='best')

    if marker_labels is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # import pdb; pdb.set_trace()
        for i, marker_label in enumerate(marker_labels):

            ax.scatter(1e10, 1e10, c='black',
                       marker=np.array(marker_style)[ordering][i],
                       # marker=MARKERS[list(marker_labels).index(marker_label)],
                       label=marker_label)
        if with_bg:
            ax.scatter(1e10, 1e10, c='xkcd:grey',
                       marker='.', label='Background')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(loc='best')

    # if marker_legend is not None:
    #     xlim = ax.get_xlim()
    #     ylim = ax.get_ylim()
    #     # import pdb; pdb.set_trace()
    #     for k, v in marker_legend.items():
    #         ax.scatter(1e10, 1e10, c='black',
    #                    marker=v, label=k)
    #     ax.set_xlim(xlim)
    #     ax.set_ylim(ylim)
    #     ax.legend(loc='best')

    if color_legend is not None and marker_legend is not None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # import pdb; pdb.set_trace()
        for label in color_legend.keys():
            ax.scatter(1e10, 1e10, c=color_legend[label],
            marker=marker_legend[label], label=label)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.legend(loc='best')

    if savefile:
        # set_size(4,2,ax)
        plt.savefig(savefile)
    # import pdb; pdb.set_trace()

    # return ax.get_window_extent(None).width, ax.get_window_extent(None).height

    return ax.get_xlim(), ax.get_ylim()


def plotMultiPane(dim_pairs, star_pars, groups, origins=None,
                  save_file='dummy.pdf', title=None):
    """
    Flexible function that plots many 2D slices through data and fits

    Takes as input a list of dimension pairs, stellar data and fitted
    groups, and will plot each dimension pair in a different pane.

    TODO: Maybe add functionality to control the data plotted in each pane

    Parameters
    ----------
    dim_pairs: a list of dimension pairs e.g.
        [(0,1), (3,4), (0,3), (1,4), (2,5)]
        ['xz', 'uv', 'zw']
        ['XY', 'UV', 'XU', 'YV', 'ZW']
    star_pars: either
        dicitonary of stellar data with keys 'xyzuvw' and 'xyzuvw_cov'
            or
        string filename to saved data
    groups: either
        a single synthesiser.Group object,
        a list or array of synthesiser.Group objects,
            or
        string filename to data saved as '.npy' file
    save_file: string, name (and path) of saved plot figure

    Returns
    -------
    (nothing)
    """

    # Tidying up inputs
    if type(star_pars) is str:
        star_pars = dt.loadXYZUVW(star_pars)
    if type(groups) is str:
        groups = np.load(groups)
        # handle case where groups is a single stored object
        if len(groups.shape) == 0:
            groups = groups.item()
    # ensure groups is iterable
    try:
        len(groups)
    except:  # groups is a single group instance
        groups = [groups]

    if origins:
        try:
            len(origins)
        except: # origins is a single group instance
            origins = [origins]

    # setting up plot dimensions
    npanes = len(dim_pairs)
    rows = int(np.sqrt(npanes)) #plots are never taller than wide
    cols = (npanes + rows - 1) // rows  # get enough cols
    ax_h = 5
    ax_w = 5
    f, axs = plt.subplots(rows, cols)
    f.set_size_inches(ax_w * cols, ax_h * rows)

    # drawing each axes
    for i, (dim1, dim2) in enumerate(dim_pairs):
        plotPane(dim1, dim2, axs.flatten()[i], groups=groups, origins=origins,
                 star_pars=star_pars, star_orbits=False,
                 group_then=True, group_now=True, group_orbit=True,
                 annotate=False)

    if title:
        f.suptitle(title)
    if save_file:
        f.savefig(save_file, format='pdf')


def sampleStellarPDFs(dim, star_pars, count=100):
    """
    Sample each provided star the build histogram from samples
    """
    all_samples = np.zeros(0)
    for mn, cov in zip(star_pars['xyzuvw'], star_pars['xyzuvw_cov']):
        samples = np.random.randn(count) * np.sqrt(cov[dim,dim]) + mn[dim]
        all_samples = np.append(all_samples, samples)
    return all_samples


def calcStellarPDFs(x, dim, star_pars):
    """
    For each point in `xs`, sum up the contributions from each star's PDF

    :param xs:
    :param star_pars:
    :return:
    """
    total = 0
    for mn, cov in zip(star_pars['xyzuvw'], star_pars['xyzuvw_cov']):
        total += dt.gauss(x, mn[dim], cov[dim,dim]**.5)
    return total


def evaluatePointInHist(x, hist_vals, hist_bins, normed=True):
    """
    Little utility function to evaluate the density of a histogram at point x

    NOTE! currently can't handle x > np.max(hist_bins)
    If x is below bin range, just extrapolate wings

    Parameters
    ----------
    x: a single value
    hist_vals: [nbin] number array; heights of each bin
    hist_bins: [nbin+1] float array; edges of bins
    """
    # if x < np.min(hist_bins) or x > np.max(hist_bins):
    #     return 0.

    bin_width = hist_bins[1] - hist_bins[0]
    if normed:
        hist_vals = hist_bins / float(np.sum(hist_vals))
    bin_height = hist_vals[np.digitize(x, hist_bins)]
    bin_density = bin_height / bin_width
    return bin_density


def plotBarAsStep(bins, hist, horizontal=False, ax=None, **kwargs):
    """
    Plot a bar plot to resemble a step plot
    :param bins:
    :param hist:
    :param horizontal:
    :param ax:
    :param kwargs:
    :return:
    """
    if ax is None:
        ax = plt.gca()

    # width = bins[1] - bins[0]
    # import pdb; pdb.set_trace()
    if horizontal:
        # ax.barh(bins[:-1], width=hist*weight/width, align='edge', edgecolor='black',
        #         color='none', height=width)
        adjusted_hist = np.hstack((0, hist[0], hist, 0))
        adjusted_bins = np.hstack((bins[0], bins[0], bins,
                                    ))
        ax.step(adjusted_hist, adjusted_bins, where='pre',
                **kwargs)
        # ax.plot(np.max(hist)*weight/width*1.1, np.median(bins), alpha=0)
        # xlim = ax.get_xlim()
        # ax.set_xlim((0, xlim[1]))
    else:
        # ax.bar(bins[:-1], height=hist*weight/width, align='edge', edgecolor='black',
        #        color='none', width=width)
        adjusted_hist = np.hstack((0, hist, hist[-1], 0))
        adjusted_bins = np.hstack((bins[0], bins, bins[-1]))
        ax.step(adjusted_bins, adjusted_hist, where='post',
                **kwargs)
        # ax.plot(np.median(bins), np.max(hist)*weight/width*1.1, alpha=0)
        # ylim = ax.get_ylim()
        # ax.set_ylim((0, ylim[1]))


def plotManualHistogram(data, bins, span=None, ax=None, weight=1.0,
                        horizontal=False, **kwargs):
    """
    TODO: Need to work out how to get span to be correctly incorporated into histograms
    :param data:
    :param bins:
    :param span:
    :param ax:
    :param weight:
    :param horizontal:
    :param kwargs:
    :return:
    """
    if ax is None:
        ax = plt.gca()
    # if restricting range, ensure weighting is accounted for
    if span:
        inv_weight = 1./weight
        data_mask = np.where((data > span[0]) & (data < span[1]))
        frac_kept = len(data_mask) / float(len(data))
        inv_weight *= frac_kept
        inv_weight = 1./inv_weight
        data = data[data_mask]

        # if bins is just an integer,

    hist, edges = np.histogram(data, bins=bins, range=span)
    width = edges[1] - edges[0]

    scaled_hist = hist*weight/width

    plotBarAsStep(edges, scaled_hist, horizontal=horizontal, ax=ax,
                  **kwargs)

    return scaled_hist, edges


def plot1DProjection(dim, star_pars, groups, weights, ax=None, horizontal=False,
                     bg_hists=None, with_bg=False, membership=None,
                     x_range=None, use_kernel=False, residual=False):
    """
    Given an axes object, plot the 1D projection of stellar data and fits

    :param dim:
    :param star_pars:
    :param groups:
    :param z:
    :param vertical:
    bg_hists: [6, 2, ~nbins] list
        for each of the six dimensions, has two elements: the bin heights,
        and the bin edges
    :return:
    """
    BIN_COUNT=19
    bg_hist_kwargs = {'c':'black', 'alpha':0.5} #, 'ls':'--'}
    comb_group_kwargs = {'c':'black', 'alpha':0.7, 'ls':'-.'}
    resid_kwargs = {'c':'black', 'alpha':0.5, 'linestyle':':'}

    # if horizontal:
    #     orientation = 'horizontal'
    # else:
    #     orientation = 'vertical'
    weights = np.array(weights).astype(np.float)
    if len(weights.shape) > 1:
        weights = weights.sum(axis=0)

    star_pars_cp = star_pars

    if x_range is None:
        x_range = [
            np.min(star_pars['xyzuvw'][:, dim]),
            np.max(star_pars['xyzuvw'][:, dim]),
        ]
        buffer = 0.1 * (x_range[1] - x_range[0])
        x_range[0] -= buffer
        x_range[1] += buffer

    if ax is None:
        ax = plt.gca()

    npoints = 100
    if use_kernel:
        nstars = len(star_pars['xyzuvw'])
        xs = np.linspace(x_range[0], x_range[1], npoints)
        kernel = stats.gaussian_kde(star_pars['xyzuvw'][:,dim], bw_method=0.3)
        if horizontal:
            ax.plot(nstars*kernel.evaluate(xs), xs, **bg_hist_kwargs) #c='black', ls='--', alpha=0.5)
        else:
            ax.plot(xs, nstars*kernel.evaluate(xs), **bg_hist_kwargs) # c='black', ls='--', alpha=0.5)
    else:
        nsamples = 1000
        data = sampleStellarPDFs(dim, star_pars_cp, count=nsamples)
        scaled_hist, bins = plotManualHistogram(
            data, bins=BIN_COUNT, span=x_range, weight=1./nsamples,
            horizontal=horizontal, **bg_hist_kwargs
        )
        xs = np.linspace(np.min(bins), np.max(bins), npoints)
        # vals, bins, _ = \
        #     ax.hist(sampleStellarPDFs(dim, star_pars_cp), normed=False, histtype='step',
        #             orientation=orientation, bins=BIN_COUNT)


    # Calculate and plot individual PDFs of fitted groups, with appropriate
    # relative weighting, but normalised such that the sum of areas of all groups
    # is 1.
    # Simultaneously, calculate the combined PDF of fitted groups
    # xs = np.linspace(np.min(bins), np.max(bins), npoints)
    combined_gauss = np.zeros(xs.shape)
    for i, (group, weight) in enumerate(zip(groups, weights)):
        mean_now = torb.trace_cartesian_orbit(group.mean, group.age, single_age=True)
        cov_now = tf.transform_covmatrix(group.generateCovMatrix(),
                                         torb.trace_cartesian_orbit,
                                         group.mean, args=[group.age])
        group_gauss = weight*dt.gauss(xs, mean_now[dim],
                                      np.sqrt(cov_now[dim,dim]))
        combined_gauss += group_gauss
        if bg_hists is not None:
            hist_contrib = weights[-1]*\
                           evaluatePointInHist(xs, bg_hists[dim][0],
                                               bg_hists[dim][1])
            combined_gauss += hist_contrib

        if horizontal:
            ax.plot(group_gauss, xs, color=COLORS[i], alpha=0.6)
        else:
            ax.plot(xs, group_gauss, color=COLORS[i], alpha=0.6)

    # Plot the combined PDF of fitted groups
    # only plot combined fit if theres more than one group
    if len(groups) > 1:
        if horizontal:
            ax.plot(combined_gauss, xs, **comb_group_kwargs) # color='black', ls='--')
        else:
            ax.plot(xs, combined_gauss, **comb_group_kwargs) # color='black', ls='--')

    # plot the difference of combined fit with histogram
    if residual:
        if use_kernel:
            if horizontal:
                ax.plot(nstars*kernel.evaluate(xs) -combined_gauss, xs, **resid_kwargs)
            else:
                ax.plot(xs, nstars*kernel.evaluate(xs)-combined_gauss, **resid_kwargs)
        else:
            bin_width = bins[1] - bins[0]
            combined_gauss_vals = np.interp(bins[:-1]+bin_width, xs, combined_gauss)
            plotBarAsStep(bins, scaled_hist - combined_gauss_vals, horizontal=horizontal,
                          **resid_kwargs)

    # Ensure histograms are flush against the data axis
    if horizontal:
        xlim = ax.get_xlim()
        ax.set_xlim(0, xlim[1])
    else:
        ylim = ax.get_ylim()
        ax.set_ylim(0, ylim[1])

    # update fontsize
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(FONTSIZE)
    if ax.get_legend() is not None:
        for item in ax.get_legend().get_texts():
            item.set_fontsize(FONTSIZE)


def plotPaneWithHists(dim1, dim2, fignum=None, groups=[], weights=None,
                      star_pars=None,
                      star_orbits=False,
                      group_then=False, group_now=False, group_orbit=False,
                      annotate=False, bg_hists=None, membership=None,
                      true_memb=None, savefile='', with_bg=False,
                      range_1=None, range_2=None, residual=False,
                      markers=None, group_bg=False, isotropic=False,
                      color_labels=[], marker_labels=[], marker_order=[],
                      ordering=None, no_bg_covs=False):
    """
    Plot a 2D projection of data and fit along with flanking 1D projections.

    Uses global constants COLORS and HATCHES to inform consistent colour
    scheme.
    Can use this to plot different panes of one whole figure

    TODO: Incorporate Z
    TODO: incoporate background histogram

    Parameters
    ----------
    dim1: x-axis, can either be integer 0-5 (inclusive) or a letter form
          'xyzuvw' (either case)
    dim2: y-axis, same conditions as dim1
    fignum: figure number in which to create the plot
    groups: a list of (or just one) synthesiser.Group objects, corresponding
            to the fit of the origin(s)
    star_pars:  dict object with keys 'xyzuvw' ([nstars,6] array of current
                star means) and 'xyzuvw_cov' ([nstars,6,6] array of current
                star covariance matrices)
    star_orbits: (bool) plot the calculated stellar traceback orbits of
                        central estimate of measurements
    group_then: (bool) plot the group's origin
    group_now:  (bool) plot the group's current day distribution
    group_orbit: (bool) plot the trajectory of the group's mean
    annotate: (bool) add text describing the figure's contents
    with_bg: (bool) treat the final column of Z as background memberships
             and color accordingly

    Returns
    -------
    (nothing returned)
    """
    labels = 'XYZUVW'
    axes_units = 3*['pc'] + 3*['km/s']
    if type(membership) is str:
        membership = np.load(membership)
    if type(star_pars) is str:
        star_pars = dt.loadXYZUVW(star_pars)
    # if ordering:
    #     membership = membership[:,ordering]

    # TODO: clarify what exactly you're trying to do here
    if weights is None and len(groups) > 0:
        if len(groups) == 1 and not with_bg:
            weights = np.array([len(star_pars['xyzuvw'])])
        elif membership is not None:
            weights = membership.sum(axis=0)
        else:
            weights = np.ones(len(groups)) / len(groups)

    if type(dim1) is not int:
        dim1 = labels.index(dim1.upper())
    if type(dim2) is not int:
        dim2 = labels.index(dim2.upper())
    if type(groups) is str:
        groups = np.load(groups)
        if len(groups.shape) == 0:
            groups = np.array(groups.item())
    if type(bg_hists) is str:
        bg_hists = np.load(bg_hists)

    # Set global plt tick params???
    tick_params = {'direction':'in', 'top':True, 'right':True}
    plt.tick_params(**tick_params)

    # Set up plot
    fig_width = 5 #inch
    fig_height = 5 #inch
    fig = plt.figure(fignum, figsize=(fig_width,fig_height))
    plt.clf()
    # gs = gridspec.GridSpec(4, 4)
    gs = gridspec.GridSpec(4, 4)

    # Set up some global plot features
    # fig.set_tight_layout(tight=True)
    plt.figure()

    # Plot central pane
    axcen = plt.subplot(gs[1:, :-1])
    xlim, ylim = plotPane(
        dim1, dim2, ax=axcen, groups=groups, star_pars=star_pars,
        star_orbits=star_orbits, group_then=group_then,
        group_now=group_now, group_orbit=group_orbit, annotate=annotate,
        membership=membership, true_memb=true_memb, with_bg=with_bg,
        markers=markers, group_bg=group_bg, isotropic=isotropic,
        range_1=range_1, range_2=range_2, marker_labels=marker_labels,
        color_labels=color_labels, ordering=ordering, no_bg_covs=no_bg_covs)
    plt.tick_params(**tick_params)
    # if range_1:
    #     plt.xlim(range_1)
    # if range_2:
    #     plt.ylim(range_2)
    # plt.grid(gridsepc_kw={'wspace': 0, 'hspace': 0})
    # plt.sharex(True)

    # Plot flanking 1D projections
    # xlim = axcen.get_xlim()
    axtop = plt.subplot(gs[0, :-1])
    axtop.set_xlim(xlim)
    axtop.set_xticklabels([])
    plot1DProjection(dim1, star_pars, groups, weights, ax=axtop,
                     bg_hists=bg_hists, with_bg=with_bg, membership=membership,
                     residual=residual, x_range=xlim)
    axtop.set_ylabel('Stars per {}'.format(axes_units[dim1]))
    plt.tick_params(**tick_params)
    # axcen.set_tick_params(direction='in', top=True, right=True)

    # ylim = axcen.get_ylim()
    axright = plt.subplot(gs[1:, -1])
    axright.set_ylim(ylim)
    axright.set_yticklabels([])
    plot1DProjection(dim2, star_pars, groups, weights, ax=axright,
                     bg_hists=bg_hists, horizontal=True, with_bg=with_bg,
                     membership=membership, residual=residual,
                     x_range=ylim)
    axright.set_xlabel('Stars per {}'.format(axes_units[dim2]))
    # axcen.set_tick_params(direction='in', top=True, right=True)
    plt.tick_params(**tick_params)
    # plt.tight_layout(pad=0.7)

    axleg = plt.subplot(gs[0,-1])
    for spine in axleg.spines.values():
        spine.set_visible(False)
    axleg.tick_params(labelbottom='off', labelleft='off', bottom='off',
                   left='off')
    # import pdb; pdb.set_trace()

    if False:
        for label_ix, marker_ix in enumerate(marker_order):
            axleg.scatter(0,0,color='black',marker=MARKERS[marker_ix],
                          label=MARKER_LABELS[label_ix])
        for i, color_label in enumerate(color_labels):
            axleg.plot(0,0,color=COLORS[i],label=color_label)
        axleg.legend(loc='best', framealpha=1.0)

    # for i, marker_label in enumerate(marker_labels):
    #     axleg.scatter(0,0,color='black',marker=MARKERS[i],label=marker_label)
    # pt = axleg.scatter(0,0, label='Dummy')
    # plt.legend([pt], ["Test"])

    # import pdb; pdb.set_trace()

    if savefile:
        plt.savefig(savefile)

    return xlim, ylim


def plotPointCov(points, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma ellipse based on the mean and covariance of a point
    "cloud" (points, an Nx2 array).
    Parameters
    ----------
        points : An Nx2 array of the data points.
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    return plotCovEllipse(cov, pos, nstd, ax, **kwargs)


