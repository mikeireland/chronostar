"""
General purpose plotting script.

Usage:
python plot_result.py -d star_data.fits -c final_comps.npy -m final_membership.npy \
    -n my_plot.pdf --dims XY UV XU YV ZW

Note that `--dims` expects a list of pairs of dimensions. For each dimension
pair a subplot will be plotted. The first dimension on the xaxis, the second
on the y axis. The number of subplots will be dynamically calculated to
accommodate the list of dimension pairs.

Requires:
- a fits file with stellar cartesian positions
- a membership file (2D array stored as .npy file)
- a component file
- if chronostar is in your python path, then this should be callable from
    anywhere

Limitations:
- currently hardcoded for SphereComponents only
- number of rows of subplots is hardcoded to 2
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np

from chronostar.component import SphereComponent
from chronostar import tabletool

# Set up command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("-d", "--datafile",
                    help="path to a fits file with all stellar data")
parser.add_argument("-c", "--compfile",
                    help="path to the component file")
parser.add_argument("-m", "--membfile",
                    help="path to the membership file")
parser.add_argument("-o", "--outdir",
                    help="path to output directory (will be created if not existing)",
                    default="./")
parser.add_argument("-n", "--plotname",
                    help="name of plot",
                    default="comp_plot")


parser.add_argument( "--dims",
                     nargs="*",
                     default=['XY', 'UV', 'XU', 'YV'],
                     help="Dimensions to plot on each subplot",
)

args = parser.parse_args()

# Input data
# data_table = tabletool.read('beta_Pictoris_with_gaia_small_everything_final_radial_velocity_reinit_masked.fits')
data_table = tabletool.read(args.datafile)

best_fit_comps = SphereComponent.load_raw_components(args.compfile)
memberships = np.load(args.membfile)

p = 0.5 # If membership probability >p, a star is a member of this component
##################

# Get python's default(?) list of colors
prop_cycle = plt.rcParams['axes.prop_cycle']
COLORS = prop_cycle.by_key()['color']

# Create data dict
means_all = tabletool.build_data_dict_from_table(
        data_table,
        get_background_overlaps=True,
        only_means=True,
)

labels = 'XYZUVW'
units = 3*['pc'] + 3*['km/s']

# Convert character dims to integers
int_dims = [[labels.index(char.upper()) for char in dim] for dim in args.dims]

"""
We build subplot arrangement dynamically to allow for variable length input lists
For simplicity the number of rows in plot is fixed to 2
"""
nplots = len(int_dims)
nrows = 2
ncols = int(np.ceil(nplots/nrows))

axsize = 5
fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                         figsize=(ncols*axsize, nrows*axsize))
fig.set_tight_layout(True)

legend_set = False
for ax, (dim1, dim2) in zip(axes.flatten(), int_dims):
    ax.tick_params(direction='in')
    print(dim1, dim2)

    ax.set_xlabel('{} [{}]'.format(labels[dim1], units[dim1]))
    ax.set_ylabel('{} [{}]'.format(labels[dim2], units[dim2]))

    # means_all = data_dict['means'] # XYZUVW

    for i, best_fit_comp in enumerate(best_fit_comps):
        # Plot the component as an ellipse. See chronostar.component.plot for
        # detailed description of parameters
        best_fit_comp.plot(dim1, dim2, comp_now=True, comp_then=False, color=COLORS[i], ax=ax,
                       comp_orbit=False, orbit_color=COLORS[i], label=chr(ord('A')+i))

        # Find all stars with memberships greater than `p` and plot them with
        # the same colour
        comp_members_mask = memberships[:,i]>p
        comp_members = means_all[comp_members_mask]
        ax.scatter(comp_members[:, dim1], comp_members[:, dim2], c=COLORS[i], s=1)

    # Plot the background stars, if they exist
    if len(memberships[0]) > len(best_fit_comps):
        # If there are more columns than components, then there must be background stars
        bg_members_mask = memberships[:,-1]>p
        bg_members = means_all[bg_members_mask]
        ax.scatter(bg_members[:, dim1], bg_members[:, dim2], c='grey', s=0.1)

    # Set the legend for the first subplot only
    if not legend_set:
        ax.legend(loc='best')
        legend_set = True


fig.savefig(args.outdir+args.plotname)
plt.show()
