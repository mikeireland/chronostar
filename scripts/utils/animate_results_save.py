"""
For efficiency, if a video already exists, this script will
skip over it. So if you wish to update a video, delete it 
first, and then run this script.

usage:
in the base results directory (i.e. you should be able to see directories 1,
2, ... etc) execute:

python animate_results_save.py path_to_your_data.fits [N_MAX_COMPS] [dim1] [dim2]
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from chronostar.component import SphereComponent
from chronostar import tabletool

N_MAX_COMPS=20
N_MAX_ITERS=200
LABELS='XYZUVW'
UNITS=3*['pc'] + 3*['km/s']

dim1 = 0
dim2 = 3

def animate(i):
    COLOURS = ['blue', 'red', 'orange', 'purple', 'brown', 'green', 'black']
    iterdir = base_dir + 'iter{:02}/'.format(i)
    print('[animate]: In ', iterdir)
    comps = SphereComponent.load_raw_components(iterdir + 'best_comps.npy')
    membs = np.load(iterdir + 'membership.npy')

    plt.clf()
    plt.title('Iter {}'.format(i))
    plt.xlabel('{} [{}]'.format(LABELS[dim1], UNITS[dim1]))
    plt.ylabel('{} [{}]'.format(LABELS[dim2], UNITS[dim2]))

    comp_assignment = np.argmax(membs, axis=1)

    # Plot color coded component members
    for comp_ix in range(len(comps)):
        memb_mask = np.where(comp_assignment == comp_ix)
        plt.plot(data['means'][memb_mask,dim1], data['means'][memb_mask,dim2], '.',
                 color=COLOURS[comp_ix%len(COLOURS)], alpha=0.6, markersize=10)

    # Plot background stars
    bg_mask = np.where(comp_assignment == len(comps))
    plt.plot(data['means'][bg_mask,dim1], data['means'][bg_mask,dim2], '.',
             color='grey', alpha=0.3, markersize=2)

    [c.plot(dim1, dim2, color=COLOURS[comp_ix%len(COLOURS)]) for comp_ix, c in enumerate(comps)]
    # [c.plot(dim1, dim2) for c in comps]


Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)

if len(sys.argv) < 2:
    print('USAGE: python animate_results_save.py datafile.fits [max_comps]')
    sys.exit()
datafile = sys.argv[1]
if len(sys.argv) > 2:
    N_MAX_COMPS = int(sys.argv[2])
if len(sys.argv) > 4:
    dim1 = sys.argv[3]
    dim2 = sys.argv[4]
    try:
        dim1 = int(dim1)
    except ValueError:
        dim1 = ord(dim1.upper()) - ord('X')
    try:
        dim2 = int(dim2)
    except ValueError:
        dim2 = ord(dim2.upper()) - ord('X')

try:
    data = tabletool.build_data_dict_from_table(datafile)
except:
    data = tabletool.build_data_dict_from_table(datafile, historical=True)

fig = plt.figure(figsize=(10,6))

ani = matplotlib.animation.FuncAnimation(fig, animate, frames=N_MAX_ITERS, repeat=True)

for i in range(1,N_MAX_COMPS+1):
    if i == 1:
        base_dir = '1/'
        save_filename = '1_{}{}.mp4'.format(LABELS[dim1],LABELS[dim2])
        if os.path.isdir(base_dir) and not os.path.isfile(save_filename):
            print('Going into ', base_dir)
            try:
                ani.save(save_filename, writer=writer)
            except: # FileNotFoundError: # Python2 consistent
                pass
    else:
        subdirs = [f.name for f in os.scandir(str(i)) if f.is_dir()] 
        print('Subdirs: ', subdirs)
        for subdir in subdirs:
            # char = chr(ord('A')+j)
            # base_dir = '{}/{}/'.format(i,char)

            base_dir = '{}/{}/'.format(i,subdir)
            save_filename = '{}_{}_{}{}.mp4'.format(i,subdir,LABELS[dim1],LABELS[dim2])
            if os.path.isdir(base_dir) and not os.path.isfile(save_filename):
                print('Going into ', base_dir)
                try:
                    ani.save(save_filename, writer=writer)
                except: # FileNotFoundError: # Python2 consistent
                    pass


