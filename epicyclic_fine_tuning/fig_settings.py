import matplotlib
import matplotlib.ticker as ticker

#~ figsize=(8.27, 11.69/3)
figsize=(11.69/3, 8.27)
fontsize=8#12


matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
font = {'size': fontsize}
matplotlib.rc('font', **font)
matplotlib.rc('ytick.major', size=8)
matplotlib.rc('ytick.minor', size=6)
matplotlib.rc('xtick.major', size=10)
matplotlib.rc('xtick.minor', size=7)
