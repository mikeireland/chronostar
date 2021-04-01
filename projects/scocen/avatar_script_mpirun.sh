##### Select resources #####
#PBS -N bg_ols_10
#PBS -l select=1:ncpus=16:mpiprocs=16
##### Queue #####
#PBS -q smallmem
##### Mail Options #####
# Send an email at job start, end and if aborted
#PBS -m abe

# This is an example script...

cd /data/mash/marusa/chronostar_projects/solar_neighbourhood/

/pkg/linux/anaconda/bin/mpirun -np 16 /pkg/linux/anaconda/bin/python bg_ols_multiprocessing.py 10
