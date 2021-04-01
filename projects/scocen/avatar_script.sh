##### Select resources #####
#PBS -N scocen
#PBS -l select=1:ncpus=16:mpiprocs=16
##### Queue #####
#PBS -q smallmem
##### Mail Options #####
# Send an email at job start, end and if aborted
#PBS -m abe

cd /priv/mulga1/marusa/chronostar_projects/scocen/

/pkg/linux/anaconda/bin/python run_chronostar.py mypars.pars
