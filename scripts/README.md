### `run_fastfit.py`
Chronostar fits all components in the Maximisation step in parallel.

Run Chronostar with the fast (parallel) code with `avatar_run_fastfit.pbs`. This will run `run_fastfit.py bpic_mpi.pars` in the parallel mode.

### Dev
- See `#parallelism` on Slack
- Line profiling `kernprof`
- There is some code in `chronostar/profiling/` that Tim wrote.
