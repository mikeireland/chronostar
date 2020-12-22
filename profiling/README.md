## Investigate time efficiency of the code
Run with 
`kernprof -l python_code.py data.fits`

Example datafile is `/home/marusa/chronostar/scripts/beta_Pictoris_with_gaia_small_everything_final_radial_velocity_reinit_masked.fits`

Read output with
`python -m line_profiler python_code.py.lprof`

- `kern_prof_compfitter.py`: Investigates `compfitter`, namely component overlaps and orbit computation. 
  **NOTE**: You need to add `@profile` decorator before `def get_lnoverlaps(...)` in `likelihood.py` in order to get profiling right.
- `profile_compfitter.py`: More general breakdown.
