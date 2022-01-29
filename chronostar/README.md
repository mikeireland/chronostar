# TODO

- `likelihood.c`: Originally only stars with `memb_prob>treshold` are used in the overlaps. I remove this from likelihood for now. This should be done before the scipy.optimisation is called. KEEP IN MIND when fitting!!!
- Same with `minimum_exp_starcount`
- `trace_epicyclic_orbit` transforms velocities to pc/Myr every single time by multiplying by 1.02. Couldn't we sample in pc/Myr space to avoid this? Stellar velocities could also be in this units from the very beginning.
- `likelihood.lnprior`: uses `nstars`=sum(memb_probs) as an estimate for a number of members. Is this number needed at any other place? Compute it once and propagate it to avoid long for loops.
- `MAX_AGE=500Myr` is hardcoded in `likelihood.c`
- `maximisationC.fit_single_comp_gradient_descent_serial`: Age split hardcoded
