## Fastfit
Expectation-maximisation algorithm `run_em.py` can be run independently. 'fastfit.py' is the main code here, and is run as 'python3 fastfit.py parameters.pars'. `fastfit.py` is a simplified version of Chronostar. It is using gradiend descent only.

- `run_em.py`: parameters can either be entered as an argument to `run_em.run_expectmax_simple` if called from another script. It `run_em.py` is run independently, it is run as `python3 run_em.py example_run_em.pars`. Note that one or both of parameters `filename_init_memb_probs` or `filename_init_comps` are mandatory.

- `fastfit.py` is run as `python3 fastfit.py example_synthetic.pars`. 

- `run_em_c_modules.c`: `run_em`, but with C modules for expectation and maximisation. TODO.


### More notes
Folder fastfit now includes an expectation-maximisation algorithm run_em.py that can be run independently. It can take a parameter file as an input (init comps and/or init memberships must be specified). It is using gradient descent for optimisation. emcee is removed.

fastfit.py  is a simplified chronostar that is using run_em.py . It’s essentially the same as run_chronostar.py, except that it’s less robust in order to gain speed.

Other differences:
- Revised convergence criteria for gradient descent (tol=1 instead of tol=0.025 makes it an order of magnitude faster)
- In the EM algorithm, we start with the maximisation step first and expectation second. This is because the initial data either includes membership probabilities from the input file or runs the expectation step from other information before the EM loop.
- EM convergence criterion: I discovered a bug in the original Chronostar (not fixed there yet!) that arose with the introduction of gradient descent. Convergence criteria were almost never met, and for this reason EM algorithm always needed many iterations.
Attached figure shows lnprob function (divided by its max value (which is negative) in this plot). This run reached 200 iterations (which is maximum allowed number - max_em_iterations in the params file). However, it is clear that the convergence occurs sooner. The convergence criterion in the original Chronostar is based on BIC. Here, I compute average lnprob value for every 5 iterations (this is a customisable parameter ‘lnlike_convergence_slice_size’) this is represented with the red line in the plot (and the y-axis in the right). As soon as the median is higher than the previous one, the convergence is achieved. E.g. this happens at iteration ~60 in this plot. This is not ideal but hopefully good enough to speed up the process. BIC criterion is removed in fastfit.py because there was a bug we didn’t fully understand.

- No ability for Chronostar to read-in results from the previous runs in case it breaks while running. However, it is possible to manually enter the last result as a starting point of a new run (init_comps or init_memb_probs). After we are sure this version of the code works well, I will remove file printouts for each EM iteration in order to gain speed.

- Component split folders are now called S1, S2 etc. instead of ABC. We reached 22 components in Sco-Cen, and will likely have more in the future.

Next step:
I’ve tested the code with bPic data. The result has 2 components, and one of them is beta Pic, but it has significantly less members than in Tim’s paper. Also, 2 comps are less than in Tim’s paper though (6 components). Need to run Chronostar with emcee again to see if results agree and try to understand why we get different results.




### Changes (making Chronostar more modular)
- Maximisation is now in `maximisation.py`. It doesn't take `Component` objects anymore, but arrays. 
- Conversion of velocities from km/s to pc/Myr should be done in the input data. This would avoid unit conversion in every single step of epicyclic propagation. Input velocities are left in km/s for now. TODO.
- Add ro, vo, Oort's constants etc. to the params list.
- Fix the LSR bug (sign). Add LSR to the params file.
- prepare_data (astrometry to cartesian): Implement distances, e.g. from Bailer-Jones. What to do with assymetric uncertainties, and correlation coefficients? See this paper as well: https://arxiv.org/abs/2111.01860

### Already implemented
- `chronostar/prepare_input_data.py`: run with `python3 ../chronostar/prepare_input_data.py example_convert_astrometry_to_XYZUVW.pars`. This takes a table with astrometry downloaded from Gaia website and transforms it into cartesian system. It replaces the missing radial velocities with the values provided. Parameters are in `example_convert_astrometry_to_XYZUVW.pars`.

### Modules in C
- `expectation.c` with corresponding swig module now works. `fastfit/test_expectation.py` compares the result from python and C modules and finds no difference. C module is 10 times faster than python. There are, however, a few points left to implement, e.g. whether to include an option to exclude background, posterior etc. Also, `get_overall_lnlikelihood` is TO BE DONE.
- `expectation.c`: `expectation_iterative_component_amplitudes`: BIC removed as a convergence criterion. Convergence achieved when `np.all((new_memb_probs-old_memb_probs)<0.1)`.
- `expectation.c`: `get_overall_lnlikelihood` doesn't compute new expectation but takes the one that has been computed just before this step in `run_em.py`

- `temporal_propagation.c`: Works for a 6D point and for a covariance matrix. Swig done, ready to use in python. Test works well (`fastfit/test_temporal_propagation.py`). `trace_epicyclic_orbit` is 30 times faster than in python, and covmatrix transformation is 10 times faster in C.

- `setup.py`: `_overlap` can be removed for C Chronostar because it is already included in `expectation.c`

### Running tests
- Tests: `test_expecctation.py`, `test_likelihood.py`, `test_temporal_propagation.py`: input data is in `data_for_testing`. It is on github.
