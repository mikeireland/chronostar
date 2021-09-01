## Fastfit
Expectation-maximisation algorithm `run_em.py` can be run independently. 'fastfit.py' is the main code here, and is run as 'python3 fastfit.py parameters.pars'. `fastfit.py` is a simplified version of Chronostar. It is using gradiend descent only.

- `run_em.py`: parameters can either be entered as an argument to `run_em.run_expectmax_simple` if called from another script. It `run_em.py` is run independently, it is run as `run_em.py example_run_em.pars`. Note that one or both of parameters `filename_init_memb_probs` or `filename_init_comps` are mandatory.

- `fastfit.py` is run as `python3 fastfit.py example_synthetic.pars`. 




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
