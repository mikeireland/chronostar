## Fastfit
Expectation-maximisation algorithm `run_em.py` can be run independently. 'fastfit.py' is the main code here, and is run as 'python3 fastfit.py parameters.pars'. `fastfit.py` is a simplified version of Chronostar. It is using gradiend descent only.

- `run_em.py`: parameters can either be entered as an argument to `run_em.run_expectmax_simple` if called from another script. It `run_em.py` is run independently, it is run as `run_em.py example_run_em.pars`. Note that one or both of parameters `filename_init_memb_probs` or `filename_init_comps` are mandatory.

- `fastfit.py` is run as `python3 fastfit.py example_synthetic.pars`. 
