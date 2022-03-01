"""
Various functions that improve the functionality but are not the core
of Chronostar
"""

import logging

def log_message(msg, symbol='.', surround=False):
    """Little formatting helper"""
    res = '{}{:^40}{}'.format(5 * symbol, msg, 5 * symbol)
    if surround:
        res = '\n{}\n{}\n{}'.format(50 * symbol, res, 50 * symbol)
    logging.info(res)


def get_gr_mns_covs_now(comps):
    """
    Get gr_mns and gr_covs from [comps] for C modules
    Temporal propagation happens here.
    """
    
    # Means
    dim = len(comps[0].get_mean())
    gr_mns = [trace_epicyclic_orbit(comp.get_mean(), comp.get_age(), 
        dim) for comp in comps]

    # Covmatrices
    c = comps[0].get_covmatrix()
    dim1 = c.shape[0]
    dim2 = c.shape[1]
    h=1e-3 # HARDCODED... TODO
    gr_covs = [trace_epicyclic_covmatrix(
        c.get_covmatrix(), c.get_mean(), c.get_age(), h, 
        dim1*dim2).reshape(dim1, dim2) for c in comps]
        
    return gr_mns, gr_covs
