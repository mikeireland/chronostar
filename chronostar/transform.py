"""
A module which aids in the transformation of a covariance matrix
between two coordinate frames.

These functions are used to convert astrometric covariance matrices into
cartesian covariance matrices. Also to project cartesian Gaussian distributions
forward (or backward) through the Galactic potential.
"""

import numpy as np


def calc_jacobian_column(trans_func, col_number, loc, dim=6, h=1e-3, args=None,
                         fin_pl=None, fin_mi=None):
    """
    Calculate a column of the Jacobian.

    A whole column can be done in one hit because we are incrementing
    the same parameter of the initial coordinate system. This is
    accurate to the first order by taking the difference of a forward increment
    with a backward increment
    See https://en.wikipedia.org/wiki/Numerical_differentiation#Higher-order_methods
    for details.

    Parameters
    ----------
    trans_func: function
        Transformation function taking us from the initial coordinate frame
        to the final coordinate frame
    col_number: int
        The index in question (which parameter of the initial frame we are
        incrementing
    loc: [dim] float array
        The position (in the initial coordinte frame) around which we are
        calculting the jacobian
    dim: integer {6}
        The dimensionality of the coordinate frames.
    h: float {1e-3}
        The size of the increment
    args: tuple
        Extra arguments required by `trans_func`, e.g 'age' for the
        traceforward function

    Returns
    -------
    result : [6] float array
        The `col_number`th column of the ultimate Jacobian matrix
    """
    offset = np.zeros(dim)
    offset[col_number] = h
    loc_pl = loc + offset
    loc_mi = loc - offset
    if args is None:
        return (trans_func(loc_pl) - trans_func(loc_mi)) / (2*h) 
    else:
        return (trans_func(loc_pl, *args) - trans_func(loc_mi, *args)) / (2*h)


def calc_jacobian(trans_func, loc, dim=6, h=1e-3, args=None):
    """
    Calculate the Jacobian of the coordinate transfromation `trans_func` about
    `loc`.

    `trans_func` should take a vector of dimension `dim` to a new vector
    of same dimension. This function then calculates the 2nd order partial
    derivative at point `loc`. Extra arguments for `trans_func` can be
    provided as a tuple to `args`.

    Parameters
    ----------
    trans_func : function
        Transformation function taking us from the initial coordinate frame
        to the final coordinate frame
    loc : [dim] float array
        The position (in the initial coordinte frame) around which we are
        calculating the jacobian
    dim : int {6}
        The dimensionality of the coordinate frames
    h : float {1e-3}
        The size of the increment, smaller values maybe run into numerical
        issues
    args : tuple {None}
        Extra arguments required by `trans_func`

    Returns
    -------
    jac : [dim,dim] float array
        A jacobian matrix

    Notes
    -----
        OPTIMISATION TARGET
    The application of `trans_func` is the bottleneck of Chronostar
    (at least when `trans_func` is traceorbit.trace_cartesian_orbit).
    Since this is a loop, there is scope for parallelisation.
    """
    if args is None:
        args = []

    jac = np.zeros((dim, dim))

    # Even with epicyclic, this constitutes 90% of chronostar work
    # so, we pass all 12 required positions to the trans_func as
    # one array, to exploit numpy's faster array operations
    start_pos = []
    for i in range(dim):
        offset = np.zeros(dim)
        offset[i] = h
        loc_pl = loc + offset
        loc_mi = loc - offset
        start_pos.append(loc_pl)
        start_pos.append(loc_mi)
    start_pos = np.array(start_pos)

    final_pos = trans_func(start_pos, *args)

    # return (trans_func(loc_pl) - trans_func(loc_mi)) / (2*h)

    for i in range(dim):
        jac[:,i] = (final_pos[2*i] - final_pos[2*i + 1]) / (2*h)

#    for i in range(dim):
#        jac[:,i] = calc_jacobian_column(trans_func, i, loc, dim, h, args)

    return jac


def transform_covmatrix(cov, trans_func, loc, dim=6, h=1e-3, args=None):
    """
    Transforming a covariance matrix from one coordinate frame to another

    Parameters
    ----------
    cov : [dim,dim] float array
        Covariance matrix in the initial frame
    trans_func : function
        Transformation function taking us from the initial
        coordinate frame to the final coordinate frame. Output must be
        mutable, i.e. single value, or an array
    loc : [dim] float array
        The position (in the initial coordinate frame)
        around which we are calculating the jacobian
        (i.e. the mean, in the example of a Gaussian distribution)
    dim : integer {6}
        The dimensionality of the coordinate frame
    h : float {1e-3}
        The size of the increment, smaller values maybe run into numerical
        issues
    args : tuple
        extra args to be passed to trans_func. E.g. for traceOrbitXYZUVW
        args = (age,) [for traceforward] or args = (-age,) [for traceback]

    Returns
    -------
    conv_cov : [dim,dim] float array
        The transformed covariance matrix
    """
    jac = calc_jacobian(trans_func, loc, dim=dim, h=h, args=args)
    return np.dot(jac, np.dot(cov, jac.T))

