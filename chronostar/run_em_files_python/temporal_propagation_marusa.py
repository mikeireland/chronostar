"""
Temporal propagation of the component: Take a component at birth time
(time 0) and propagate it for its age t to the current position.

trace_epicyclic_orbit traces a 6D point
trace_epicyclic_covmatrix traces covariance matrix

"""

import numpy as np


def convert_cart2curvilin(data, ro=8., vo=220.):
    """
    Converts cartesian coordinates XYZUVW (given with respect to the 
    LSR) to the curvilinear system.
    Curvilinear system is corotating so its radial component xi is 
    always pointing towards the galactic center.
    Coordinates in the curvilinear system are [xi, eta, zeta, xidot, etadot, zetadot].

    Parameters
    ----------
    data: [6, (npoints)] float np.array
        [pc, pc, pc, km/s,km/s,km/s] --> TODO: Change this to pc/Myr
        [X,  Y,  Z,  U,   V,   W]

    Returns
    -------
    curvilin_coord: [6, (npoints)] float np.array
        xi     : radial distance from the origin in LSR
        eta    :
        zeta   : vertical distance from plane
        xidot  :
        etadot :
        zetadot:

    """
    #~ data = np.array(data)

    X, Y, Z, U, V, W = data.T

    # TODO: to this conversion at the upper level
    R0 = ro*1000.0 # pc
    Omega0 = vo/R0 # km/s / pc # This could also be done earlier

    # Place the velocities in a rotating frame
    U = U - Y*Omega0
    V = V + X*Omega0

    R = np.sqrt(Y**2 + (R0-X)**2)
    phi = np.arctan2(Y, R0-X)

    xi = R0-R
    eta = phi*R0
    zeta = Z
    xidot = U*np.cos(phi) - V*np.sin(phi)
    etadot = R0/R * (V*np.cos(phi) + U*np.sin(phi))
    zetadot = W

    curvilin_coord = np.array([xi, eta, zeta, xidot, etadot, zetadot])

    return curvilin_coord.T


def convert_curvilin2cart(data, ro=8., vo=220.):
    """
    Converts curvilinear coordinates [xi, eta, zeta, xidot, etadot, 
    zetadot] to cartesian coordinates XYZUVW (given with respect to the 
    LSR).
    Curvilinear system is corotating so its radial component xi is 
    always pointing towards the galactic center.

    Parameters
    ----------
    data: [6, (npoints)] float np.array
        [pc, pc, pc, km/s,km/s,km/s] --> TODO: Change this to pc/Myr
        [xi, eta, zeta, xidot, etadot, zetadot]

    Returns
    -------
    cart_coordinates: [6, (npoints)] float np.array
        [pc, pc, pc, km/s,km/s,km/s] --> TODO: Change this to pc/Myr
        [X,  Y,  Z,  U,   V,   W]

    """

    xi, eta, zeta, xidot, etadot, zetadot = data.T

    # Todo: compute this at the upper level
    R0 = ro*1000.0

    R = R0 - xi
    phi = eta/R0

    X = xi*np.cos(phi) + R0*(1.0-np.cos(phi)) #R0 - R*np.cos(phi)
    Y = R*np.sin(phi)
    Z = zeta

    U = xidot*np.cos(phi) + R/R0*etadot*np.sin(phi)
    V = - xidot*np.sin(phi) + R/R0*etadot*np.cos(phi)
    W = zetadot

    # Convert to a non-rotating observed frame
    Omega0 = vo/R0 # km/s / pc
    U = U + Y*Omega0
    V = V - X*Omega0

    cart_coordinates = np.array([X, Y, Z, U, V, W])

    return cart_coordinates.T


def epicyclic_approx(data, times=None, sA=0.89, sB=1.15, sR=1.21):
    """
    Epicyclic approximation following the Makarov et al. 2004 paper
    in the curvilinear coordinate system:
    The radial component xi is pointing towards the Galactic center
    at all times and equals 0 at R0.
    The circular component eta circles around the Galaxy; eta = phi*R.
    The vertical component is defined as a displacement from the Galactic plane.

    This approximation works close to the LSR.

    Parameters
    ------------
    data : [pc, pc, pc, km/s, km/s, km/s] # *parsecs in the eta component are scales parsecs...
           xi, eta, zeta, xidot, etadot, zetadot
    """
    xi0, eta0, zeta0, xidot0, etadot0, zetadot0 = data.T

    # Bovy 2017
    #~ A0 = 15.3  # km/s/kpc
    #~ B0 = -11.9  # km/s/kpc

    # Unit conversion: convert from km/s/kpc to Myr-1
    #~ A = A0 * 0.0010227121650537077  # Myr-1
    #~ B = B0 * 0.0010227121650537077  # Myr-1
    
    # Bovy 2017, converted from km/s/kpc to Myr-1
    # TODO: Put this (both Oort's constants and the scaling factors) on the input params list. This is where all the conversions should be done, too.
    A0 = 0.01564749613 # 15.3 km/s/kpc * 0.0010227121650537077 = 0.01564749613 Myr-1
    B0 = -0.01217027476 # -11.9 km/s/kpc * 0.0010227121650537077 = -0.01217027476 Myr-1

    # Bovy 2017. Scale factors to match MW2014.
    # Mike's scale factors
    #A0 = 0.97*15.3 # km/s/kpc
    #B0 = 1.12*(-11.9) # km/s/kpc

    # Marusa's factors TODO: possibly do this at a higher level so this multiplication is not repeated every single time
    #~ A0 = A0*sA # km/s/kpc
    #~ B0 = B0*sB # km/s/kpc
    
    A = A0*sA # km/s/kpc
    B = B0*sB # km/s/kpc




    # Fine tuning rho. TODO: do this at a higher level so it is not repeated every single time
    rho_scale_factor = sR #1.36
    rho = rho_scale_factor * 0.0889  # M0/pc3
    Grho = rho * 0.004498502151575285  # Myr-2; rho should be given in M0/pc3
    kappa = np.sqrt(-4.0 * B * (A - B)) # Myr-1
    nu = np.sqrt(4.0 * np.pi * Grho + (A + B) * (A - B)) # Myr-1

    t=times

    kt=kappa*t
    nt=nu*t    

    # Propagate positions
    xi = xi0 + xidot0/kappa*np.sin(kt) + (etadot0 - 2.0*A*xi0) *\
        (1.0 - np.cos(kt)) / (2.0*B)
        
    eta = eta0 - xidot0 * (1.0 - np.cos(kt)) / (2.0*B) + etadot0 *\
        (A*kt - (A-B)*np.sin(kt)) /\
        (kappa*B) - xi0 * 2.0*A*(A-B)*(kt-np.sin(kt)) / (kappa*B)
    
    zeta = zeta0*np.cos(nt) + zetadot0/nu*np.sin(nt)

    # Propagate velocities
    xidot = xidot0*np.cos(kt) + (etadot0 - 2.0*A*xi0) *\
        kappa*np.sin(kt) / (2.0*B)
    
    etadot = -xidot0*kappa/(2.0*B)*np.sin(kt) +\
        etadot0/B*(A-(A-B)*np.cos(kt)) -\
        2.0*A*xi0*(A-B)*(1.0-np.cos(kt))/B
    
    zetadot = -zeta0*nu*np.sin(nt) + zetadot0*np.cos(nt)

    # Assemble the array with results
    new_position = np.array([xi, eta, zeta, xidot, etadot, zetadot])
    new_position = new_position.T
    
    return new_position


def trace_epicyclic_orbit(xyzuvw_start, times=None, sA=0.89, sB=1.15, 
    sR=1.21, ro=8., vo=220., single_age=True):
    """
    Given a star's XYZUVW relative to the LSR (at any time), project its
    orbit forward (or backward) to each of the times listed in *times*
    using epicyclic approximation. This only works close to the LSR.

    Positive times --> traceforward
    Negative times --> traceback

    Parameters
    ----------
    xyzuvw : [pc, pc, pc, pc/Myr, pc/Myr, pc/Myr]
    times : (float) or ([ntimes] float array)
        Myr - time of 0.0 must be present in the array. Times need not be #TODO: time 0.0 really? [TC: this was true for galpy]
        spread linearly.
    single_age: (bool) {True}
        Set this flag if only providing a single age to trace to
        This is there for the plotting purposes.

    sA, sB, sR: fine tuning factors for epicyclic approx. This works for
        Sco-Cen, but not sure about other regions. Also, these are
        determined for LSR with the wrong signs... TODO!!

    Returns
    -------
    xyzuvw_tf : [ntimes, 6] array
        [pc, pc, pc, km/s, km/s, km/s] - the traced orbit with positions
        and velocities
    """
    
    # unnecessary if sentence
    if single_age:
        # replace 0 with some tiny number
        try:
            if times == 0.:
                times = 1e-15
            # times = np.array([0., times])
        except ValueError as err:
            if not err.args:
                err.args = ('',)
            err.args = err.args + ('WARNING: comparing array to float? '
                                   'Did you leave single_age as True?',)
            raise

    else:
        raise UserWarning('Multi age orbit integation no longer supported')
        times = np.array(times)

    #~ times = np.array([0, age])

    # Make sure numbers are floats, and reshape into 2d
    xyzuvw_start = np.atleast_2d(xyzuvw_start.astype(np.float))


    # Units: Velocities are in km/s, convert into pc/Myr
    xyzuvw_start[:,3:] = xyzuvw_start[:,3:] * 1.0227121650537077 # pc/Myr
    #~ xyzuvw_start[3:] = xyzuvw_start[3:] * 1.0227121650537077 # pc/Myr

    # Transform to curvilinear
    curvilin = convert_cart2curvilin(xyzuvw_start, ro=ro, vo=vo)

    # Trace orbit with epicyclic approx.
    new_position = epicyclic_approx(curvilin, times=times, sA=sA, sB=sB,
        sR=sR)

    #~ print('new_position')
    #~ print(new_position)

    # Transform back to cartesian
    xyzuvw_new = convert_curvilin2cart(new_position, ro=ro, vo=vo)

    # Units: Transform velocities from pc/Myr back to km/s
    xyzuvw_new[:,3:] /= 1.0227121650537077
    #~ xyzuvw_new[3:] /= 1.0227121650537077


    #~ return xyzuvw_new
    # Remove empty dimensions
    return np.squeeze(xyzuvw_new)
    

def calc_jacobian_epicyclic(loc, dim=6, h=1e-3, age=None):
    """
    Calculate the Jacobian of the coordinate transfromation 
    `trans_func` about `loc`.

    `trans_func` should take a vector of dimension `dim` to a new vector
    of same dimension. This function then calculates the 2nd order 
    partial derivative at point `loc`. Extra arguments for `trans_func` 
    can be provided as a tuple to `args`.

    Parameters
    ----------
    trans_func : function
        Transformation function taking us from the initial coordinate 
        frame to the final coordinate frame
    loc : [dim] float array
        The position (in the initial coordinte frame) around which we 
        are calculating the jacobian
    dim : int {6}
        The dimensionality of the coordinate frames
    h : float {1e-3}
        The size of the increment, smaller values maybe run into 
        numerical issues
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

    #~ print('args', args)

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

    final_pos = trace_epicyclic_orbit(start_pos, times=age)

    print('temporal_propagation_marusa')
    for i in range(dim):
        jac[:,i] = (final_pos[2*i] - final_pos[2*i + 1]) / (2*h)
        print(final_pos[2*i], final_pos[2*i + 1])

#    for i in range(dim):
#        jac[:,i] = calc_jacobian_column(trans_func, i, loc, dim, h, args)

    return jac


def trace_epicyclic_covmatrix(cov, loc, dim=6, h=1e-3, age=None):
    """
    This is 'transform.transform_covmatrix' from Tim's Chronostar.
    

    Calculates covariance matrix of current day distribution.
    Transforms a covariance matrix from one coordinate frame to another.

    Calculated as a first-order Taylor approximation of the coordinate
    transformation that takes the initial mean to the current day mean.
    This is the most expensive aspect of Chronostar, so we first make
    sure the covariance matrix hasn't already been projected.



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
        The size of the increment, smaller values maybe run into 
        numerical issues
    args : tuple
        extra args to be passed to trans_func. E.g. for traceOrbitXYZUVW
        args = (age,) [for traceforward] or args = (-age,) 
        [for traceback]

    Returns
    -------
    conv_cov : [dim,dim] float array
        The transformed covariance matrix
    """
    jac = calc_jacobian_epicyclic(loc, age=age, dim=dim, h=h)
    print('jac P')
    print(jac)
    
    cov_transformed = np.dot(jac, np.dot(cov, jac.T))
    
    return cov_transformed


if __name__=='__main__':
    import pickle
    
    with open('../fastfit/data_for_testing/input_data_to_expectation.pkl', 'rb') as f:
        input_data = pickle.load(f)
    _, comps, _, _, _ = input_data
    comp = comps[0]

    # Component at time=0 and age
    mean0 = comp.get_mean()
    cov0 = comp.get_covmatrix()
    age = comp.get_age()
    
    #~ print('mean0')
    #~ print(mean0)
    
    # Tim's transformation
    mean_now, cov_now = comp.get_currentday_projection()

    # Testing transformation
    #~ mean_now_test = trace_epicyclic_orbit(mean0, times=age)
    
    #~ print('mean_now')
    #~ print(mean_now)
    
    #~ diff_mean = mean_now_test - mean_now
    #~ mask = np.abs(diff_mean)>1e-8
    #~ print('diff mean dimensions that differ',  np.sum(mask))
    
    #~ cov_now_test = trace_epicyclic_covmatrix(cov0, loc=mean0, age=age)

    #~ diff_cov = cov_now_test - cov_now
    #~ mask = np.abs(diff_mean)>1e-6
    #~ print('diff cov now',  np.sum(mask))
    
    #~ from chronostar import transform
    #~ from chronostar import traceorbit
    #~ cov_now1 = transform.transform_covmatrix(cov0, 
        #~ traceorbit.trace_epicyclic_orbit, mean0, args=(age,))

    #~ mean_now1 = traceorbit.trace_epicyclic_orbit(mean0, times=age)

    # TEST convert_cart2curvilin
    curvilin = convert_cart2curvilin(mean0)
    #~ print('curvilin')
    #~ print(curvilin)
    
    new_position = epicyclic_approx(curvilin, times=age)
    #~ print('new_position')
    #~ print(new_position)
