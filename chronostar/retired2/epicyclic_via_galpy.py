import numpy as np

from chronostar import traceorbit as torb

# Base units for orbits: pc, Myr,

# Set up some ugly globals. Can tidy later.

kms_to_pcMyr = 1.0227121650537077 # (pc/Myr) / (km/s)

V0 = 220. * kms_to_pcMyr # pc/Myr
R0 = 8000. # pc
omega = V0/R0

# Since A-B is defined to be omega, and omega is defined by our initialisation
# of galpy, why not determine B0 implicitly? That way, only one term need
# be tweaked
A0 = 0.99*15.3 # km/s/kpc
# B0 = 1.04*(-11.9) # km/s/kpc

A = A0 / 1000 * kms_to_pcMyr # Myr-1
# B = B0 / 1000 * kms_to_pcMyr # Myr-1
B = A - omega # Myr-1

P = np.sqrt( (A-B) / -B)  # (dimensionless)
kappa = np.sqrt(-4.0*B*(A-B)) # Myr-1

rho = 1.15*0.0889 # M0/pc3
rho = 1.15*0.0889 # M0/pc3
Grho = rho * 0.004498502151575285 # Myr-2; rho should be given in M0/pc3
nu = np.sqrt(4.0*np.pi * Grho + (A+B)*(A-B))

# d(circular velcoity)/dr at R0. Can use psi to get circular velocity
# at different radii near R0.
psi = A + B

def quadratic_formula(a, b, c):
    """
    Implemented my own quadratic formula. As I do not know the complexity of
    np.root.
    """
    pos = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    neg = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)
    return np.array([pos, neg])


def get_guiding_radius(xyzuvw=None, galpy_coords=None):
    """
    Returns radius of guiding centre, in pc
    Accepts as input either Chronostar xyzuvw, with
    units pc and km/s
    or galpy [R, vR, vT, Z, vZ, phi]

    Exploits conservation of angular momentum, and `psi` to construct and then
    solve a quadratic equation for R_g. See latex document for derivation.
    """

    if galpy_coords is None:
        galpy_coords = torb.convert_cart2galpycoords(xyzuvw)

    R, _, V_T, _, _, _ = galpy_coords

    # Scale up from galpy dimensionless values
    R *= R0
    V_T *= V0

    psi = A + B

    # Calculate the coefficients from ax^2 + bx + c, where x is R_L
    # See section 1.1 in latex file for derivation
    a_coeff = psi # myr-1
    b_coeff = -(V0 + psi*R0) # pc/myr
    c_coeff = R*V_T # pc^2/myr

    results = quadratic_formula(a_coeff, b_coeff, c_coeff)

    # Return the closest value to initial radius, because the other value
    # is irrelevant
    diff = np.abs(results - R)
    return results[np.argmin(diff)]


def convert_galpy2epi(galpy_coords):
    """
    Convert galpy coordinates to epicyclic coordinates 
    [R,vR,vT,Z,vZ,phi] --> [R_g,T_g,b,phi_b,Z_g,phi_z] 

    Parameters
    ----------
    galpy_coords: [6] float np.array
        Phase-space description. All distances should be normalised such that
        1 = 8000 pc, and all velocities should be normalised such that 1 = 220 km/s

        R   : radius of point
        V_R : radial velocity of point (+ve away from GC)
        V_T : tangential velocity of point (+ve in direction of rotation)
        Z   : vertical height above plane
        V_Z : vertical velocity
        phi : azimuthal position

    Returns
    -------
    epi_coords: [6] float np.array
        R_g   : radius of guiding centre
        T_g   : equal to phi*R_g, where phi is azimuthal position of
                guiding centre
        b     : amplitude of ocillation in R or x
        phi_b : angle point makes in xy ellipse and uv circle, where xyuv
                are oscillations about the guiding centre
        Z_g   : max height in Z
        phi_z : angle the point makes in zw ellipse
    """
    R, V_R, V_T, Z, V_Z, phi = galpy_coords

    # Scale up from dimensionless (implicitly incorporating our unit convention)
    R   *= R0
    Z   *= R0
    V_R *= V0
    V_T *= V0
    V_Z *= V0

    # Calculate radius of guiding centre by exploiting conservation of ang. momentum
    R_g = get_guiding_radius(galpy_coords = galpy_coords)

    # Get velocity of guiding centre by exploiting Oort constants 
    # (change of V_circ w.r.t R)
    V_g = V0 + psi*(R0 - R_g)
    
    # Calculate offsets from guiding centre
    # Note, y is tricky to establish as we don't yet know T_g, ~ guiding centres azmithual 
    # position
    x = R_g - R
    z = Z
    u = -V_R
    v = V_T - V_g
    w = V_Z
    
    # Calculate phi_b from ratio of u and v
    # opp = U_FAC*(-u).to('km/s').value
    # opp = (-u/P).to('km/s').value
    # adj =   v.to('km/s').value
    phi_b = np.arctan2(-u, P*v)

    # Calculate b
    # looks messy because we need to avoid division by zero,
    bs = []
    if abs(np.cos(phi_b)) > 1e-5:
        temp_b = v/(omega*np.cos(phi_b))
        # temp_b = v/(-2*B*np.cos(phi_b)) # Fuchs 2006
        bs.append(temp_b)

        temp_b = x/np.cos(phi_b)
        bs.append(temp_b)

    if abs(np.sin(phi_b)) > 1e-5:
        temp_b = -u/(omega*P*np.sin(phi_b))
        # temp_b = u/(-kappa*np.sin(phi_b)) # Fuchs 2006
        bs.append(temp_b)
    b = np.mean(bs)

    # Calculate azimuthal position of guiding centre,
    # but in terms of distance along perimeter. (Like Y in curvilinear coords)
    # We restrict angle to [-pi,pi] so we can plot things nicely later
    phi = restrict_angle_range(phi)
    T_g = phi*R_g - P*b*np.sin(phi_b)

    # Calculate phi_z from ratio of w and z
    # opp = -w.to('km/s').value
    # adj = (z*gr.nu).to('km/s').value
    phi_z = np.arctan2(-w,z*nu)

    # Calculate Z_g
    # looks messy because we need to avoid division by zero,
    Z_gs = []
    if abs(np.cos(phi_z)) > 1e-5:
        temp_Z_g = z/np.cos(phi_z)
        Z_gs.append(temp_Z_g)
    if abs(np.sin(phi_z)) > 1e-5:
        temp_Z_g = -w/(nu*np.sin(phi_z))
        Z_gs.append(temp_Z_g)
    Z_g = np.mean(Z_gs)
    
    return np.array([R_g, T_g, b, phi_b, Z_g, phi_z])


def convert_epi2galpy(epi_coords):
    """
    Convert epicyclic coordinates to galpy coordinates
    [R_g,T_g,b,phi_b,Z_g,phi_z] --> [R,vR,vT,Z,vZ,phi]

    Parameters
    ----------
    epi_coords: [6] float np.array
        R_g   : radius of guiding centre
        T_g   : equal to phi*R_g, where phi is azimuthal position of
                guiding centre
        b     : amplitude of ocillation in R or x
        phi_b : angle point makes in xy ellipse and uv circle, where xyuv
                are oscillations about the guiding centre
        Z_g   : max height in Z
        phi_z : angle the point makes in zw ellipse

    Returns
    -------
    galpy_coords: [6] float np.array
        R/R0   : radius of point
        V_R/V0 : radial velocity of point (+ve away from GC)
        V_T/V0 : tangential velocity of point (+ve in direction of rotation)
        Z/R0   : vertical height above plane
        V_Z/V0 : vertical velocity
        phi    : azimuthal position

        where R0 = 8000pc and V0 = 200 km/s and are the radial position and
        rotational velocity of the LSR
    """
    R_g, T_g, b, phi_b, Z_g, phi_z = epi_coords

    # Calculate circular velocity at guiding radius
    V_g = V0 + psi*(R0 - R_g)

    # Calculate offsets from guiding centre
    x =     b*np.cos(phi_b)
    y = P * b*np.sin(phi_b)
    u = - omega * P * b*np.sin(phi_b)
    # u = -kappa*b*np.sin(phi_b)        # Fuchs 2006
    v = omega * b*np.cos(phi_b)
    # v =  -2*B*b*np.cos(phi_b)         # Fuchs 2006
    z =          Z_g*np.cos(phi_z)
    w = -nu * Z_g*np.sin(phi_z)
    
    # Calculate galpy coordinates from offsets
    R = R_g - x
    V_R = -u
    V_T = V_g + v
    Z = z
    V_Z = w
    phi = (T_g + y)/R_g

    # Restrict angle to [-pi,pi] for simpler plotting later on
    phi = restrict_angle_range(phi)

    # Scale to dimensionless values and remove astropy units
    return np.array([(R/R0), (V_R/V0), (V_T/V0), (Z/R0), (V_Z/V0), phi])


def evolve_epi(epi_start, time):
    """
    Assume time is given in Myr

    Will return an [ntime, 6] array, even if only one time is given.
    """
    # t = time
    try:
        npoints = len(time)
    except TypeError:
        npoints = 1
    results = np.zeros((npoints, 6))

    R_g, T_g, b, phi_b, Z_g, phi_z = epi_start

    # Evolve azimuthal position of guiding centre
    V_g = V0 + psi * (R0 - R_g)

    T_g = T_g + V_g*time

    # Evolve angles
    phi_b = phi_b + kappa*time
    phi_z = phi_z + nu*time

    # Mod angles to be in range [-pi,pi]
    phi_b = restrict_angle_range(phi_b)
    phi_z = restrict_angle_range(phi_z)

    # Collect result, and convert to output units
    # NOTE: there is undoubtedly a simpler way to implicitly tile the constant
    # scalars, with the [len(time)] arrays of the angles...
    results[:,0] = R_g
    results[:,1] = T_g
    results[:,2] = b
    results[:,3] = phi_b
    results[:,4] = Z_g
    results[:,5] = phi_z

    return results

def restrict_angle_range(phis):
    # Put phi in range [-pi, pi]
    phis = np.mod(phis, 2*np.pi)
    try:
        phis[np.where(phis > np.pi)] -= 2*np.pi
    except:
        if phis > np.pi:
            phis -= 2*np.pi
    return phis

