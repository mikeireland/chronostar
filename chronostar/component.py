"""
Class object that encapsulates a component, the phase-space model
of an unbound set of stars formed from the same starburst/filament.

A component models the initial phase-space distribution of stars
as a Gaussian. As such there are three key attributes:
- mean: the central location
- covariance matrix: the spread in each dimension along with any correlations
- age: how long the stars have been travelling

Method decorators are used for two methods of AbstractComponent.

load_components() is a "static" method (and hence has a @staticmethod
decorator). This means that it can be called directly from the Class, e.g.:
my_comps = SphereComponent.load_components('filename')
In practical terms, this method has no 'self' in the signature and thus
cannot access any attributes that would otherwise be accessible by e.g.
self.blah

get_sensible_walker_spread() is a "class" method (and hence has a
@classmethod decorator). This means that it can be called directly from
the Class, e.g.:
sensible_spread = SphereComponent.get_sensible_walker_spread()
This is similar to a static method, but needs access to the class
attribute SENSIBLE_WALKER_SPREADS.
In practical terms, instead of the first argument of the method's
signature being 'self', it is 'cls', meaning the method has access to
the class attributes.

It doesn't make sense for these methods to be used by instantiated objects
of the class, but they are still very closely tied to the Class. They could
be left in the global namespace of this module, however then two separate
imports would be required throughout Chronostar, and it would complicate
the process of plugging in a different, modularised Component class.
"""

from __future__ import print_function, division, unicode_literals

try:
    import matplotlib as mpl
    mpl.use('Agg') # stops auto displaying plots upon generation
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
except:
    ImportError

import numpy as np
from scipy.stats.mstats import gmean

from . import transform
from .traceorbit import trace_cartesian_orbit

# Including plotting capabilities

class AbstractComponent(object):
    """
    An abstract class that (when implmented) encapsulates a component,
    the phase-space model of an unbound set of stars formed from the
    same starburst/filament.

    A component models the initial phase-space distribution of stars
    as a Gaussian. As such there are three key attributes:
    - mean: the central location
    - covariance matrix: the spread in each dimension along with any
    correlations
    - age: how long the stars have been travelling

    This class has been left abstract so as to easily facilitate
    (and encourage) alternate parameterisations of components.

    In brief, just copy-paste SphereComponent below to make your new
    class, and modify the methods and attribute to suit your new
    parametrisation.

    In order to implement this class and make a concrete class, only
    one variable must be set, and four methods implmented. In short,
    the class must be told how to turn raw parameters into attributes,
    and vice verse.

    Attributes to set:
    `PARAMETER_FORMAT`
        This parameter must be set. An ordered list of labels
        describing what purpose each input serves. e.g. for a
        SphereComponent, the list is
        3*['pos'] + 3*['vel'] + ['log_pos_std', 'log_vel_std', 'age']
        See `SENSIBLE_WALKER_SPREADS` for a set of viable labels, and
        include your own as needed! Note that this is the parameters in
        "internal form", i.e. the form of the parameter space that
        emcee explores.

    Methods to define
    internalise(pars) and externalise(pars)
        You must tell the Component class how to switch between internal
        and external formats. These methods are static because there is
        not always a need to instantiate an entire Component object
        simply to convert between parameter forms.

        There is perhaps scope to have the Component class to intuit
        how to convert between forms based on `PARAMETER_FORMAT` values.
    _set_covmatrix(covmatrix=None), (REQUIRED)
    _set_mean(mean=None), _set_age(age=None) (both optional)
        These methods instruct the class how to (if input is None) build
        the attribute from self.pars, or (if input is provided) to set
        the self._mean (for e.g.) attribute but also to reverse engineer
        the self.pars values and update accordingly.

        These methods should only be called internally, (from the
        __init__() method, or the update_attributes() method) as it is
        critical to do some tidying up (setting mean_now and
        covmatrix_now to None) whenever self.pars is modified.

        If you stick to the convention of the mean=pars[:6] and
        age=pars[-1] then the default methods will suffice and you will
        only need to implement _set_covmatrix(). Of course if you wish,
        you can override _set_mean() or _set_age().
    """
    # __metaclass__ = ABCMeta

    DEFAULT_TINY_AGE = 1e-10

    _pars = None
    _mean = None
    _covmatrix = None
    _age = None
    _sphere_dx = None
    _sphere_dv = None

    _mean_now = None
    _covmatrix_now = None

    # Set these in concrete class, matching form with 'SENSIBLE_WALKER_SPREADS'
    # See SphereComponent and EllipComponent for examples
    PARAMETER_FORMAT = None

    # This is used to guide the scale of variations in each parameter
    # Super useful when initialising emcee walkers across a sensible
    # volume of parameter space.
    SENSIBLE_WALKER_SPREADS = {
        'pos':10.,
        'pos_std':1.,
        'log_pos_std':0.5,
        'vel':2.,
        'vel_std':1.,
        'log_vel_std':0.5,
        'corr':0.05,
        'age':1.,
        'angle_rad':0.25*np.pi,
        'angle_deg':45.,
    }

    def __init__(self, pars=None, emcee_pars=None, attributes=None,
                 trace_orbit_func=None):
        """
        An abstraction for the parametrisation of a moving group
        component origin. As a 6D Gaussian, a Component has three key
        attributes; mean, covariance matrix, and age. There are many
        ways to parameterise a covariance matrix to various degrees
        of freedom.

        Parameters
        ----------
        pars: 1D float array_like
            Raw values for the parameters of the component. Parameters
            should be provided in real space (as opposed to any
            modifications made for emcee's sake). In simple terms,
            if you are initialising a component based on parameters
            that have real units, then use this argument.
        emcee_pars: 1D float array_like
            Raw values for the parameters of the component but in
            converted style used by emcee (e.g. log space for standard
            deviations etc). In simple terms, if you are initialising
            a component based on parameters taken from an emcee chain,
            then use this argument.
        attributes: dict with all the following keys:
            mean: [6] float array_like
                The mean of the initial Gaussian distribution in
                cartesian space:
                [X(pc), Y(pc), Z(pc), U(km/s), V(km/s), W(km/s)]
            covmatrix: [6,6] float array_like
                the covariance matrix of the initial Gaussian
                distribution, with same units as `mean`
            age: float
                the age of the component (positive) in millions of
                years
        trace_orbit_func: function {trace_cartesian_orbit}
            Function used to calculate an orbit through cartesian space
            (centred on, and co-rotating with, the local standard of
            rest). Function must be able to take two parameters, the
            starting location and the age, with positive age
            corresponding to forward evolution, and negative age
            backward evolution.

            Signature should be:
            def func(xyzuvw_start, times)

            where xyzuvw_start is a 6 element list or array corresponding
            to [X,Y,Z,U,V,W]
            and `times` : is a single float value

            It should also be "odd", i.e.:

            >> func(loc_then, +age) = loc_now
            >> func(loc_now,  -age) = loc_then

            where loc_then

            odd: -f(x) = f(-x)
            basically: tracing a point forward by age, then back by age
            should get to the same place

        Returns
        -------
        res: Component object
            An astraction of a set of values parametrising the origin of
            a moving group component.
        """
        # Some basic implementation checks
        self.check_parameter_format()


        # Set cartesian orbit tracing function
        if trace_orbit_func is None:
            self.trace_orbit_func = trace_cartesian_orbit
        else:
            self.trace_orbit_func = trace_orbit_func

        # If parameters are provided in internal form (the form used by emcee),
        # then externalise before setting of various other attributes.
        if pars is not None and emcee_pars is not None:
            raise UserWarning('Should only initialise with either `pars` or '
                              '`emcee_pars` but not both.')

        # Check length of parameter input (if provided) matches implementation
        if pars is not None or emcee_pars is not None:
            par_length = len(pars) if pars is not None else len(emcee_pars)
            if par_length != len(self.PARAMETER_FORMAT):
                raise UserWarning('Parameter length does not match '
                                  'implementation of {}. Are you using the '
                                  'correct Component class?'.\
                                  format(self.__class__))


        # If initialising with parameters in 'emcee' parameter space, then
        # convert to 'real' parameter space before constructing attributes.
        if emcee_pars is not None:
            pars = self.externalise(emcee_pars)

        # Set _pars, setting to all zeroes if no pars input is provided.
        if pars is not None:
            self._pars = np.copy(pars)
        else:
            self._pars = np.zeros(len(self.PARAMETER_FORMAT))

            # Age *must* be non-zero
            self._set_age(self.DEFAULT_TINY_AGE)

        # Using provided parameters, set up the three model attributes:
        # mean, covariance and age. If attributes are provided, then use
        # those.
        if attributes is None:
            attributes = {}
        self._set_mean(attributes.get('mean', None))
        self._set_covmatrix(attributes.get('covmatrix', None))
        self._set_age(attributes.get('age', None))

        # For some purposes (e.g. virialisation estimation) it is useful to
        # approximate position and velocity volumes as spherical. Calculate
        # and set those attributes.
        self.set_sphere_stds()

    def __str__(self):
        x,y,z,u,v,w = self.get_mean_now()
        return 'Currentday(' \
               'X: {:5.1f}pc, Y: {:5.1f}pc, Z: {:5.1f}pc, '  \
               'U {:4.1f}km/s, V {:4.1f}km/s, W {:4.1f}km/s, ' \
               'age: {:4.1f}Myr)'.format(x,y,z,u,v,w, self._age)

    def __repr__(self):
        return self.__str__()

    def check_parameter_format(self):
        """
        A check for valid implementation. If this throws an error then
        the PARAMETER_FORMAT attribute has been incorrectly defined.
        """
        if self.PARAMETER_FORMAT is None:
            raise NotImplementedError('Need to define PARAMETER_FORMAT '
                                      'as a class parameter')
        if not np.all(np.isin(self.PARAMETER_FORMAT,
                              list(self.SENSIBLE_WALKER_SPREADS.keys()))):
            raise NotImplementedError(
                    'Label in PARAMETER_FORMAT doesn\'t seem to be in '
                    'SENSIBLE_WALKER_SPREADS. Extend dictionary in '
                    'AbstractComponent accordingly: {}'.format(
                            self.PARAMETER_FORMAT
                    )
            )

    @staticmethod
    def externalise(pars):
        """
        Take parameter set in internal form (as used by emcee) and
        convert to external form (as used to build attributes).

        Notes
        -----
        This is a "staticmethod" because often there is a desire to convert
        between parameterisations without actually having a component
        object.

        Tasks
        -----
        There is scope to implement this here, and use cls.PARAMETER_FORMAT
        to guide the parameter conversions
        """
        raise NotImplementedError

    @staticmethod
    def internalise(self, pars):
        """
        Take parameter set in external form (as used to build attributes)
        and convert to internal form (as used by emcee).

        Notes
        -----
        This is a "staticmethod" because often there is a desire to convert
        between parameterisations without actually having a component
        object.

        Tasks
        -----
        There is scope to implement this here, and use cls.PARAMETER_FORMAT
        to guide the parameter conversions
        """
        raise NotImplementedError

    def get_pars(self):
        """
        Return a copy of the raw (external) parameterisation of
        the Component
        """
        return np.copy(self._pars)

    def get_emcee_pars(self):
        """
        Return a copy of the 'emcee' space parameterisation of the
        Component
        """
        return self.internalise(self._pars)

    def _set_mean(self, mean=None):
        """
        Builds mean from self.pars. If setting from an externally
        provided mean then updates self.pars for consistency

        If implementation does use the first 6 values in self._pars
        to set the mean then this method should be overridden.
        """
        # If mean hasn't been provided, generate from self._pars
        # and set.
        if mean is None:
            self._mean = self._pars[:6]
        # If mean has been provided, reverse engineer and update
        # self._pars accordingly.
        else:
            self._mean = np.copy(mean)
            self._pars[:6] = self._mean

    def get_mean(self):
        """Return a copy of the mean (initial) of the component"""
        return np.copy(self._mean)

    def _set_covmatrix(self, covmatrix=None):
        """
        Builds covmatrix from self._pars. If setting from an externally
        provided covmatrix then update self._pars for consistency.

        This is the sole method that needs implmentation to build a
        usable Component class
        """
        raise NotImplementedError

    def get_covmatrix(self):
        """Return a copy of the covariance matrix (initial)"""
        return np.copy(self._covmatrix)

    def _set_age(self, age=None):
        """Builds age from self.pars. If setting from an externally
        provided age then updates self.pars for consistency"""
        if age is None:
            self._age = self._pars[-1]
        else:
            self._age = age
            self._pars[-1] = age

    def get_age(self):
        """Returns the age of the Component"""
        return self._age

    def get_attributes(self):
        """
        Get a dictionary of all three key attributes of the Component
        model. Done this way for easy of initialising a new Component.
        """
        return {'mean':self.get_mean(),
                'covmatrix':self.get_covmatrix(),
                'age':self.get_age()}

    def set_sphere_stds(self):
        """
        Set the spherical standard deviations in position space and
        velocity space. Calculated in such a way so as to preserved
        volume in position space and velocity space retrospectively.
        Note that combined phase-space volume is not conserved by this
        implementation.
        """
        self._sphere_dx = gmean(np.sqrt(
                np.linalg.eigvalsh(self._covmatrix[:3, :3]))
        )
        self._sphere_dv = gmean(np.sqrt(
                np.linalg.eigvalsh(self._covmatrix[3:, 3:]))
        )

    def get_sphere_dx(self):
        """
        Return the spherical standard deviation in position space.

        First check if it is None (which may be the case if covmatrix
        has been updated for e.g.) and recalculate at need.
        """
        if self._sphere_dx is None:
            self.set_sphere_stds()
        return self._sphere_dx

    def get_sphere_dv(self):
        """
        Return the spherical standard deviation in velocity space.

        First check if it is None (which may be the case if covmatrix
        has been updated for e.g.) and recalculate at need.
        """
        if self._sphere_dv is None:
            self.set_sphere_stds()
        return self._sphere_dv

    def update_attribute(self, attributes=None):
        """
        Update attributes based on input dictionary.

        Parameters
        ----------
        attributes: dict
            A dictionary with the any combination (including none) of the
            following:
            'mean': [6] float array_like
                the mean of the initial 6D Gaussian
            'covmatrix': [6,6] float array_like
                the covariance matrix of the initial 6D Gaussian
            'age': float
                the age of the component

        Notes
        -----
        A potential source of subtle bugs is that one can modify attributes
        (e.g. mean) but if `covmatrix_now` has already been calculated, it
        won't update. So it is critical to use only this method to modify
        attributes such that we can force the recalculation of current-day
        projections as required.
        """
        if type(attributes) is not dict:
            raise TypeError('Attributes must be passed in as dictionary')
        if 'mean' in attributes.keys():
            self._set_mean(mean=attributes['mean'])
        if 'covmatrix' in attributes.keys():
            self._set_covmatrix(covmatrix=attributes['covmatrix'])
        if 'age' in attributes.keys():
            self._set_age(age=attributes['age'])

        # Reset any attribute that may have already been calculated
        self._mean_now = None
        self._covmatrix_now = None
        self._sphere_dx = None
        self._sphere_dv = None

    def get_mean_now(self):
        """
        Calculates the mean of the component when projected to the current-day
        """
        if self._mean_now is None:
            self._mean_now =\
                self.trace_orbit_func(self._mean, times=self._age)
        return self._mean_now

    def get_covmatrix_now(self):
        """
        Calculates covariance matrix of current day distribution.

        Calculated as a first-order Taylor approximation of the coordinate
        transformation that takes the initial mean to the current day mean.
        This is the most expensive aspect of Chronostar, so we first make
        sure the covariance matrix hasn't already been projected.
        """
        if self._covmatrix_now is None:
            self._covmatrix_now = transform.transform_covmatrix(
                    self._covmatrix, trans_func=self.trace_orbit_func,
                    loc=self._mean, args=(self._age,),
            )
        return self._covmatrix_now

    def get_currentday_projection(self):
        """
        Calculate (as needed) and return the current day projection of Component

        Returns
        -------
        mean_now : [6] float array_like
            The phase-space centroid of current-day Gaussian distribution of
            Component
        covmatrix_now : [6,6] float array_like
            The phase-space covariance matrix of current-day Gaussian
            distribution of Component
        """
        return self.get_mean_now(), self.get_covmatrix_now()

    def splitGroup(self, lo_age, hi_age):
        """
        Generate two new components that share the current day mean, and
        initial covariance matrix of this component but with different ages:
        `lo_age` and `hi_age`.

        Parameters
        ----------
        lo_age : float
            Must be a positive (and ideally smaller) value than self.age.
            Serves as the age for the younger component.
        hi_age : float
            Must be a positive (and ideally larger) value than self.age
            Serves as the age for the older component.

        Returns
        -------
        lo_comp : Component
            A component that matches `self` in current-day mean and initial
            covariance matrix but with a younger age
        hi_comp : Component
            A component that matches `self` in current-day mean and initial
            covariance matrix but with an older age
        """
        comps = []
        for new_age in [lo_age, hi_age]:
            # Give new component identical initial covmatrix, and a initial
            # mean chosen to yield identical mean_now
            new_mean = self.trace_orbit_func(self.get_mean_now(),
                                             times=-new_age)
            new_comp = self.__class__(attributes={'mean':new_mean,
                                                  'covmatrix':self._covmatrix,
                                                  'age':new_age})
            comps.append(new_comp)

        return comps

    def get_peak(self, amplitude=1.):
        """
        Get the density at the peak of distribution.

        Use this as a proxy of the characteristic density of the distribution,
        with the option to scale by the amplitude of the Gaussian. Note, the
        height of the peak is only dependent on the covariance matrix.

        Notes
        -----
        since we are evaluating the distribution at the mean, the exponent
        reduces to 0, and so we are left with only the coefficient of the
        multi-variate Gaussian formula
        """
        det = np.linalg.det(self.get_covmatrix_now())
        coeff = 1./np.sqrt((2*np.pi)**6 * det)
        return amplitude * coeff

    @staticmethod
    def load_components(filename):
        """
        Load Component objects from a *.npy file.

        Used to standardise result if loading a single component vs multiple
        components.

        Parameters
        ----------
        filename : str
            name of the stored file

        Returns
        -------
        res : [Component] list
            A list of Component objects
        """
        res = np.load(filename)
        if res.shape == ():
            return np.array([res.item()])
        else:
            return res

    @classmethod
    def load_raw_components(cls, filename, use_emcee_pars=False):
        """
        Load parameters from a *.npy file and build Component objects

        Parameters
        ----------
        filename: str
            Name of file from which data is loaded
        use_emcee_pars: bool {False}
            Set to true if stored data is parameters in emcee parametrisation

        Returns
        -------
        comps: [Component] list
            A list of Component objects

        Notes
        -----
        This is a class method (as opposed to static or normal method) because
        this method needs access to information on which implementation to use
        to convert parameters into objects. One *could* rewrite this function
        to accept the component class as input, but this would be ugly and look
        like:
        SphereComponent.load_raw_components(SphereComponent, filename)
        as opposed to
        SphereComponent.load_raw_components(filename)
        """
        pars_array = cls.load_components(filename)

        comps = []
        for pars in pars_array:
            if use_emcee_pars:
                comps.append(cls(emcee_pars=pars))
            else:
                comps.append(cls(pars=pars))
        return comps

    def store_raw(self, filename, use_emcee_pars=False):
        """
        Helper method that utilises the static method
        """
        self.store_raw_components(filename=filename, components=[self],
                                  use_emcee_pars=use_emcee_pars)

    @staticmethod
    def store_raw_components(filename, components, use_emcee_pars=False):
        """
        Store components as an array of raw parameters, in either
        real space (external) or emcee parameters space (internal)

        Parameters
        ----------
        filename: str
            The name of the file to which we are saving parameter data
        components: [Component] list
            The list of components that we are saving
        use_emcee_pars: bool {False}
            Set to true to store parameters in emcee parametrisation form

        Returns
        -------
        None

        Notes
        -----
        This is a static method because it needs as input a list of
        components, not just the component itself
        """
        if type(components) is not list:
            components = [components]
        if use_emcee_pars:
            pars = np.array([c.get_emcee_pars() for c in components])
        else:
            pars = np.array([c.get_pars() for c in components])
        np.save(filename, pars)

    def store_attributes(self, filename):
        """
        Store the attributes (mean, covmatrix and age) of single component

        Parameters
        ----------
        filename: str
            The name of the file to which we are saving attributes
        """
        attributes = {'mean':self.get_mean(),
                      'covmatrix':self.get_covmatrix(),
                      'age':self.get_age()}
        np.save(filename, attributes)

    @classmethod
    def get_best_from_chain(cls, chain_file, lnprob_file):
        """
        Little helper method that constructs a Component object
        representing the best sample from output of an emcee run

        Parameters
        ----------
        chain_file: str (or array)
            A numpy stored array of the chain output with dims
            [nwalkers, nsteps, npars]
        lnprob_file: str (or array)
            A numpy stored array of the lnprob output with dims
            [nwalkers, nsteps]

        Returns
        -------
        Component object
            A component corresponding to the sample with the highest
            lnprob from provided samples
        """
        if type(chain_file) is str:
            chain_file = np.load(chain_file)
        if type(lnprob_file) is str:
            lnprob_file = np.load(lnprob_file)
        best_ix = np.argmax(lnprob_file)
        npars = len(cls.PARAMETER_FORMAT)
        best_pars = chain_file.reshape(-1, npars)[best_ix]
        return cls(emcee_pars=best_pars)

    def get_alpha(self, nstars):
        """
        Calculate the virial-ness of the component by plugging spread in
        poisiton and velocity into simple virial equation.

        Parameters
        ----------
        nstars: float
            The expected number of stellar members

        Returns
        -------
        alpha: float
            the simplified virial value where < 1 implies a bound initial
            state

        Notes
        -----
            Assuming we have identified 100% of star mass, and that average
            star mass is 1 M_sun.
        """
        G_const = 0.004300917270069976  # pc (km/s)^2 / Msun
        M_sol = 1.  # Msun
        dx = self.get_sphere_dx()
        dv = self.get_sphere_dv()
        return (dv ** 2 * dx) / (G_const * nstars * M_sol)

    # --------------------------------------------------
    # --  PLOTTING METHODS  ----------------------------
    # --------------------------------------------------
    def add_arrow(self, line, position=None, indices=None, direction='right',
                  size=15, color=None, **kwargs):
        """
        Add an arrow along a plotted line.

        Parameters
        ----------
        line:       Line2D object
        position:   x-position of the arrow. If None, mean of xdata is taken
        direction:  'left' or 'right'
        size:       size of the arrow in fontsize points
        color:      if None, line color is taken.

        -- credit to some forgotten contributor to stackoverflow --
        https://stackoverflow.com/questions/34017866/arrow-on-a-line-plot
        -with-matplotlib
        thomas - https://stackoverflow.com/users/5543796/thomas
        """
        if color is None:
            color = line.get_color()

        xdata = line.get_xdata()
        ydata = line.get_ydata()

        if indices is None:
            if position is None:
                position = xdata.mean()
            # find closest index
            indices = [np.argmin(np.absolute(xdata - position))]

        for start_ind in indices:
            end_ind = start_ind + 1 if direction == 'right' else start_ind - 1

            line.axes.annotate('',
                               xytext=(xdata[start_ind], ydata[start_ind]),
                               xy=(xdata[end_ind], ydata[end_ind]),
                               arrowprops=dict(arrowstyle="->", color=color, **kwargs),
                               size=size,
                               )

    def plot_orbit(self, dim1, dim2, ax, ntimes=50,
                  with_arrow=False, annotate=False, color=None, alpha=0.3,
                   **kwargs):
        """
        For traceback use negative age

        Parameters
        ----------
        pos_now: [6] array, known position of object
        dim1: integer, x-axis dimension
        dim2: integer, y-axis dimension
        ax: axes object, axes on which to plot line
        end_age: non-zero number, time to orbit till.
            Negative value --> traceback
            Positive value --> trace forward
        ntimes: integer {50], number of timesteps to calculate
        group_ix: index of group being plotted (for coloring reasons)
        with_arrow: (bool) {False}, whether to include arrows along orbit
        annotate: (bool) {False}, whether to include text
        """
        # alpha=0.3
        if color is None:
            color = 'black'
            # if group_ix is None:
            #     color = COLORS[0]
            # else:
            #     color = COLORS[group_ix]

        # orb_alpha = 0.1
        comp_orb = trace_cartesian_orbit(
                self.get_mean(),
                times=np.linspace(0, self.get_age(), ntimes),
                single_age=False
        )
        line_obj = ax.plot(comp_orb[:, dim1], comp_orb[:, dim2], ls='-',
                           alpha=alpha,
                           color=color, zorder=1, **kwargs)
        indices = [int(ntimes / 3), int(2 * ntimes / 3)]
        if with_arrow:
            # make sure arrow is always pointing forwards through time
            direction = 'right' if self.get_age() > 0 else 'left'
            self.add_arrow(line_obj[0], indices=indices, direction=direction,
                           color=color, alpha=alpha, zorder=1)
        if annotate:
            ax.annotate("Orbital trajectory",
                        (comp_orb[int(ntimes / 2), dim1],
                         comp_orb[int(ntimes / 2), dim2]),
                        color=color)

    def plot_cov_ellipse(self, cov, pos, nstd=2, ax=None, with_line=True,
                         zorder=4, **kwargs):
        """
        Plots an `nstd` sigma error ellipse based on the specified covariance
        matrix (`cov`). Additional keyword arguments are passed on to the
        ellipse patch artist.
        Parameters
        ----------
            cov : The 2x2 covariance matrix to base the ellipse on
            pos : The location of the center of the ellipse. Expects a 2-element
                sequence of [x0, y0].
            nstd : The radius of the ellipse in numbers of standard deviations.
                Defaults to 2 standard deviations.
            ax : The axis that the ellipse will be plotted on. Defaults to the
                current axis.
            zorder: Integer
                The priority of overlay. The higher the number, the more 'ontop
                of' the plot the ellipse will be.
            Additional keyword arguments are pass on to the ellipse patch.
        Returns
        -------
            A matplotlib ellipse artist
        """

        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:, order]

        if ax is None:
            ax = plt.gca()

        # largest eigenvalue is first
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(vals)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

        if 'alpha' not in kwargs.keys():
            ellip.set_alpha(0.3)
        if 'color' not in kwargs.keys():# and 'c' not in kwargs.keys():
            ellip.set_facecolor('red')

        ellip.zorder = zorder

        ax.add_patch(ellip)

        # THEN just f***ing plot an invisible line across the ellipse.
        if with_line:
            # brute forcing axes limits so they contain ellipse patch
            # maybe a cleaner way of doing this, but I couldn't work it out
            x_extent = 0.5*(abs(width*np.cos(np.radians(theta))) +
                            abs(height*np.sin(np.radians(theta))))
            y_extent = 0.5*(abs(width*np.sin(np.radians(theta))) +
                            abs(height*np.cos(np.radians(theta))))

            lx = pos[0] - x_extent
            ux = pos[0] + x_extent
            ly = pos[1] - y_extent
            uy = pos[1] + y_extent
            ax.plot((lx, ux), (ly, uy), alpha=0.)

        return ellip

    def plot(self, dim1, dim2, ax=None, comp_now=True, comp_then=False,
             color='red', comp_orbit=False, alpha=0.3, marker='+',
             marker_alpha=0.9,
             msize=10,
             orbit_color=None, orbit_alpha=None,
             comp_then_fill=True,
             comp_then_linewidth='0.1',
             comp_now_linewidth='2',
             orbit_arrow=True,
             mzorder=2,
             comp_orbit_kwargs={},
             **kwargs):
        """
        Conveniently displays the component on the provided axes (or most
        recently used axes) on the provided phase-space plane.

        :param ax:
        :param comp_now:
        :param comp_then:
        :param color:
        :return:
        """
        if ax is None:
            ax = plt.gca()
        labels = 'XYZUVW'

        if type(dim1) is not int:
            dim1 = labels.index(dim1.upper())
        if type(dim2) is not int:
            dim2 = labels.index(dim2.upper())

        if comp_now:
            ax.scatter(self.get_mean_now()[dim1], self.get_mean_now()[dim2], color=color,
                       linewidth=0.0,
                       marker=marker, s=msize, zorder=mzorder)
            self.plot_cov_ellipse(self.get_covmatrix_now()[np.ix_([dim1, dim2], [dim1, dim2])],
                                  self.get_mean_now()[np.ix_([dim1, dim2])],
                                  ax=ax, alpha=alpha, linewidth=comp_now_linewidth,
                                  linestyle='--',
                                  fill=False,
                                  color=color, **kwargs)
        if comp_then:
            ax.scatter(self.get_mean()[dim1], self.get_mean()[dim2], color=color,
                       # linewidth=0.0,
                       marker=marker, s=msize, zorder=mzorder, alpha=marker_alpha)
            self.plot_cov_ellipse(self.get_covmatrix()[np.ix_([dim1, dim2], [dim1, dim2])],
                                  self.get_mean()[np.ix_([dim1, dim2])],
                                  ax=ax, alpha=alpha, linewidth=comp_then_linewidth,
                                  fill=comp_then_fill,
                                  color=color, **kwargs)

        if comp_orbit:
            if orbit_alpha is None:
                orbit_alpha = alpha
            if orbit_color is None:
                orbit_color = color
            self.plot_orbit(dim1, dim2, ax,
                            with_arrow=orbit_arrow, annotate=False, color=orbit_color,
                            alpha=orbit_alpha,
                            )

        return ax

    @classmethod
    def load_from_attributes(cls, filename):
        """
        Load single component from attributes saved to file in dictionary format
        """
        attributes = np.load(filename).item()
        comp = cls(attributes=attributes)
        return comp

    @classmethod
    def get_sensible_walker_spread(cls):
        """Get an array of sensible walker spreads (based on class
        constants `PARAMTER_FORMAT` and `SENSIBLE_WALKER_SPREADS` to
        guide emcee in a sensible starting range of parameters.

        The sensible walker spreads are intuitively set by Tim Crundall.
        The values probably only matter as far as converging quickly to a
        good fit.

        Notes
        -----
        This is a class method because this needs access to certain
        attributes that are class specific, yet doesn't make sense to
        have a whole component object in order to access this.
        """
        sensible_spread = []
        for par_form in cls.PARAMETER_FORMAT:
            sensible_spread.append(cls.SENSIBLE_WALKER_SPREADS[par_form])
        return np.array(sensible_spread)


class SphereComponent(AbstractComponent):
    """
    External Pars: X y Z U V W, dX, dV, age
    (Internal Pars: X y Z U V W, log dX, log dV, age)
    """
    # The format of internal (emcee treated) parameters...
    # TODO: make this clearer
    PARAMETER_FORMAT = ['pos', 'pos', 'pos', 'vel', 'vel', 'vel',
                        'log_pos_std', 'log_vel_std',
                        'age']

    @staticmethod
    def externalise(pars):
        """
        Take parameter set in internal form (as used by emcee) and
        convert to external form (as used to build attributes).
        """
        extern_pars = np.copy(pars)
        extern_pars[6:8] = np.exp(extern_pars[6:8])
        return extern_pars

    @staticmethod
    def internalise(pars):
        """
        Take parameter set in external form (as used to build attributes)
        and convert to internal form (as used by emcee).
        """
        intern_pars = np.copy(pars)
        intern_pars[6:8] = np.log(intern_pars[6:8])
        return intern_pars

    def _set_covmatrix(self, covmatrix=None):
        """Builds covmatrix from self.pars. If setting from an externally
        provided covariance matrix then updates self.pars for consistency"""
        # If covmatrix hasn't been provided, generate from self._pars
        # and set.
        if covmatrix is None:
            dx = self._pars[6]
            dv = self._pars[7]
            self._covmatrix = np.identity(6)
            self._covmatrix[:3, :3] *= dx ** 2
            self._covmatrix[3:, 3:] *= dv ** 2
        # If covmatrix has been provided, reverse engineer the most
        # suitable set of parameters and update self._pars accordingly
        # (e.g. take the geometric mean of the (square-rooted) velocity
        # eigenvalues as dv, as this at least ensures constant volume
        # in velocity space).
        else:
            self._covmatrix = np.copy(covmatrix)
            dx = gmean(np.sqrt(
                np.linalg.eigvalsh(self._covmatrix[:3, :3]))
            )
            dv = gmean(np.sqrt(
                np.linalg.eigvalsh(self._covmatrix[3:, 3:]))
            )
            self._pars[6] = dx
            self._pars[7] = dv
            self.set_sphere_stds()


class EllipComponent(AbstractComponent):
    PARAMETER_FORMAT = ['pos', 'pos', 'pos', 'vel', 'vel', 'vel',
                        'log_pos_std', 'log_pos_std', 'log_pos_std',
                        'log_vel_std',
                        'corr', 'corr', 'corr',
                        'age']

    @staticmethod
    def externalise(pars):
        """
        Take parameter set in internal form (as used by emcee) and
        convert to external form (as used to build attributes).
        """
        extern_pars = np.copy(pars)
        extern_pars[6:10] = np.exp(extern_pars[6:10])
        return extern_pars

    @staticmethod
    def internalise(pars):
        """
        Take parameter set in external form (as used to build attributes)
        and convert to internal form (as used by emcee).
        """
        intern_pars = np.copy(pars)
        intern_pars[6:10] = np.log(intern_pars[6:10])
        return intern_pars

    def _set_covmatrix(self, covmatrix=None):
        """Builds covmatrix from self.pars. If setting from an externally
        provided covariance matrix then updates self.pars for consistency"""
        # If covmatrix hasn't been provided, generate from self._pars
        # and set.
        if covmatrix is None:
            dx, dy, dz = self._pars[6:9]
            dv = self._pars[9]
            c_xy, c_xz, c_yz = self._pars[10:13]
            self._covmatrix = np.array([
                [dx**2,      c_xy*dx*dy, c_xz*dx*dz, 0.,    0.,    0.],
                [c_xy*dx*dy, dy**2,      c_yz*dy*dz, 0.,    0.,    0.],
                [c_xz*dx*dz, c_yz*dy*dz, dz**2,      0.,    0.,    0.],
                [0.,         0.,         0.,         dv**2, 0.,    0.],
                [0.,         0.,         0.,         0.,    dv**2, 0.],
                [0.,         0.,         0.,         0.,    0.,    dv**2],
            ])
        # If covmatrix has been provided, reverse engineer the most
        # suitable set of parameters and update self._pars accordingly
        # (e.g. take the geometric mean of the (square-rooted) velocity
        # eigenvalues as dv, as this at least ensures constant volume
        # in velocity space).
        else:
            self._covmatrix = np.copy(covmatrix)
            pos_stds = np.sqrt(np.diagonal(self._covmatrix[:3, :3]))
            dx, dy, dz = pos_stds
            pos_corr_matrix = (self._covmatrix[:3, :3]
                               / pos_stds
                               / pos_stds.reshape(1,3).T)
            c_xy, c_xz, c_yz = pos_corr_matrix[np.triu_indices(3,1)]
            dv = gmean(np.sqrt(
                np.linalg.eigvalsh(self._covmatrix[3:, 3:]))
            )
            self._pars[6:9] = dx, dy, dz
            self._pars[9] = dv
            self._pars[10:13] = c_xy, c_xz, c_yz


class FreeComponent(AbstractComponent):
    PARAMETER_FORMAT = ['pos', 'pos', 'pos', 'vel', 'vel', 'vel',
                        'log_pos_std', 'log_pos_std', 'log_pos_std',
                        'log_vel_std', 'log_vel_std', 'log_vel_std',
                        'corr', 'corr', 'corr', 'corr', 'corr',
                        'corr', 'corr', 'corr', 'corr',
                        'corr', 'corr', 'corr',
                        'corr', 'corr',
                        'corr',
                        'age']

    @staticmethod
    def externalise(pars):
        """
        Take parameter set in internal form (as used by emcee) and
        convert to external form (as used to build attributes).
        """
        extern_pars = np.copy(pars)
        extern_pars[6:12] = np.exp(extern_pars[6:12])
        return extern_pars

    @staticmethod
    def internalise(pars):
        """
        Take parameter set in external form (as used to build attributes)
        and convert to internal form (as used by emcee).
        """
        intern_pars = np.copy(pars)
        intern_pars[6:12] = np.log(intern_pars[6:12])
        return intern_pars

    def _set_covmatrix(self, covmatrix=None):
        """Builds covmatrix from self.pars. If setting from an externally
        provided covariance matrix then updates self.pars for consistency"""
        # If covmatrix hasn't been provided, generate from self._pars
        # and set.
        if covmatrix is None:
            dx, dy, dz = self._pars[6:9]
            du, dv, dw = self._pars[9:12]
            c_xy, c_xz, c_xu, c_xv, c_xw, \
                  c_yz, c_yu, c_yv, c_yw, \
                        c_zu, c_zv, c_zw, \
                              c_uv, c_uw, \
                                    c_uw, = self._pars[12:27]
            self._covmatrix = np.array([
                [dx**2,      c_xy*dx*dy, c_xz*dx*dz, c_xu*dx*du, c_xv*dx*dv, c_xw*dx*dw],
                [c_xy*dx*dy, dy**2,      c_yz*dy*dz, c_yu*dy*du, c_yv*dy*dv, c_yw*dy*dw],
                [c_xz*dx*dz, c_yz*dy*dz, dz**2,      c_zu*dz*du, c_zv*dz*dv, c_zw*dz*dw],
                [c_xu*dx*du, c_yu*dy*du, c_zu*dz*du, du**2,      c_uv*du*dv, c_uw*du*dw],
                [c_xv*dx*dv, c_yv*dy*dv, c_zv*dz*dv, c_uv*du*dv, dv**2,      c_uw*dv*dw],
                [c_xw*dx*dw, c_yw*dy*dw, c_zw*dz*dw, c_uw*du*dw, c_uv*dw*dv, dw**2     ],
            ])
        # If covmatrix has been provided, reverse engineer the most
        # suitable set of parameters and update self._pars accordingly
        # (e.g. take the geometric mean of the (square-rooted) velocity
        # eigenvalues as dv, as this at least ensures constant volume
        # in velocity space).
        else:
            self._covmatrix = np.copy(covmatrix)
            stds = np.sqrt(np.diagonal(self._covmatrix))
            dx, dy, dz, du, dv, dw = stds

            corr_matrix = (self._covmatrix
                           / stds
                           / stds.reshape(1,6).T)
            c_xy, c_xz, c_xu, c_xv, c_xw, \
                  c_yz, c_yu, c_yv, c_yw, \
                        c_zu, c_zv, c_zw, \
                              c_uv, c_uw, \
                                    c_uw, = corr_matrix[np.triu_indices(6,1)]
            self._pars[6:9] = dx, dy, dz
            self._pars[9:12] = du, dv, dw
            self._pars[12:27] = c_xy, c_xz, c_xu, c_xv, c_xw, \
                                      c_yz, c_yu, c_yv, c_yw, \
                                            c_zu, c_zv, c_zw, \
                                                  c_uv, c_uw, \
                                                        c_uw
