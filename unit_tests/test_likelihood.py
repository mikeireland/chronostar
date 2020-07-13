"""
Check a bunch of things from the likelihood module, a set of functions that
encapsulate emcee fitting of a Component
"""

from astropy.table import Table
import astropy.units as u
import astropy.constants as const
import numpy as np

import sys
sys.path.insert(0,'..')
from chronostar import likelihood
from chronostar.component import SphereComponent, EllipComponent
from chronostar import tabletool
from chronostar.synthdata import SynthData


def astropyCalcAlpha(dx,dv,nstars):
    """
    Assuming we have identified 100% of star mass, and that average
    star mass is 1 M_sun.

    Calculated alpha is unitless
    """
    return ((dv * u.km / u.s) ** 2 * dx * u.pc /
            (const.G * nstars * const.M_sun)).decompose().value


def test_calcAlpha():
    dx = 10.
    dv = 10.
    nstars = 100.
    # Using constant for Msun from python3's astrpy
    if sys.version[0] == '2':
        rtol=1e-3
    else:
        rtol=1e-10
    assert np.isclose(astropyCalcAlpha(dx,dv,nstars),
                      likelihood.calc_alpha(dx, dv, nstars),
                      rtol=rtol)


def test_lnprior():
    dim = 6
    mean = np.zeros(dim)
    covmatrix = np.identity(dim)
    age = 10.
    sphere_comp = SphereComponent(attributes={
        'mean':mean,
        'covmatrix':covmatrix,
        'age':age,
    })
    memb_probs = np.ones(10)

    assert np.isfinite(likelihood.lnprior(sphere_comp, memb_probs))

    # Now increase age to something ridiculous
    sphere_comp.update_attribute(attributes={
        'age':1e10,
    })
    assert np.isinf(likelihood.lnprior(sphere_comp, memb_probs))

    # Try an EllipComponent with a non-symmetrical covariance matrix
    covmatrix[0,1] = 1.01
    # covmatrix[1,0] = 100
    ellip_comp = EllipComponent(attributes={
        'mean':mean,
        'covmatrix':covmatrix,
        'age':age,
    })
    assert np.isinf(likelihood.lnprior(ellip_comp, memb_probs))

    # Try an EllipComponent with a very broken correlation value
    covmatrix[0,1] = 1.01
    covmatrix[1,0] = 1.01
    ellip_comp = EllipComponent(attributes={
        'mean':mean,
        'covmatrix':covmatrix,
        'age':age,
    })
    assert np.isinf(likelihood.lnprior(ellip_comp, memb_probs))


def test_get_lnoverlaps():
    """
    Confirms that star-component overlaps get smaller as stars get further
    away.

    First generates a component `sphere_comp`. Then generates three stars.
    The first one is identical to `sphere_comp` in mean and covmatrix.
    The other two share the same covmatrix yet are separated in X.
    We check that the overlap integral is smaller for the more separated
    stars.
    """
    dim = 6
    mean = np.zeros(dim)
    covmatrix = np.identity(dim)
    age = 1e-10
    sphere_comp = SphereComponent(attributes={
        'mean':mean,
        'covmatrix':covmatrix,
        'age':age,
    })

    dx_offsets = [0., 1., 10.]

    star_comps = []
    for dx_offset in dx_offsets:
        star = SphereComponent(attributes={
            'mean':sphere_comp.get_mean()+np.array([dx_offset,0.,0.,0.,0.,0.]),
            'covmatrix':sphere_comp.get_covmatrix(),
            'age':sphere_comp.get_age(),
        })
        star_comps.append(star)

    nstars = len(star_comps)
    dummy_table = Table(data=np.arange(nstars).reshape(nstars,1),
                        names=['name'])
    tabletool.append_cart_cols_to_table(dummy_table)

    for star_comp, row in zip(star_comps, dummy_table):
        tabletool.insert_data_into_row(row,
                                       star_comp.get_mean(),
                                       star_comp.get_covmatrix(),
                                       cartesian=True,
                                       )
    dummy_data = tabletool.build_data_dict_from_table(dummy_table)
    ln_overlaps = likelihood.get_lnoverlaps(sphere_comp, data=dummy_data)

    # Checks that ln_overlaps is descending
    assert np.allclose(ln_overlaps, sorted(ln_overlaps)[::-1])


def test_lnprob_func():
    """
    Generates two components. Generates a synthetic data set based on the
    first component. Confrims that the lnprob is larger for the first
    component than the second.
    """
    measurement_error = 1e-10
    star_count = 500
    tiny_age = 1e-10
    dim = 6
    comp_covmatrix = np.identity(dim)
    comp_means = {
        'comp1': np.zeros(dim),
        'comp2': 10 * np.ones(dim)
    }
    comps = {}
    data = {}

    for comp_name in comp_means.keys():
        comp = SphereComponent(attributes={
            'mean':comp_means[comp_name],
            'covmatrix':comp_covmatrix,
            'age':tiny_age
        })

        synth_data = SynthData(pars=[comp.get_pars()], starcounts=star_count,
                                measurement_error=measurement_error)
        synth_data.synthesise_everything()
        tabletool.convert_table_astro2cart(synth_data.table)
        data[comp_name] = tabletool.build_data_dict_from_table(synth_data.table)
        comps[comp_name] = comp

    lnprob_comp1_data1 = likelihood.lnprob_func(pars=comps['comp1'].get_pars(),
                                                data=data['comp1'])
    lnprob_comp2_data1 = likelihood.lnprob_func(pars=comps['comp2'].get_pars(),
                                                data=data['comp1'])
    lnprob_comp1_data2 = likelihood.lnprob_func(pars=comps['comp1'].get_pars(),
                                                data=data['comp2'])
    lnprob_comp2_data2 = likelihood.lnprob_func(pars=comps['comp2'].get_pars(),
                                                data=data['comp2'])
    
    print(lnprob_comp1_data1)
    print(lnprob_comp2_data1)
    print(lnprob_comp1_data2)
    print(lnprob_comp2_data2)
    
    assert lnprob_comp1_data1 > lnprob_comp2_data1
    assert lnprob_comp2_data2 > lnprob_comp1_data2

    # Check that the different realisations only differ by 20%
    assert np.isclose(lnprob_comp1_data1, lnprob_comp2_data2, rtol=2e-1)
    assert np.isclose(lnprob_comp1_data2, lnprob_comp2_data1, rtol=2e-1)
