"""
Tests the SynthData class which is used to generate synthetic stellar
kinematic data from simple, Gaussian distributions.
"""
from __future__ import print_function, division, unicode_literals

from astropy.table import Table, join
import numpy as np
import pytest

import sys
sys.path.insert(0,'..')
from chronostar.synthdata import SynthData
from chronostar.component import SphereComponent, EllipComponent
from chronostar import tabletool
from chronostar import traceorbit

PARS =  np.array([
    [0., 0., 0., 0., 0., 0., 10., 5., 1e-5],
    [5., 0.,-5., 0., 0., 0., 10., 5., 40.]
])
STARCOUNTS = [50, 30]
COMPONENTS = SphereComponent

# python3 throws FileNotFoundError that is essentially the same as IOError
try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

def test_initialisation():
    """Basic sanity check to see if things start off ok"""
    sd = SynthData(pars=PARS, starcounts=STARCOUNTS, Components=COMPONENTS)

    assert np.allclose(PARS, sd.pars)
    assert sd.ncomps == len(PARS)
    assert np.allclose(PARS[0], sd.components[0].get_pars())
    assert np.allclose(np.array(STARCOUNTS), sd.starcounts)


    sd2 = SynthData(pars=PARS[0], starcounts=STARCOUNTS[0],
                    Components=COMPONENTS)
    assert np.allclose(np.array([STARCOUNTS[0]]), sd2.starcounts)

    starcounts = 50.
    sd3 = SynthData(pars=PARS[0], starcounts=starcounts,
                    Components=COMPONENTS)
    assert np.allclose(np.array([np.int(starcounts)]), sd3.starcounts)


def test_generateInitXYZUVW():
    """Check that the mean of initial xyzuvw of stars matches that of the
    initialising component"""
    starcounts = (int(1e6),)
    sd = SynthData(pars=PARS[:1], starcounts=starcounts, Components=COMPONENTS)
    sd.generate_all_init_cartesian()

    comp = SphereComponent(PARS[0])
    init_xyzuvw = sd.extract_data_as_array([dim + '0' for dim in 'xyzuvw'])
    assert np.allclose(comp.get_mean(), np.mean(init_xyzuvw, axis=0),
                       atol=0.1)


def test_projectStars():
    """Check that the mean of stars after projection matches the mean
    of the component after projection"""
    starcounts = (int(1e3),)
    sd = SynthData(pars=PARS[:1], starcounts=starcounts, Components=COMPONENTS,
                   trace_orbit_func=traceorbit.trace_epicyclic_orbit)
    sd.generate_all_init_cartesian()
    sd.project_stars()

    comp_mean_now, comp_covmatrix_now = \
        sd.components[0].get_currentday_projection()
    final_xyzuvw = sd.extract_data_as_array([dim + '_now' for dim in 'xzyuvw'])
    assert np.allclose(comp_mean_now, final_xyzuvw.mean(axis=0), atol=1.)


def test_measureXYZUVW():
    """Check measurements of xyzuvw_now to astrometry occur properly.
    Will use extremely dense component as case study as this ensures stars
    all have more or less the same true values"""
    compact_comp_pars = np.copy(PARS[0])
    compact_comp_pars[6] = 1e-15
    compact_comp_pars[7] = 1e-15
    compact_comp_pars[8] = 1e-15
    starcounts = [1000]

    sd = SynthData(pars=np.array([compact_comp_pars]), starcounts=starcounts,
                   Components=COMPONENTS,
                   trace_orbit_func=traceorbit.trace_epicyclic_orbit)
    sd.generate_all_init_cartesian()
    sd.project_stars()
    sd.measure_astrometry()

    for colname in SynthData.DEFAULT_ASTR_COLNAMES:
        assert np.allclose(sd.GERROR[colname + '_error'],
                           sd.table[colname + '_error'])
        # Check spread of data is similar to Gaia error, we use
        # a large tolerance so a small number of stars can be used
        assert np.isclose(sd.GERROR[colname + '_error'],
                          np.std(sd.table[colname]),
                          rtol=1e-1)


def test_storeTable():
    """Check storing table and loading works"""
    filename = 'temp_data/test_storeTable_output.fits'
    sd = SynthData(pars=PARS, starcounts=STARCOUNTS, Components=COMPONENTS,
                   trace_orbit_func=traceorbit.trace_epicyclic_orbit)
    sd.synthesise_everything()
    sd.store_table(filename=filename, overwrite=True)
    stored_table = Table.read(filename)

    assert np.allclose(sd.table['parallax'], stored_table['parallax'])


def test_synthesiseEverything():
    """Check everything goes to plan with single call"""
    sd = SynthData(pars=PARS, starcounts=STARCOUNTS, Components=COMPONENTS,
                   trace_orbit_func=traceorbit.trace_epicyclic_orbit)
    sd.synthesise_everything()

    assert np.isclose(np.sum(STARCOUNTS), len(sd.table))


def test_storeAndLoad():
    """Check that storing and loading works as expected"""
    filename = 'temp_data/test_synthesiseEverything_output.fits'
    sd = SynthData(pars=PARS, starcounts=STARCOUNTS, Components=COMPONENTS,
                   trace_orbit_func=traceorbit.trace_epicyclic_orbit)
    sd.synthesise_everything(filename=filename, overwrite=True)

    # Trying to store table at `filename` without overwrite throws error
    try:
        sd.synthesise_everything(filename=filename, overwrite=False)
    except IOError:
        pass

    #TODO: implement some means of storing (and loading) entire object


def test_artificialMeasurement():
    """Ensure that scaling the measurement uncertainty scales the reported
    uncertainties appropriately, and that offsets in data due to error scale
    with input error"""
    pars = PARS[:1]
    starcounts = [100]
    sd_dict = {}
    names = ['perf', 'good', 'norm', 'bad']
    m_err_dict = {
        'perf':1e-10,
        'good':1e-1,
        'norm':1.0,
        'bad':1e1,
    }
    for name in names:
        np.random.seed(1)
        sd = SynthData(pars=pars, starcounts=starcounts,
                       measurement_error=m_err_dict[name],
                       Components=COMPONENTS,
                       trace_orbit_func=traceorbit.trace_epicyclic_orbit)
        sd.synthesise_everything()
        sd_dict[name] = sd

    # Assert that measurement errors are stored correctly in columns
    for name in names[1:]:
        assert np.allclose(
                sd_dict[name].table['radial_velocity_error'],
                m_err_dict[name]*SynthData.GERROR['radial_velocity_error']
        )

    # Get reference for degree of offset expected
    norm_offset = np.mean(
            np.abs(sd_dict['perf'].table['radial_velocity']
                   - sd_dict['norm'].table['radial_velocity'])
    )

    bad_offset = np.mean(
            np.abs(sd_dict['perf'].table['radial_velocity']
                   - sd_dict['bad'].table['radial_velocity'])
    )
    good_offset = np.mean(
            np.abs(sd_dict['perf'].table['radial_velocity']
                   - sd_dict['good'].table['radial_velocity'])
    )

    # Check the average offset scales with incorporated measurement error
    assert np.isclose(norm_offset*m_err_dict['bad'], bad_offset)
    assert np.isclose(norm_offset*m_err_dict['good'], good_offset)

def test_multiple_synth_components():
    """Check initialising with multiple components works"""
    age = 1e-10
    dx = 5.
    dv = 2.
    ass_pars1 = np.array([10, 20, 30, 40, 50, 60, dx, dv, age])
    comp1 = SphereComponent(ass_pars1)
    ass_pars2 = np.array([0., 0., 0, 0, 0, 0, dx, dv, age])
    comp2 = SphereComponent(ass_pars2)
    starcounts = [100, 100]
    try:
        synth_data = SynthData(pars=[ass_pars1, ass_pars2],
                               starcounts=starcounts[0],
                               Components=SphereComponent,
                               trace_orbit_func=traceorbit.trace_epicyclic_orbit)
        raise UserWarning('AssertionError should have been thrown by synthdata')
    except AssertionError:
        pass

    synth_data = SynthData(pars=[ass_pars1, ass_pars2],
                           starcounts=starcounts,
                           Components=SphereComponent,
                           trace_orbit_func=traceorbit.trace_epicyclic_orbit)
    synth_data.synthesise_everything()

    assert len(synth_data.table) == np.sum(starcounts)
    means = tabletool.build_data_dict_from_table(
            synth_data.table,
            main_colnames=[el+'0' for el in 'xyzuvw'],
            only_means=True
    )
    assert np.allclose(comp2.get_mean(), means[starcounts[0]:].mean(axis=0),
                       atol=2.)
    assert np.allclose(comp1.get_mean(), means[:starcounts[0]].mean(axis=0),
                       atol=2.)

def test_different_component_forms():
    """Check component forms can be different"""
    tiny_age = 1e-10

    mean1 = np.zeros(6)
    covmatrix1 = np.eye(6) * 4
    comp1 = SphereComponent(attributes={
        'mean':mean1,
        'covmatrix':covmatrix1,
        'age':tiny_age,
    })

    mean2 = np.zeros(6) + 10.
    covmatrix2 = np.eye(6) * 9
    comp2 = EllipComponent(attributes={
        'mean':mean2,
        'covmatrix':covmatrix2,
        'age':tiny_age,
    })
    starcounts = [100,100]

    synth_data = SynthData(pars=[comp1.get_pars(), comp2.get_pars()],
                           starcounts=starcounts,
                           Components=[SphereComponent, EllipComponent],
                           trace_orbit_func=traceorbit.trace_epicyclic_orbit)
    synth_data.synthesise_everything()
    assert len(synth_data.table) == np.sum(starcounts)


def test_background_component():
    """Create artificial association composed of two stars at opposite vertices
    of unit 6D rectangle. Then base background density distribution on that."""
    background_density = 100

    # Since the background double the span of data, by setting the means as
    # follows, the backbround should extend from 0 to 1 in each dimension,
    # which greatly simplifies reasoning about densities and starcounts.
    upper_mean = np.zeros(6) + 0.75
    lower_mean = np.zeros(6) + 0.25
    narrow_dx = 1e-10
    narrow_dv = 1e-10
    tiny_age = 1e-10
    upper_pars = np.hstack((upper_mean, narrow_dx, narrow_dv, tiny_age))
    lower_pars = np.hstack((lower_mean, narrow_dx, narrow_dv, tiny_age))

    starcounts = [1,1]

    synth_data = SynthData(pars=[upper_pars, lower_pars],
                           starcounts=starcounts,
                           background_density=background_density,
                           bg_span_scale=2.0,
                           trace_orbit_func=traceorbit.trace_epicyclic_orbit)
    # synth_data.generate_all_init_cartesian()
    # synth_data.project_stars()
    synth_data.synthesise_everything()

    means = tabletool.build_data_dict_from_table(
            synth_data.table[2:],
            main_colnames=[el+'_now' for el in 'xyzuvw'],
            only_means=True,
    )
    assert np.allclose(0.5, np.mean(means, axis=0), atol=0.1)
    assert np.allclose(1.0, np.max(means, axis=0), atol=0.1)
    assert np.allclose(0.0, np.min(means, axis=0), atol=0.1)
    assert len(synth_data.table) == background_density + 2

def test_non_spherical_component():
    from chronostar.component import FreeComponent


    XU_CORR = 0.5
    NSTARS = 1000

    # Initialise pars with everything = 0
    my_free_pars = np.zeros(len(FreeComponent.PARAMETER_FORMAT))

    # Set standard deviations to 1.
    # WARNING! Rederivation of correlation relies on standard deviations
    # being set to 1.
    my_free_pars[6:12] = 1.
    my_free_pars[14] = XU_CORR  # Set XU correlation to XU_CORR

    my_free_comp = FreeComponent(pars=my_free_pars)

    my_synth_data = SynthData(pars=my_free_pars, starcounts=NSTARS,
                              Components=FreeComponent,
                               trace_orbit_func=traceorbit.trace_epicyclic_orbit)
    my_synth_data.generate_all_init_cartesian()
    ##  Don't actually need everything
    ##  my_synth_data.synthesise_everything()

    # Use this for initial star positions
    mean_colnames = [el + '0' for el in
                     'xyzuvw']
    ##  Use this for current day star positions, will need to uncomment
    ##  synthesise_everything() though
    ##  mean_colnames = [el+'_now' for el in 'xyzuvw']

    means = tabletool.build_data_dict_from_table(
            my_synth_data.table[2:],
            main_colnames=mean_colnames,
            only_means=True,
    )

    # Extract only the X and U positions
    xu_pos = means[:,np.array([0,3])]

    # Calculate xu correlation of synthesised stars via their covariance
    calc_xu_corr = np.cov(xu_pos.T)[0,1]

    assert np.isclose(calc_xu_corr, XU_CORR, atol=0.05)


@pytest.mark.skip(reason="Need Neelish to insert (and initialise) his component")
def test_neelish_component():
    from chronostar.component import SomeComponent as NeelishComponent

    XU_CORR = 0.5
    NSTARS = 1000

    # Initialise pars with everything = 0
    my_pars = np.zeros(len(NeelishComponent.PARAMETER_FORMAT))

    # Initialise other parameters
    # !<TODO: Initialise your desired parameters here to yield required xu correlation >
    # !<TODO: For simplicity set standard deviations to 1. >

    # Generate synthetic data, but only initial mean positions
    my_synth_data = SynthData(pars=my_pars, starcounts=NSTARS,
                              Components=NeelishComponent,
                               trace_orbit_func=traceorbit.trace_epicyclic_orbit)
    my_synth_data.generate_all_init_cartesian()

    # Build an array of means from stars' initial positions
    mean_colnames = [el + '0' for el in 'xyzuvw']
    means = tabletool.build_data_dict_from_table(
            my_synth_data.table[2:],
            main_colnames=mean_colnames,
            only_means=True,
    )

    # Extract only the X and U positions
    xu_pos = means[:,np.array([0,3])]

    # Calculate xu correlation of synthesised stars via their covariance
    calc_xu_corr = np.cov(xu_pos.T)[0,1]

    assert np.isclose(calc_xu_corr, XU_CORR, atol=0.05)

if __name__ == '__main__':
    pass
