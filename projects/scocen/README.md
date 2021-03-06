# Scorpius-Centaurus Association


## TODO
- Crossing time at time NOW not 0
- data_table=data_table.filled() # TODO: fill with a stupid value!!!!
- component overlaps: # old_memb_probs are used to scale the amplitude of the components # TODO: This gives all the components the same amplitude! This is not OK!!!


## Typical workflow
### (1) Data preparation
- Get Gaia data with ADQL query `data/gaia_DR2_scocen.adql` (this includes RVs and other values from some external catalogs)
- Sort out radial velocities with `gaia_table_best_radial_velocity_and_lithium.py`
- Prepare a parameter file for data conversion `convert_gaia_to_XYZUVW.pars`
- Convert astrometry to the Cartesian system: `python /Users/marusa/chronostar/scripts/prepare_data.py convert_gaia_to_XYZUVW.pars`
- Background overlaps: compute them and add them to the data table
- Prepare a parameter file for Chronostar

### (2) Run Chronostar
- On `mash`: start `screen` so the job keeps running in the background.
`/pkg/linux/anaconda/bin/python run_chronostar.py mypars.pars`
- On `avatar`: `qsub avatar_script.sh` or `qsub avatar_script_mpirun.sh` for the parallel code.


I think this is `python3`. It works.

### (3) Data visualisation
- `quickplot.py` on `mash`



### Data
#### Location
- I copied all data to mash (April 3): `mash:/priv/mulga1/marusa/chronostar_projects/scocen/paper/`
- `final_comps_21.fits`: Components
- `scocen_vac_DR2_with_21_overlaps.fits`: ScoCen data with component overlaps
- `scocen_5k_candidates_only_members_with_significant_membership_from_tims_3_subsets.fits`: Input data for the fit
- `lithium_isochrones/`: Code to plot lithium isochrones

### Data management
- `convert_gaia_to_XYZUVW.pars`: Parameter file for Chronostar's `prepare_data.py` that converts astrometry to the Cartesian coordinate system
- `component_overlaps.py`: Take components and compute star-component overlaps for a big sample of stars.
- `gaia_table_best_radial_velocity_and_lithium.py`: In case there are RVs available from the multiple sources, find the best value and create a new column `radial_velocity` and the same for the errors. Similar for lithium.
- `best_component_membership.py`: Find components with the best memberships, add kinematic age for each star etc.
- `prepare_component_members_for_further_splitting.py`: Print subsets of the big data table that contain members with membership>small value.

### Code
- `scocenlib.py`: This is where you set filenames and good components. This file is used by the rest of the code.
- `cmd_components_simple.py`: Plot CMDs for all the components
- `pds70.py`: Find what component PDS70 belongs to and its component age
- `individual_component_diagnostics.py`: Plots for each component
- `XU_comps_CUT.py`: CUT components in the XU spacec
- `YV_comps.py`: YV space
- `ZW_comps.py`: ZW space
- `galaxy_ages.py`: Members in (l, b) coordinates, with overplotted component ages
- `galaxy_velocities.py`: Members in (l, b) space with overplotted U velocity
- `manage_background_overlaps.py`: Add background overlaps to the data table
- `galaxy_input_data_with_RVs.py`: Plot input data
- `cmd_black.py`: CMD with all components in black
- `galaxy_black.py`: All stars with black color. It is easier to see substructure here.


### Paper plots
- `scocenlib.py`: Config file with data and component filenames, a list of good/bad components, colours for each component etc.
- `galaxy_components.py`: Plot components in `(l, b)` space, `gx.pdf`
- `membership_histogram.py`: Membership probability distribution for each component. Figure `memb_hist.pdf`.
- `cmd_age_sequence_CUT_with_lithium.py`: CMD for CUT components that show a kinematic age sequence. `cmd_li_CUT.pdf`
- `cmd_age_sequence_AG_with_lithium.py`: CMD for AG components. `cmd_li_AG.pdf`
- `XU_comps.py`: Plot components in the X-U space. `XU_comps.pdf`
- `lithium_vs_color_isochrones.py`: `lithium_isochrones.pdf`, TODO. Needs `isochrones` folder with models
- `fig_settings.py`: Library for pretty plots

### Paper tables
- `print_components_overlaps_table_for_paper.py`: Print table with number of members for each component (`componentsoverlaps.tex` and `tab.components_overlaps` in the paper)
- `print_components_table_for_paper.py`: Print table with component properties at time 0 (`componentsfit.tex` and `tab.components_fit`)
- TODO: Print a table with the main results: stellar ids and their membership probabilities (`tab.results`)




