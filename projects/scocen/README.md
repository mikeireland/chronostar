# Scorpius-Centaurus Association

## Data
### Results from December 2020
Currently in `Google drive` but will move this to `mash` after some re-organisation:
- `final_comps.fits`: Components
- `members_from_5k_fit_and_overlaps.fits`: Input data and membership probabilities

## Data management
- `convert_gaia_to_XYZUVW.pars`: Parameter file for Chronostar's `prepare_data.py` that converts astrometry to the Cartesian coordinate system
- `component_overlaps.py`: Take components and compute star-component overlaps for a big sample of stars.
- `gaia_table_best_radial_velocity_and_lithium.py`: In case there are RVs available from the multiple sources, find the best value and create a new column `radial_velocity` and the same for the errors. Similar for lithium.
- `best_component_membership.py`: Find components with the best memberships, add kinematic age for each star etc.
- `prepare_component_members_for_further_splitting.py`: Print subsets of the big data table that contain members with membership>small value.

## Code
- `cmd_components_simple.py`: Plot CMDs for all the components
- `pds70.py`: Find what component PDS70 belongs to and its component age

## Paper plots
- `scocenlib.py`: Config file with data and component filenames, a list of good/bad components, colours for each component etc.
- `galaxy_components.py`: Plot components in `(l, b)` space, `fig.gx`
- `membership_histogram.py`: Membership probability distribution for each component. Figure `fig.memb_hist`.
- `cmd_age_sequence_CUT_with_lithium.py`: CMD for CUT components that show a kinematic age sequence. `cmd_li_CUT.pdf`
- `cmd_age_sequence_AG_with_lithium.py`: CMD for AG components. `cmd_li_AG.pdf`
- `XU_comps.py`: Plot components in the X-U space. `XU_comps.pdf`
- `lithium_vs_color_isochrones.py`: `lithium_isochrones.pdf`, TODO. Needs `isochrones` folder with models
- `fig_settings.py`: Library for pretty plots

## Paper tables
- `print_components_overlaps_table_for_paper.py`: Print table with number of members for each component (`componentsoverlaps.tex` and `tab.components_overlaps` in the paper)
- `print_components_table_for_paper.py`: Print table with component properties at time 0 (`componentsfit.tex` and `tab.components_fit`)
- TODO: Print a table with the main results: stellar ids and their membership probabilities (`tab.results`)



TODO: age --> Age
