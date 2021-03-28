# Utils
Useful scripts (not super polished yet). These files should probably be moved somewhere else, e.g. in `scripts`?

#### Data visualisation
- `spherical_cow.py`: Plot data and components (ellipses) in XU, UW space etc.
- `animate_results_save.py`: Animation of a given set of components, along with memberships, over time.
- `plot_result.py`: Conveniently plot a given set of components, along with memberships.
- `quickplot.py`: Make a quick plot for specific split, e.g. `4/C/final/`. Plot XU, CMD, (l, b) etc.

#### Pre-compute background overlaps for large tables (this needs to be done only once)
- `bg_ols_multiprocessing.py`: Compute background overlaps (multiprocessing)
- `merge_bg_results_into_one_fits_file.py`: Each process from `bg_ols_multiprocessing.py` saves results in a separate file. This script merges them together into one.
- `manage_background_overlaps_in_the_table.py`: Save results from `bg_ols_multiprocessing.py` into the original Gaia table

#### Data prep
- `gaia_adql_query_example.txt`: Select data from the Gaia archive
- `component_overlaps.py`: Compute overlaps between given components and stars. This is useful if e.g. there are stars that were not used in the fit, e.g. stars with no RVs.
- `banyan_component_overlaps.py`: Compute overlaps between Banyan components and stars
- `best_radial_velocity.py`: If a data table contains radial velocities from different sources, this script will find values with smallest uncertainties. Merges lithium information into one column.

#### Data management
- `best_component_membership.py`: Add columns to the table that already has membership info: here, add a column with the most likely component, add its age etc.
