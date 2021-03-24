#### Pre-compute background overlaps for large tables (this needs to be done only once)
- `bg_ols_multiprocessing.py`: Compute background overlaps (multiprocessing)
- `merge_bg_results_into_one_fits_file.py`: Each process from `bg_ols_multiprocessing.py` saves results in a separate file. This script merges them together into one.
- `manage_background_overlaps_in_the_table.py`: Save results from `bg_ols_multiprocessing.py` into the original Gaia table

#### Data prep
- `gaia_adql_query_example.txt`: Select data from the Gaia archive
- `component_overlaps.py`: Compute overlaps between given components and stars. This is useful if e.g. there are stars that were not used in the fit, e.g. stars with no RVs.
- `banyan_component_overlaps.py`: Compute overlaps between Banyan components and stars

#### Component visualisation
- `spherical_cow.py`: Plot data and components (ellipses) in XU, UW space etc.
