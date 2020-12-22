# Prepare data

- `background_log_overlaps.fits`: Background overlaps for 1.7M nearby stars. `source_id` is from Gaia DR2.
- `bg_ols_multiprocessing.py`: Compute background overlaps using multiprocessing. This is using `/home/tcrun/chronostar/data/gaia_cartesian_full_6d_table.fits` for background.
- `manage_background_overlaps_in_the_table.py`, `merge_bg_results_into_one_fits_file.py`: Add background overlaps to the table
- `prepare_data_add_kinematics.py`: Deal with missing RVs: change them to 0 and change uncertainties to 1e+4
