# Chronostar scripts

## Prepare data
- `prepare_data.py`: Convert astrometry to the Cartesian space XYZUVW, compute background overlaps, make a cut in the parameter space. Takes input parameters from `example_dataprep.pars`.
- `example_dataprep.pars`: Parameters for `prepare_data.py`

## Running Chronostar
- `example_runnaivefit.pars`: Parameters for `run_chronostar.py`
- `run_chronostar.py` [recommended]: The main script to run Chronostar. Takes input from `example_runnaivefit.pars`
- `run_smart_chronostar.py`: Runs `SmartSplitFit` that is more efficient in component splitting by accepting all components that improve the fit in each step [TODO: make this description more clear].

## Data management
- `prepare_fits_file_with_results.py`: Convert npy file with results into a fits file

