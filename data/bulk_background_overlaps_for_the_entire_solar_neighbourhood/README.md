Determine background overlaps for all stars in the Solar neighbourhood. Currently everything <200 pc and parallax_error<0.3mas.

For now `background_log_overlap_merged.fits` is in `/data/mash/marusa/chronostar_projects/solar_neighbourhood/`, but it is not finished yet.

## TODO
- Implement this database in `chronostar`: when uploading data table, take background overlaps from this database and compute only those that are not available yet. Potentially add these new stars to the central database.
