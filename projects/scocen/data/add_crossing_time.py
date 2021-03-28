"""
Add Crossing time to the components fits file

TODO: WARNING: These are coordinates at time 0 in the past! You should traceforward these to the present and only then determine crossing time!

"""


from astropy.table import Table

comps_filename = 'data/final_comps_21.fits'

# Read components (WARNING: These are coordinates at time 0 in the past! You should traceforward these to the present!)
comps = Table.read(comps_filename)

# Crossing time. Only components with sigma<age have reliable ages.
crossing_time = comps['dX']/comps['dV'] * 0.977813106 # pc/km*s to Myr
comps['Crossing_time'] = crossing_time
mask = crossing_time < comps['age'] # sigma < age
comps['Age_reliable'] = mask

comps.write(comps_filename, overwrite=True)
