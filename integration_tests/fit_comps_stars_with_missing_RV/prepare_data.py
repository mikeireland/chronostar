"""
Author: Marusa Zerjal, 2018 - 08 - 20

Prepare data for component fit. The idea is to have two sets of data:
A moving group... TODO
"""

# First see if a data savefile path has been provided, and if
# so, then just assume this script has already been performed
# and the data prep has already been done
if (config.config['data_savefile'] != '' and
        os.path.isfile(config.config['data_savefile'])):
    log_message('Loading pre-prepared data')
    datafile = config.config['data_savefile']
    data_table = tabletool.load(datafile)
    historical = 'c_XU' in data_table.colnames

# Otherwise, perform entire process
else:
    # Construct synthetic data if required
    if config.synth is not None:
        log_message('Getting synthetic data')
        datafile = config.config['data_savefile']
        if not os.path.exists(datafile) and config.config['pickup_prev_run']:
            synth_data = SynthData(pars=config.synth['pars'],
                                   starcounts=config.synth['starcounts'],
                                   Components=Component)
            synth_data.synthesise_everything(filename=datafile,
                                             overwrite=True)
            np.save(rdir+'true_synth_pars.npy', config.synth['pars'])
            np.save(rdir+'true_synth_starcounts.npy', config.synth['starcounts'])
        else:
            log_message('Synthetic data already exists')
    else:
        datafile = config.config['data_loadfile']
    assert os.path.exists(datafile)

    # Read in data as table
    log_message('Read data into table')
    data_table = tabletool.read(datafile)

    historical = 'c_XU' in data_table.colnames

    # If data cuts provided, then apply them
    if config.config['banyan_assoc_name'] != '':
        bounds = get_region(
                config.config['banyan_assoc_name'],
                pos_margin=config.advanced.get('pos_margin', 30.),
                vel_margin=config.advanced.get('vel_margin', 5.),
                scale_margin=config.advanced.get('scale_margin', None),
        )
    elif config.data_bound is not None:
        bounds = (config.data_bound['lower_bound'],
                  config.data_bound['upper_bound'])
    else:
        bounds = None

    if bounds is not None:
        log_message('Applying data cuts')
        star_means = tabletool.build_data_dict_from_table(
                datafile,
                main_colnames=config.cart_colnames.get('main_colnames', None),
                only_means=True,
                historical=historical,
        )
        data_mask = np.where(
                np.all(star_means < bounds[1], axis=1)
                & np.all(star_means > bounds[0], axis=1))
        data_table = data_table[data_mask]
    log_message('Data table has {} rows'.format(len(data_table)))
