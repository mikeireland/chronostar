import numpy as np

from chronostar import compfitter

from chronostar import datatool
from chronostar import tabletool

import sys

if __name__ == '__main__':
    data_file = sys.argv[1]

    data_dict = tabletool.build_data_dict_from_table(data_file)
    res = compfitter.fit_comp(data_dict)

    # nstars = len(data_dict['means'])
    # memb_prob = np.ones(nstars)


