"""
This module defines the parameter reading function.

Credit: Mark Krumholz
"""


def readParam(paramFile, noCheck=False):
    """
    This function reads a parameter file.

    Parameters
    ----------
    paramFile : string
       A string giving the name of the parameter file
    noCheck : bool
       If True, no checking is performed to make sure that all
       mandatory parameters have been specified

    Returns
    -------
    paramDict : dict
       A dict containing a parsed representation of the input file
    """

    # Prepare an empty dict to hold inputs
    paramDict = {}

    # Try to open the file
    fp = open(paramFile, 'r')

    # Read the file
    for line in fp:

        # Skip blank and comment lines
        if line == '\n':
            continue
        if line.strip()[0] == "#":
            continue

        # Break line up based on equal sign
        linesplit = line.split("=")
        if len(linesplit) < 2:
            print("Error parsing input line: " + line)
            raise IOError
        if linesplit[1] == '':
            print("Error parsing input line: " + line)
            raise IOError

        # Trim trailing comments from portion after equal sign
        linesplit2 = linesplit[1].split('#')

        # Store token-value pairs, as strings for now. Type conversion
        # happens below.
        paramDict[linesplit[0].strip()] = linesplit2[0].strip()

    # Close file
    fp.close

    # Try converting parameters to numbers, for convenience
    for k in paramDict.keys():
        try:
            paramDict[k] = int(paramDict[k])
        except ValueError:
            try:
                paramDict[k] = float(paramDict[k])
            except ValueError:
                pass

    # if not noCheck:
    #     mandatory = ['alpha', 'gamma', 'ibc_pres_type', 'ibc_enth_type',
    #                  'ibc_pres_val', 'obc_pres_type', 'obc_enth_type',
    #                  'obc_pres_val']
    #     for m in mandatory:
    #         if not m in paramDict:
    #             raise ValueError("Error: must specify parameter " + m + "!\n")

    # Return the dict
    return paramDict
