"""
Encountering some difficulties getting pytest to
generate standard logging files. Exploring here
"""

import logging

def test_logfile():
    logging.log(logging.INFO, 'whattup test')