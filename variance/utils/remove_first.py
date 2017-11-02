#!/bin/python
import sys
import argparse
from os import listdir
from os.path import join, basename

import util as ut
import common as cm

# logger
import logging
logger = logging.getLogger(__name__)


def get_instance(filename):
    return filename.split('_')[-1]


def first_visits(dpath, batch_instances):
    """Return the first visit of a batch."""
    first_visits = []
    for f in listdir(dpath):
        i = get_instance(f)
        if int(i) % batch_instances == 0:
            fpath = join(dpath, f)
            logger.info("first instance of batch %s" % fpath)
            first_visits.append(fpath)
    return first_visits
