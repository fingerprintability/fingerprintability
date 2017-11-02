#!/bin/python
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

import util as ut
import common as cm

# logger
import logging
logger = logging.getLogger(__name__)


# default plot filename
CDF_FILE = join(cm.BASE_DIR, 'cdf_insta_class.png')

# threshold percentage
PERC_THRESHOLD = 80


def plot_cdf(data, cumulative, base, filename=None):
    """Plot the CDF and save to `filename`."""
    plt.plot(base[:-1], len(data) - cumulative, c='blue')
    if type(filename) is str:
        plt.savefig(filename)
        logger.info("Printed CDF to %s" % filename)


def threshold(dpath, P, plot_file=None):
    """Find minimum number of instances that P% of classes have."""
    counts = ut.instance_counts(dpath)
    data = counts.values()
    values, base = np.histogram(data, bins=len(counts))
    cumulative = np.cumsum(values)
    plot_cdf(data, cumulative, base, plot_file)

    for i, (c, v) in enumerate(zip(cumulative, values)):
        index = -i
        if sum(values[index - 1:]) * 100 / cumulative[-1] > P:
            break
    min_instances = int(base[index - 1])

    logger.info("Num inst that covers %s%% of classes %s" % (P, min_instances))
    logger.debug("Will need to remove %s instances" % sum(values[index - 1:]))
    return min_instances, len(data)


def find_websites_with_less_instances(dpath, N):
    """Remove websites with less than `N` instances."""
    to_remove = []
    sites = 0
    for site, instances in ut.walk_instances(dpath):
        logger.info("Website %s, instances %s" % (site, len(instances)))
        if len(instances) < N:
            logger.warning("\t%s < %s" % (len(instances), N))
            to_remove += instances
            sites += 1
    logger.info("Total instances removed because site has less than threshold "
                "(%s): %s" % (N, len(to_remove)))
    return to_remove, sites


def find_instances_until_threshold(dpath, N):
    """Remove instances of websites with more than `N` instances."""
    to_remove = []
    for site, instances in ut.walk_instances(dpath):
        logger.info("Website %s, instances %s" % (site, len(instances)))
        if len(instances) > N:
            logger.warning("\t%s > %s" % (len(instances), N))
            extra = N - len(instances)
            to_remove += sorted(instances)[:-extra]
    logger.info("Total instances removed because site has more than threshold "
                "(%s): %s" % (N, len(to_remove)))
    return to_remove
