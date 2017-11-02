#!/bin/python
import sys
import argparse
import math
import numpy as np

import util as ut

# logger
import logging
logger = logging.getLogger(__name__)


# outlier removal strategies
SIMPLE_TYPE = 'simple'
STRICT_TYPE = 'strict'
WANG_TYPE = 'wang'
TYPES = [WANG_TYPE, SIMPLE_TYPE, STRICT_TYPE]


def simple_strategy(instances, sizes):
    q1 = np.percentile(sizes, 25)
    q3 = np.percentile(sizes, 75)
    remove = []
    for instance, size in zip(instances, sizes):
        if (size < (q1 - 1.5 * (q3 - q1))) or (size > (q3 + 1.5 * (q3 - q1))):
            logger.warning('Instance %s is outlier: size = %s and q1 = %s, '
                         'q3 = %s' % (instance, size, q1, q3))
            remove.append(instance)
    return remove


def wang_strategy(instances, sizes):
    remove = []
    med = np.median(sizes)
    for instance, size in zip(instances, sizes):
        if size < 0.2 * med:
            logger.debug('Instance %s is outlier: size = %s and median = %s'
                         % (instance, size, med))
            remove.append(instance)
    return remove


def strict_strategy(instances, sizes):
    remove = []
    med = np.median(sizes)
    for instance, size in zip(instances, sizes):
        if size < 0.2 * med or size > 1.8 * med:
            remove.append(instance)
            logger.warning('Instance %s is outlier: size = %s and median = %s'
                         % (instance, size, med))
    remove += simple_strategy([x for x in instances if x not in remove], sizes)
    return remove


def find_outliers(dpath, strategy):
    """Rewrite of Panchenko's code to detect outliers."""
    outliers = []
    for site, instances in ut.walk_instances(dpath):
        logger.debug("Find outliers in site %s (instances %s)"
                     % (site, len(instances)))

        # Remove instances with less than 2 incoming packets
        incoming_sizes = []
        for instance in instances:
            incoming = [x for x in ut.get_incoming(instance)]
            insize = abs(sum(incoming))
            logger.debug("Instance %s, size %s, incoming packets %s"
                         % (instance, insize, len(incoming)))
            if len(incoming) <= 2 or insize < 2 * 512:
                logger.warning("Instance %s has less than two incoming packets"
                               % instance)
                outliers.append(instance)
                continue
            incoming_sizes.append(insize)
        instances = [x for x in instances if x not in outliers]

        if strategy == SIMPLE_TYPE:
            outliers += simple_strategy(instances, incoming_sizes)

        elif strategy == WANG_TYPE:
            outliers += wang_strategy(instances, incoming_sizes)

        elif strategy == STRICT_TYPE:
            outliers += strict_strategy(instances, incoming_sizes)

    return outliers


def remove_outliers(path, strategy, dry):
    # find outliers
    logger.info("Start %s outlier detection on %s" % (strategy, path))
    outliers = find_outliers(path, strategy)

    # log outliers
    for outlier in outliers:
        logger.warning("Instance %s is an outlier." % outlier)

    # remove outliers
    if not dry:
        ut.remove_instances(path, outliers)
