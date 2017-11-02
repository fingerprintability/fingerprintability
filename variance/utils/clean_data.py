#!/bin/python2
import sys
import operator
from os import listdir, remove, mkdir
from os.path import isfile, join, isdir, split
from shutil import rmtree, copytree, copy
import argparse
import ConfigParser
from tempfile import gettempdir
from time import strftime

import util as ut
import common as cm
import threshold as th
from format import knn_format
from remove_first import first_visits
from outliers import find_outliers, SIMPLE_TYPE
from duplicates import get_duplicate_list

# logger
import logging
logger = logging.getLogger()


def main():
    # Read configuration file
    config = ConfigParser.RawConfigParser()
    config.read(cm.CONFIG_FILE)

    # parse argumens
    parser = get_parser(config)
    args = parser.parse_args()
    # get config params
    config = {k: v for k, v in config.items(args.config)}

    # config logging
    logger.setLevel(args.loglevel)
    logfile = join(cm.SRC_DIR, '%s.log' % config['id'])
    if isfile(logfile):
        remove(logfile)
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setFormatter(logging.Formatter(cm.FORMAT))
    logger.addHandler(fileHandler)
    del args.loglevel

    clean(config, args.dry)


def reset_state(dirs, files):
    for d in dirs:
        if isdir(d):
            rmtree(d)
            mkdir(d)
    for f in files:
        if isfile(f):
            remove(f)


def clean(config, dry=None):
    cid = config['id']
    output = join(cm.SRC_DIR, cid)
    path = join(cm.OURS_DIR, cid)

    # remove files and dirs from previous runs
    cdf_file = join(cm.SRC_DIR, '%s_cdf.png' % cid)
    map_file = join(cm.SRC_DIR, '%s_map.txt' % cid)
    url_file = join(cm.ETC_DIR, '%s_urls.txt' % cid)
    temp_dir = join(gettempdir(), strftime('%y%m%d_%H%M%S'))
    dir_name, fold_name = split(output)
    knn_dir = join(dir_name, "knn_" + fold_name)
    reset_state([output, knn_dir], [cdf_file, map_file])

    # parse tshark files
    parsed_dir = join(cm.OURS_DIR, cid + '_parsed')
    copytree(parsed_dir, temp_dir)

    logger.info("Parsing %s to %s" % (path, output))

    # remove first visits
    to_remove = []
    if config['remove_first'] == 'True':
        to_remove += first_visits(temp_dir, int(config['visits']))

    # remove errors
    error_list = join(cm.ERRORS_DIR, '%s_errors.txt' % cid)
    if isfile(error_list):
        for fname in open(error_list):
            logger.info("Remove errored instance: %s" % fname.strip())
            to_remove.append(fname.strip())

    if not dry:
        ut.remove_instances(temp_dir, to_remove)

    # remove outliers
    to_remove = find_outliers(temp_dir, SIMPLE_TYPE)
    if not dry:
        ut.remove_instances(temp_dir, to_remove)

    # remove duplicates
    to_remove = []
    instances = ut.instances(temp_dir)
    for duplicates in get_duplicate_list():
        logger.info("Remove duplicates %s" % duplicates)
        dinstances = {k: v for k, v in instances.iteritems() if k in duplicates}
        dcounts = {k: len(v) for k, v in dinstances.iteritems()}
        if len(dcounts) == 0:
            logger.warning("Duplicates have been removed in a previous step!")
            continue
        max_inst_site = max(dcounts.iteritems(), key=operator.itemgetter(1))[0]
        logger.info("Keep site %s with %s instances (max)"
                    % (max_inst_site, dcounts[max_inst_site]))
        for site, inst in dinstances.iteritems():
            if site == max_inst_site:
                continue
            to_remove += inst

    if not dry:
        ut.remove_instances(temp_dir, to_remove)

    # set the threshold
    T = 70
    _, num_sites = th.threshold(temp_dir, th.PERC_THRESHOLD, cdf_file)

    # find instances over threshold
    to_remove, sites_removed = th.find_websites_with_less_instances(temp_dir, T)

    # find instances that exceed threshold
    to_remove += th.find_instances_until_threshold(temp_dir, T)

    if not dry:
        ut.remove_instances(temp_dir, to_remove)
    left_sites = int(num_sites) - int(sites_removed)
    logger.info("Websites after threshold: %s" % left_sites)
    logger.info("Instances per site after threshold: %s" % T)

    # move
    with open(url_file, 'w+') as fo:
        for f in listdir(temp_dir):
            fo.write(f + '\n')
            newfile = join(output, f)
            original = join(path, f, 'capture.tcpdump')
            logger.info("Moving original %s to %s" % (original, newfile))
            copy(original, newfile)

    # rename filenames to knn format
    knn_format(temp_dir, knn_dir, map_file)


def get_parser(config):
    parser = argparse.ArgumentParser(description='Run cleaning methods on data',
             formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config',
                        help="crawl configurations to clean.",
                        choices=config.sections(),
                        default="default")

    parser.add_argument('--dry', '-d',
                        action='store_true',
                        help='do not apply actions (for testing).')

    parser.add_argument('--loglevel', '-l',
                        type=str,
                        choices=[logging.getLevelName(logging.DEBUG),
                                 logging.getLevelName(logging.INFO)],
                        default=logging.getLevelName(logging.INFO),
                        help='logging level.')
    return parser


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
