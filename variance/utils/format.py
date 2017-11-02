#!/bin/python2
import sys
import argparse
from shutil import copy
from os.path import join, basename

import util as ut
import common as cm

# logger
import logging
logger = logging.getLogger(__name__)


def load_map(path):
    classes = []
    instances = {}
    for line in open(path):
        onion_fname, knn_id = line.strip().split()
        instances[knn_id] = onion_fname
        class_index = int(knn_id.split('-')[0])
        onion_url = onion_fname.split('_')[1]
        if onion_url not in classes:
            classes.append(onion_url)
            assert(classes.index(onion_url) == class_index)
    return classes, instances


def reformat(path, output, map_file, dry=False):
    classes, instances = load_map(map_file)
    with open(output, 'w') as fo:
        for line in open(path):
            new_attrib = []
            knn_instance, guessed = line.strip().split('\t')[0:2]
            prob_tups = line.strip().split('\t')[2:]
            new_attrib.append(instances[knn_instance])
            if guessed == 'NA':
                new_attrib.append(guessed)
            else:
                new_attrib.append(classes[int(guessed)])
            for prob_tup in prob_tups:
                c, p = prob_tup.split(',')
                new_attrib.append(','.join([classes[int(c)], p]))
            if not dry:
                fo.write('\t'.join(new_attrib) + '\n')


def knn_format(path, output, map_file, dry=False):
    etc_mapfile = join(cm.ETC_DIR, basename(map_file))
    with open(map_file, 'w') as fout, open(etc_mapfile, 'w') as fo_etc:
        for i, (_, instances) in enumerate(ut.walk_instances(path)):
            onion = basename(instances[0]).split('_')[1]
            fo_etc.write('%s %s\n' % (str(i), onion))
            for j, instance in enumerate(instances):
                new_fname = '%s-%s' % (i, j)
                new_path = join(output, new_fname)
                logger.info('renaming %s to %s' % (instance, new_path))
                fout.write('%s:%s\n' % (basename(instance), new_fname))
                if not dry:
                    copy(instance, new_path)
