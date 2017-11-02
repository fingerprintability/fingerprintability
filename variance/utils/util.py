from sys import stdout
from glob import glob
from os import listdir, remove
from os.path import join, isfile
import contextlib

import common as cm

# logger
import logging
logger = logging.getLogger(__name__)


def get_incoming(fpath):
    for line in open(fpath):
        size = int(line.strip().split(cm.FIELD_SEP)[1])
        if size < 0:
            yield size * cm.CELL_SIZE


def walk_sites(dpath):
    """Iterate over sites that have instances in `dpath`."""
    sites = set()
    for f in listdir(dpath):
        site = f.split('_')[1]
        if site in sites:
            continue
        yield site
        sites.add(site)


def get_fnames_by_onion(dpath, onion):
    return [x for x in glob(join(dpath, '*_%s_*' % onion))]


def walk_instances(dpath):
    """Yield site and all its instances found in `dpath`."""
    for site in walk_sites(dpath):
        yield site, get_fnames_by_onion(dpath, site)


def get_num_instances(dpath, site):
    return len([x for x in glob(join(dpath, '*_%s_*' % site))])


def instances(dpath):
    return {site: instances for site, instances in walk_instances(dpath)}


def instance_counts(dpath):
    """Return instance count for each website in `dpath`."""
    return {k: len(v) for k, v in instances(dpath).iteritems()}


def remove_instances(dpath, files, dry=False):
    """Remove a list of instances from `dpath`."""
    for f in files:
        fpath = join(dpath, f)
        if not isfile(fpath):
            logger.info("%s cannot be removed because doesn't exist" % fpath)
            continue
        if not dry:
            remove(fpath)
    return len(files)


@contextlib.contextmanager
def wopen(filename=None):
    if filename and filename != '-':
        fh = open(filename, 'wb')
    else:
        fh = stdout
    try:
        yield fh
    finally:
        if fh is not stdout:
            fh.close()
