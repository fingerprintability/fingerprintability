from os import makedirs, walk
from os.path import exists, join
import fnmatch


def gen_find_files(filepat, top):
    """ http://www.dabeaz.com/generators/
    Return filenames that matches the given pattern under a given
    directory
    """
    for path, _, filelist in walk(top):
        for name in fnmatch.filter(filelist, filepat):
            yield join(path, name)


def ensure_dir_exists(dir_path):
    """Create a directory if it doesn't exist."""
    if not exists(dir_path):
        makedirs(dir_path)
    return dir_path


def write_to_file(file_path, data):
    """Write data to file and close."""
    f = open(file_path, 'w')
    f.write(data)
    f.close()


def append_to_file(file_path, data):
    """Append to file and close."""
    with open(file_path, 'a') as f:
        f.write(data)
