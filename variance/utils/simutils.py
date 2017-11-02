from difflib import SequenceMatcher
import simhash
from itertools import izip
from PIL import Image


def diff_bits(a, b):
    a = int(a)
    b = int(b)
    a = a ^ b
    count = 0
    while a:
        count += 1
        a = a & (a - 1)
    return count


def get_simhash_similarity(str1, str2):
    hash1 = simhash.hash(str1)
    hash2 = simhash.hash(str2)
    # hash1 = simhash(str1)
    # hash2 = simhash(str2)
    # return hash1.similarity(hash2)
    return diff_bits(hash1, hash2)


def get_line_based_diff(str1, str2):
    s = SequenceMatcher(None, str1, str2)
    return s.ratio()


def jaccard_index(set1, set2):
    union_set = set1.union(set2)
    if not union_set:
        return 0
    else:
        return float(len(set1.intersection(set2))) / len(union_set)


def get_img_diff(img1, img2):
    i1 = Image.open(img1)
    i2 = Image.open(img2)
    assert i1.mode == i2.mode, "Different kinds of images."
    assert i1.size == i2.size, "Different sizes."

    pairs = izip(i1.getdata(), i2.getdata())
    if len(i1.getbands()) == 1:
        # for gray-scale jpegs
        dif = sum(abs(p1-p2) for p1, p2 in pairs)
    else:
        dif = sum(abs(c1-c2) for p1, p2 in pairs for c1, c2 in zip(p1, p2))

    ncomponents = i1.size[0] * i1.size[1] * 3
    print "Difference (percentage):", (dif / 255.0 * 100) / ncomponents
    return (dif / 255.0 * 100) / ncomponents


if __name__ == '__main__':
    pass
