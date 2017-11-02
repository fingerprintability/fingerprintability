# use this module for checking duplicate HSes
# should contain the list of duplicate onion addresses on each line
HS_DUPLICATES_FILE = "hs_duplicates.txt"


def get_duplicate_list(dup_file=HS_DUPLICATES_FILE):
    dup_sets_arr = []
    for line in open(dup_file).readlines():
        dup_sets_arr.append(set([hs for hs in line.split()]))
    return dup_sets_arr


def are_duplicates(site1, site2, dup_file=HS_DUPLICATES_FILE):
    for line in open(dup_file).readlines():
        if site1 in line:
            return site2 in line
    else:
        return False


def find_duplicates(site, dup_file=HS_DUPLICATES_FILE):
    duplicates = []
    for line in open(dup_file):
        if site in line:
            duplicates = line.strip().split()
            break
    return duplicates
