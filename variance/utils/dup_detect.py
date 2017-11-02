# Use variance.utils.duplicates.py to check the duplicate HSes
# This file is to be run once, followed by a manual analysis
import os
from os.path import join
from variance.utils import file_utils as fu
from variance.utils import simutils

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_BASE_DIR = join(THIS_DIR, "reports")


def get_closest_simhash_dist(sim_hash_set1, sim_hash_set2):
    min_dist = 64
    for hash1 in sim_hash_set1:
        for hash2 in sim_hash_set2:
            bit_dist = simutils.diff_bits(hash1, hash2)
            if bit_dist < min_dist:
                # print "Found a lower dist", bit_dist, min_dist
                min_dist = bit_dist
    # print "Returning", min_dist
    return min_dist


def check_duplicate_candidates(domain_pair, domain_infos):
    (domain1, domain2) = domain_pair
    (domain_fail_count, domain_screenshot_hashes,
        domain_html_src_hashes, domain_html_src_simhashes, domain_titles, domain_req_paths) = domain_infos

    screenshot_hashes_d1 = domain_screenshot_hashes[domain1]
    screenshot_hashes_d2 = domain_screenshot_hashes[domain2]

    html_src_hashes_d1 = domain_html_src_hashes[domain1]
    html_src_hashes_d2 = domain_html_src_hashes[domain2]

    html_src_simhashes_d1 = domain_html_src_simhashes[domain1]
    html_src_simhashes_d2 = domain_html_src_simhashes[domain2]

    page_titles_d1 = domain_titles[domain1]
    page_titles_d2 = domain_titles[domain2]

    req_paths_d1 = domain_req_paths[domain1]
    req_paths_d2 = domain_req_paths[domain2]

    img_hash_match = 0
    html_src_hash_match = 0
    html_src_simhash_match = 0
    title_match = 0
    req_path_match = 0
    matching_titles = ""
    if screenshot_hashes_d1.intersection(screenshot_hashes_d2):
        img_hash_match = 1
    if html_src_hashes_d1.intersection(html_src_hashes_d2):
        html_src_hash_match = 1
    min_simhash_dist = 64
    if html_src_simhashes_d1.intersection(html_src_simhashes_d2):
        html_src_simhash_match = 1
    else:
        min_simhash_dist = get_closest_simhash_dist(html_src_simhashes_d1,
                                                    html_src_simhashes_d2)
        if min_simhash_dist <= 3:
            # print "SIMHASH dist", domain1, domain2, min_dist
            html_src_simhash_match = 1

    if page_titles_d1.intersection(page_titles_d2):
        title_match = 1
        matching_titles = " AND ".join(page_titles_d1.
                                       intersection(page_titles_d2))
    if title_match and not (html_src_simhash_match or
                            html_src_hash_match or img_hash_match):
        # print "Title match but not html hash match", domain1, domain2, page_titles_d1, page_titles_d2
        pass
    jac = simutils.jaccard_index(req_paths_d1, req_paths_d2)
    if jac > 0.8:
        req_path_match = 1
        if jac < 1:
            print "JAC", jac, domain1, domain2, req_paths_d1, req_paths_d2

    if img_hash_match or html_src_hash_match or html_src_simhash_match or (title_match and req_path_match):
        # print "Only title matches", domain1, domain2, "IMG:", img_hash_match, "HTML:", html_src_hash_match, "HTML_simhash:", html_src_simhash_match, "req_path_match:", ut.jaccard_index(req_paths_d1, req_paths_d2), "Title:", title_match, matching_titles, min_dist
        print "MATCH", domain1, domain2, "IMG:", img_hash_match, "HTML:", html_src_hash_match,
        print "HTML_simhash:", html_src_simhash_match, "simhash min dist", min_simhash_dist, "req_path_match:",
        print simutils.jaccard_index(req_paths_d1, req_paths_d2), min_simhash_dist, "Title:", title_match, matching_titles
        return True
    return False


def find_duplicates(crawl_dir, crawled_domains_set, domain_infos):
    dup_set = set()
    dup_pairs_set = set()
    domain_pairs = []
    crawled_domains = list(crawled_domains_set)
    n_crawled_domains = len(crawled_domains)
    for domain1_idx, domain1 in enumerate(crawled_domains):
        for domain2 in crawled_domains[domain1_idx:]:
            if domain1 != domain2:
                domain_pairs.append((domain1, domain2))
    assert len(domain_pairs) == (n_crawled_domains * (n_crawled_domains - 1)) / 2
    for pair in domain_pairs:
        if check_duplicate_candidates(pair, domain_infos):
            dup_set.add(pair[0])
            dup_set.add(pair[1])
            dup_pairs_set.add(pair)
    print "Total websites", len(dup_set) 
    merged_clusters = get_dup_clusters(dup_pairs_set)
    total_domains_in_clusters = 0
    domains_in_clusters = set()
    for idx, cluster_ in enumerate(merged_clusters):
        # logging.debug("%d %s %s" % (idx, len(cluster_), cluster_))
        print("%d %s %s" % (idx, len(cluster_), cluster_))
        domains_in_clusters.update(cluster_)
        total_domains_in_clusters += len(cluster_)
    generate_dup_report(crawl_dir, domain_infos, merged_clusters)
    assert total_domains_in_clusters == len(dup_set)
    assert total_domains_in_clusters == len(domains_in_clusters)


def merge_clusters(clusters):
    while 1:
        clusters2 = merge(clusters)
        if clusters2 == clusters:
            return clusters
        else:
            clusters = clusters2


def merge(clusters):
    clusters2 = clusters[:]
    for idx, cluster in enumerate(clusters):
        for cluster2 in clusters[idx+1:]:
            if cluster.intersection(cluster2):
                clusters2[idx].update(cluster2)
                clusters2.remove(cluster2)
                return clusters2
    return clusters2


def get_dup_clusters(dup_pairs_set):
    fu.ensure_dir_exists(REPORTS_BASE_DIR)
    clusters = []
    for pair in dup_pairs_set:
        clusters.append(set(pair))
    return merge_clusters(clusters)


def generate_dup_report(crawl_dir, domain_infos, clusters):
    for idx, cluster in enumerate(clusters):
        print idx, cluster


if __name__ == '__main__':
    pass
