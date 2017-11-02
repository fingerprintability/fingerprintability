from __future__ import division
import networkx as nx
from os.path import join, realpath, dirname
from _collections import defaultdict
from variance.utils.report import get_hi_level_feat_as_dict


def dict_sorted_by_val(d):
    return sorted(((v, k) for k, v in d.iteritems()), reverse=True)


def reverse_conf_pair(conf_pair):
    "Reverse a confusion pair in the form a.onion_b.onion to b.onion_a.onion"
    site_a, site_b = conf_pair.split("_")
    return "%s_%s" % (site_b, site_a)


def count_symmetric_misclassifications(conf_dict, classifer_name):
    rev_confusions = defaultdict(int)
    for conf_pair in conf_dict.keys():
        rev_conf_pair = reverse_conf_pair(conf_pair)
        # print conf_pair, freq, rev_conf_pair
        if rev_conf_pair in conf_dict:
            rev_confusions[rev_conf_pair] = conf_dict[rev_conf_pair]
    print "********************************************"
    print "\nResults for", classifer_name
    print "********************************************"
    distinct_conf_pairs = len(conf_dict)
    total_confusions = sum(conf_dict.itervalues())
    distinct_sym_conf_pairs = len(rev_confusions)
    total_sym_confusions = sum(rev_confusions.itervalues())
    print "Misclassifications: %s distinct, %s total" % (distinct_conf_pairs,
                                                         total_confusions)
    print "Symmetric misclassifications %s distinct, %s total" %\
        (distinct_sym_conf_pairs, total_sym_confusions)
    print "Ratio of distinct pairs with symmetric misclassification: ",\
        (distinct_sym_conf_pairs / distinct_conf_pairs)


def add_site_size_to_graph(G, feat_name="med_total_http_download"):
    feat_dict = get_hi_level_feat_as_dict(feat_name, G.nodes_iter())
    # convert to integer, otherwise cannot write the graph file
    feat_dict_int = {k: int(v) for k, v in feat_dict.iteritems()}
    nx.set_node_attributes(G, feat_name, feat_dict_int)


def build_confusion_graph(conf_dict):
    G = nx.DiGraph()
    for conf_pair, freq in conf_dict.iteritems():
        true_label, pred_label = conf_pair.split("_")
        G.add_edge(true_label, pred_label, weight=freq)
    add_site_size_to_graph(G)  # add med_total_http_download as node attribute
    return G


def read_confusion_matrix(conf_csv):
    cumul_confusions = defaultdict(int)
    knn_confusions = defaultdict(int)
    kfp_confusions = defaultdict(int)
    skipped_header = False
    for l in open(conf_csv):
        if not skipped_header:
            skipped_header = True
            continue
        fname, pred_knn, pred_kfp, pred_cumul = l.rstrip().split(",")
        true_label = fname.split("_")[1]
        if true_label != pred_cumul:
            conf_pair_str = "%s_%s" % (true_label, pred_cumul)
            cumul_confusions[conf_pair_str] += 1
        if true_label != pred_knn:
            conf_pair_str = "%s_%s" % (true_label, pred_knn)
            knn_confusions[conf_pair_str] += 1
        if true_label != pred_kfp:
            conf_pair_str = "%s_%s" % (true_label, pred_kfp)
            kfp_confusions[conf_pair_str] += 1
    return cumul_confusions, knn_confusions, kfp_confusions


def save_confusion_graph(G, classifer_name):
    d = nx.degree(G)
    nx.draw(G, nodelist=d.keys(), node_size=[v * 100 for v in d.values()],
            pos=nx.spring_layout(G))
    nx.write_gexf(G, "%s_graph.gexf" % classifer_name)


def analyze_confusion_graph(G):
    num_sites_misclassified_once_or_more = 0
    num_sites_with_symmetric_misclassification = 0
    num_sites_never_misclassified = 0
    num_sites_never_confused_as_another_site = 0

    for node, adj_dict in G.adjacency_iter():
        if not len(adj_dict):  # no outgoing edges
            num_sites_never_misclassified += 1
            # print "Never misclassified?", node
            continue
        num_sites_misclassified_once_or_more += 1
        if G.in_degree(node) == 0:
            # print "No false positive", node
            num_sites_never_confused_as_another_site += 1
        for prediction in adj_dict.keys():
            if not G.has_edge(prediction, node):
                # print "confusion is not symmetric", prediction, node
                break
        else:
            num_sites_with_symmetric_misclassification += 1
            # print node, "confusion is symmetric", adj_dict.keys()
    print ("Site A is misclassified as other sites and other sites are "
           "misclassified as Site A: %s / %s (%0.1f%%)" %
           (num_sites_with_symmetric_misclassification,
            num_sites_misclassified_once_or_more,
            (100*(num_sites_with_symmetric_misclassification /
                  num_sites_misclassified_once_or_more))))

    print ("Asymmetrical: One or more sites are misclassified as Site A, but A"
           " is consistently classified as A: %s" %
           num_sites_never_misclassified)

    print ("Asymmetrical: Site A is misclassified as one or more other sites "
           "but other sites are never misclassified as A: %s" %
           num_sites_never_confused_as_another_site)


def analyze_confusions(conf_dict, classifer_name):
    count_symmetric_misclassifications(conf_dict, classifer_name)
    G = build_confusion_graph(conf_dict)
    analyze_confusion_graph(G)
    save_confusion_graph(G, classifer_name)


def check_symmetry(conf_csv):
    cumul_confusions, knn_confusions, kfp_confusions =\
        read_confusion_matrix(conf_csv)
    analyze_confusions(cumul_confusions, "cumul")
    analyze_confusions(kfp_confusions, "kfp")
    analyze_confusions(knn_confusions, "knn")


if __name__ == '__main__':
    cd = dirname
    all_predictions_csv = join(cd(cd(cd(realpath(__file__)))),
                               "data", "confusion", "all_predictions.csv")
    check_symmetry(all_predictions_csv)
