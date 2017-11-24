'''
classifier model funct - pred_proba



Basic code that takes a feature file and performs cross-validation on the
features using lib_svm


Examples of how to use it:
    - python classification.py linear - python classification.py rbf
    - python classification.py -f <featureset_filename> linear
    - python classification.py -g rbf

@author bekah
'''
from __future__ import print_function, division

import argparse
import multiprocessing as mp
import random
from collections import defaultdict

import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

# threshold to remove classes with low probability
THRESHOLD = 0.0001

# num folds in cross-validation
NUM_FOLDS = 10

# number of cores
NUM_PROCS = int(mp.cpu_count())

random.seed(1)

cfg = {"VERBOSE": False}  # lazy... make a dict so that we don't need global to set it


def print_log(log_str, log_file=None):
    if cfg["VERBOSE"]:
        if log_file:
            with open(log_file, 'a') as f:
                f.write("\n%s" % log_str)
        else:
            print(log_str)


def setParams(X_train, y_train, kernel):
    n_jobs = NUM_PROCS  # Use half of the CPUs for parallel GridSearch
    gamma = []
    for g in xrange(-15, 4, 4):
        gamma.append(2 ** g)

    C = []
    for g in xrange(-5, 16, 4):
        C.append(2 ** g)

    print_log("grid search: ")

    tuned_parameters = {'kernel': [kernel], 'C': C}
    print_log(tuned_parameters)
    if kernel == "rbf":
        tuned_parameters["gamma"] = gamma

    clf = GridSearchCV(svm.SVC(C=1), tuned_parameters, n_jobs=n_jobs, cv=5)
    clf.fit(X_train, y_train)

    print_log("best_params: ")
    print_log(clf.best_params_)
    return clf.best_params_

    # for params, mean_score, scores in clf.grid_scores_:
    #     print_log("%0.3f (+/-%0.03f) for %r"
    #        % (mean_score, scores.std() * 2, params))

    # y_true, y_pred = y_test, clf.predict(X_test)
    # print_log(classification_report(y_true, y_pred))


def test_with_probabilities(clf, test):
    conf_matrix = []
    tp = 0.0
    for i, t in enumerate(test):
        if i % 100 == 0 or i == len(test):
            print(mp.current_process(), i, len(test))
        proba = clf.predict_proba(np.array(t.features).reshape(1, -1))[0]
        # predicted = clf.predict(np.array(t.features).reshape(1, -1))[0]
        tup_probs = zip(clf.classes_, proba)
        sorted_tups = sorted(tup_probs, key=lambda tup: tup[1])[::-1]
        guess_label = indices[int(sorted_tups[0][0])]
        true_label = t.file_name.split('_')[1]
        if true_label == guess_label:
            tp += 1.0
        probs_str = '\t'.join(['%s,%s' % (indices[int(k)], v)
                               for k, v in sorted_tups if v > THRESHOLD])
        str_line = '\t'.join([t.file_name, guess_label, probs_str])
        conf_matrix.append(str_line)
    print("TPR=", tp / i, tp, i)
    return conf_matrix


def labels(l):
    return [x.label for x in l]


def features(l):
    return [x.features for x in l]


def rf(dataset):
    '''Return confusion matrix for the xval of the random forest classifier.'''
    train, test = dataset[:2]
    clf = RandomForestClassifier(n_jobs=4, n_estimators=1000, oob_score=True)
    clf.fit(features(train), labels(train))
    return test_with_probabilities(clf, test)


def svm_rbf(dataset):
    '''Return confusion matrix for the x-val of the svm-rbf classifier.'''
    train, test = dataset[:2]
    chosen_params = setParams(features(train), labels(train), "rbf")
    clf = svm.SVC(kernel='rbf', gamma=chosen_params["gamma"],
                  C=chosen_params["C"], probability=True)
    clf.fit(features(train), labels(train))
    return test_with_probabilities(clf, test)


def svm_linear(dataset):
    '''Return confusion matrix for the x-val of the svm-linear classifier.'''
    train, test = dataset[:2]
    chosen_params = setParams(features(train), labels(train), "linear")
    clf = svm.SVC(kernel='linear', C=chosen_params["C"])
    clf.fit(features(train), labels(train))
    return test_with_probabilities(clf, test)


def find_missclassifications(clf, X_test, y_test):
    predicted = clf.predict(X_test)
    real_labels = []
    print_log("real,predicted")
    for i in range(0, len(y_test)):
        predicted_label = predicted[i]
        real_label = y_test[i]
        real_labels.append(real_label)
        if predicted_label != real_label:
            print_log("%s %s" % (real_label, predicted_label))
    # conf_mat = confusion_matrix(real_labels, predicted)
    return real_labels, predicted


def cross_validate(svm_fun, labels, features, output):

    print('Total features: ', len(features.toarray()[0]))

    total = 0
    n_labels = len(list(labels))
    kfolds = cross_validation.KFold(n_labels, n_folds=10, shuffle=True)
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    all_truelabels, all_predictions = [], []

    confusion_set = set()
    for fold, (train, test) in enumerate(kfolds, start=1):
        print_log("*** Fold %d ***" % fold)
        X_train, X_test = features[train], features[test]
        y_train, y_test = labels[train], labels[test]
        score, truelabels, predictions = svm_fun(X_train, y_train,
                                                 X_test, y_test)
        all_truelabels += list(truelabels)
        all_predictions += list(predictions)
        total += score
        for idx, (_truelabel, _predicted) in enumerate(zip(truelabels,
                                                           predictions)):
            truelabel = int(_truelabel)
            predicted = int(_predicted)
            if truelabel == predicted:
                tp[truelabel] += 1
            if truelabel != predicted:
                confusion_set.add((fold, truelabel, predicted, test[idx]))
                fn[truelabel] += 1
                fp[predicted] += 1

    analyze_confusion(confusion_set, features, tp, fp, fn)
    return float(total) / float(fold), all_truelabels, all_predictions


def analyze_confusion(confusion_set, features, tp, fp, fn):
    TOTAL_N_SITES = 100
    site_nums = set(range(1, TOTAL_N_SITES+1))
    conf_log = "confusion.log"  # we append to this file
    print_log("fold feat_index truelabel predicted", conf_log)
    for fold, truelabel, predicted, test_vector_idx in confusion_set:
        print_log("%s %s %s %s" % (fold, test_vector_idx, truelabel, predicted),
                  log_file=conf_log)
        site_nums.discard(truelabel)
        site_nums.discard(predicted)
    print_log("Never misclassified: %s " % site_nums, log_file=conf_log)
    print_log("False Negatives", log_file=conf_log)
    for site, false_negatives in fn.iteritems():
        if false_negatives:
            print_log("%s %s" % (site, false_negatives), log_file=conf_log)

    print_log("False Positives", log_file=conf_log)
    for site, false_positives in fp.iteritems():
        if false_positives:
            print_log("%s %s" % (site, false_positives), log_file=conf_log)


class Sample:

    def __init__(self, label, file_name, line):
        self.label = int(label)
        self.file_name = file_name
        line = line.split("#")[0]
        self.line = line
        self.features = [float(f.split(':')[1]) for f in line.split()[1:-2]]

    def __str__(self):
        return str(self.label) + "," + self.file_name


class Class:

    def __init__(self, label):
        self.label = int(label)
        self.samples = []

    def add_sample(self, sample):
        self.samples.append(sample)
        random.shuffle(self.samples)

    def __str__(self):
        return str(self.label) + ". samples: " + str(len(self.samples))


def parse_classes(svm_file):
    """Return list of samples by parsing an svm file."""
    classes = []
    for line in open(svm_file):
        label = line.split()[0]
        name = line.split("#")[1].strip()

        s = Sample(label, name, line)

        # find class
        for c in classes:
            if c.label == int(label):
                break
        else:
            c = Class(label)
            classes.append(c)
        c.add_sample(s)

    return classes


def chunk(l, k, n):
    """Return k-th n-sized chunk from l."""
    for j, i in enumerate(xrange(0, len(l), n)):
        if j == k:
            return l[i:i + n], l[:i] + l[i+n:]


def it_xval(classes, num_folds=NUM_FOLDS):
    num_samples = len(classes[0].samples)
    num_test = int(num_samples / num_folds)
    for i in xrange(num_folds):
        train, test = [], []
        for c in classes:
            test_chunk, train_chunk = chunk(c.samples, i, num_test)
            train.extend(train_chunk)
            test.extend(test_chunk)
        yield train, test


def custom_cross_validation(svm_file, classifier, outfile):
    """Custom cross-validation."""
    # read in svm file to get names of instances
    classes = parse_classes(svm_file)

    # SEQUENTIAL
    conf_mat_list = []
    for fold in it_xval(classes):
        conf_mat_list.append(classifier(fold))

    # PARALLEL
    # multiprocess by xval folds
    # p = mp.Pool(int(NUM_FOLDS / 2))
    # p = mp.Pool(1)
    # conf_mat_list = p.map(classifier, it_xval(classes))
    # p.close()
    # p.join()

    # concatenate output_files
    with open(outfile, "w") as f:
        for line in sum(conf_mat_list, []):
            f.write(line + '\n')


def main():
    # python classification.py -f features.svm -g -v -n

    # Parse arguments
    parser = argparse.ArgumentParser(description='Run libSVM on a feature set.')
    parser.add_argument('-f', '--filename',
                        help="Filename with feature vectors.")
    parser.add_argument('-o', '--output',
                        help="output log location")
    parser.add_argument('-i', '--index',
                        help="index of instance names file location")
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=True,
                        help="whether to verbose.")
    parser.add_argument("--classifier_type", default="linear",
                        choices=['rbf', 'linear', 'rf'],
                        help='choose one of rbf linear or rf')

    # normaization is done in scale.py. This is just classification.
    # deleted grid search option. We should ALWAYS grid search

    # Parse arguments
    args = parser.parse_args()

    if args.classifier_type == 'rbf':
        classifier_f = svm_rbf
    elif args.classifier_type == 'linear':
        classifier_f = svm_linear
    elif args.classifier_type == 'rf':
        classifier_f = rf

    # whether to print or not
    cfg["VERBOSE"] = args.verbose

    features, labels = load_svmlight_file(args.filename)

    # load indices
    global indices
    indices = [l.strip().split(':')[1] for l in open(args.index)]

    custom_cross_validation(args.filename, classifier_f, args.output)

    # avg_score, true_labels, predicted_labels = cross_validate(classifier_f,
    # labels, features, args.output)
    # print("AVG SCORE=", avg_score)


def lazy_remove_features(file_path, out, i=None, j=None):
    '''
    Removes the i-jth featres from the feature file.
    Used to remove the non-interpolated features from Panchenko's feature files.
    To remove all features above/below k, set i/j=None
    Lazy because it doesn't change the index numbers.
    '''
    with open(file_path) as f, open(out) as o:
        for line in f:
            line = line.split(" ")
            if i is None:
                i = 1
            if j is None:
                j = len(line)
            interp_feats = str(line[i+1:j]).replace("[", "").replace("]", "")\
                .replace(",", "").replace("\'", "").replace("\\n", "")
            o.write(line[0]+" "+interp_feats+"\n")


if __name__ == '__main__':
    main()
