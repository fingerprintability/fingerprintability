
'''

A way to map the fingerprintability score we found from classifying the traffic trace features to high-level site
features that a designer of a hidden service page can use to make their site less-identifiable.

An instance is a visit.

We use the fingerprintability score as the label - which we get from analyzing the low level features.
Features - high level features - things that can be changed on a site.

@author bekah
'''

import datetime
import os
import sys

import numpy as np
import pylab as pl
from numpy import arange
from py2app.recipes import scipy
from scipy import stats, random
from sklearn import cross_validation
from sklearn.datasets import load_svmlight_file
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression,Ridge,LassoCV,Lasso
from sklearn.preprocessing import StandardScaler

feat_names = []
random.seed(14)
i_to_name = []

def save_insts_to_svm_file(instances, feat_file):
    print('saving insts to svm')
    dir = os.path.dirname(feat_file)
    if not os.path.exists(dir):
        print("making: "+dir)
        os.makedirs(dir)


    with open(feat_file, 'w') as f:
        first = True
        for url in instances.keys():
            inst = instances[url]
            if not first:
                f.write('\n')
            first = False
            f.write(str(inst.label)[0:5] + ' ')
            vals = inst.get_feature_vector()
            for j in range(1, len(vals)):
                val = str(vals[j]).strip()
                if val == '':
                    val = '0'
                f.write(str(j) + ':' + val + ' ')
            f.write('#'+str(url))
        f.close()


def average(vect):
    return np.average(vect)


def mode(vect):
    return stats.mode(vect)[0][0]


def median(vect):
    for i in range(0,len(vect)):
        try:
            vect[i] = float(vect[i])
        except:
            print('COULD NOT CONVERT: ' + str(vect))
            vect[i] = 0
    return np.median(vect)


def var(vect):
    for i in range(0, len(vect)):
        try:
            vect[i] = float(vect[i])
        except:
            print('COULD NOT CONVERT: ' + str(vect))
            vect[i] = 0
    return np.var(vect)


def num_unique(vect):
    return len(np.unique(vect))


class Instance:
    feature_names = []
    def __init__(self, label, url):
        self.label = float(label)
        self.url = url
        self.visits = []

    def add(self,line):
        self.visits.append(line)

    def set_feature_names(self, feature_names):
        self.feature_names = feature_names

    def get_feature_vector(self):
        vect = {}

        for visit in self.visits:
            for i in range(0,len(visit)):
                if i in vect:
                    t = vect[i]
                    t.append(visit[i])
                    vect[i] = t
                else:
                    t = []
                    t.append(visit[i])
                    vect[i] = t
        final_vect = []
        c = 0

        for i1 in range(0, len(vect)):
            feat = self.feature_names[i1]
            feat_type = feat.split('_')[0]
            if 'i' == feat_type:
                continue
            if not aggregated_feat_file:
                if 'mo' == feat_type:
                    final_vect.append(mode(vect[i1]))
                elif 'med' == feat_type:
                    final_vect.append(median(vect[i1]))
                elif 'made' == feat_type:
                    final_vect.append(mode(vect[i1]))
                elif 'var' == feat_type:
                    final_vect.append(var(vect[i1]))
                else:
                    print('unknown feat_type: '+feat_type + ':'+feat)
            else:
                final_vect.append(vect[i1][0])

        return final_vect

    def __str__(self):
        return str(self.label) + ',' + self.url +':' + str(self.get_feature_vector())


def parse_files(fability_score, feats_file):
    with open(fability_score, 'rU') as f1, open(feats_file, 'rU') as f2:
        feats = {}

        line = f1.next().split(',')
        for i in range(0,len(line)):
            try:
                line[i] = float(line[i])
            except ValueError:
                pass

        for i in range(0,len(line)):
            if str(type) in str(line[i]):
                index = i
                break

        for line in f1:
            line = line.split(',')
            url = line[0]
            try:
                label = line[index]
            except:
                print("Could not find index of metric: "+type)
                quit()
            try:
                feats[url] = Instance(float(label), url)
            except:
                feats[url] = Instance(float(0), url)

        feat_names = []
        firstline =f2.next()
        for f in firstline.replace("\"","").split('\t'):
            feat_names.append(f)

        for line in f2:
            line = line.strip().replace("\"","").split('\t')
            url = line[0]
            inst = feats[url]
            inst.set_feature_names(feat_names)
            inst.add(line)

    return feats


def save_original_to_svm_file(instances, feat_file):
    with open(feat_file, 'w') as f:
        for url in instances.keys():
            inst = instances[url]
            f.write(str(inst.label)[0:5] + ' ')
            visits = inst.visits
            for vals in visits:
                for j in range(0, len(vals)):
                    f.write(str(j) + ':' + str(vals[j]).strip() + ' ')
                f.write('#'+str(url)+'\n')
                f.write('\n')
        f.close()


def info_gain(labels, features, info_gain_res_file, type=''):
    print("**** Info Gain ****")
    print('Total features: %d' % len(features.toarray()[0]))
    total_score =0
    n_folds = 10
    n_labels = len(list(labels))
    kfolds = cross_validation.KFold(n_labels, n_folds=n_folds, shuffle=True)
    feat_imp = [0]*len(features.toarray()[0])

    for fold, (train, test) in enumerate(kfolds, start=1):
        print("Fold %d" % fold)
        X_train, X_test = features[train], features[test]
        y_train, y_test = labels[train], labels[test]
        # rf = RandomForestRegressor(n_estimators=250, random_state=26)
        rf = RandomForestClassifier(n_estimators=250, random_state=26)
        rf.fit(X_train, y_train)

        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]

        for f in range(X_train.shape[1]):
            feat_imp[indices[f]] = feat_imp[indices[f]] + importances[indices[f]]
        score = rf.score(X_test, y_test)
        # print("Fold Score: "+str(score))
        total_score += score
    # print("Total Score ("+str(type)+"): "+str(total_score/n_folds))

    add_colm_to_file(info_gain_res_file, [type]+feat_imp)


def add_colm_to_file(file, colm):
    if os.path.exists(file):
        cur = []
        with open(file, "rU") as f:
            for line in f:
                cur.append(line.strip())
            f.close()
        with open(file, "w") as f:
            for i in range(0, len(colm)):
                f.write(cur[i] + ","+str(colm[i])+"\n")
            f.close()
    else:
        with open(file, "w") as f:
            for i in range(0, len(colm)):
                f.write(str(colm[i]) + "\n")
            f.close()


def bucket_list(inf, outf): #hehe
    with open (inf, 'r') as f1, open(outf, 'w') as f2:
        for line in f1:
            n = line.split('#')[1].strip()
            l = line.split(' ')[0].strip()
            f2.write(l + "," + n + '\n')
        f1.close()
        f2.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Meta Learner')
    parser.add_argument('-f', '--featurefile',
                        help="Filename with site level features (csv)")
    parser.add_argument('-s', '--fabilityscorefile',
                        help="Filename with fability score (csv)")
    parser.add_argument('-n', '--columnheader',
                        help="Column header to look for the fability score csv file (e.g. ens_tpr)")
    parser.add_argument('-t', '--topnum', default=0.95,
                        help="The threshold for positive classes")
    parser.add_argument('-b', '--bottomnum', default=0.33,
                        help="The threshold for negative classes")
    parser.add_argument('-o', '--output',
                        help="Output file DIRECTORY")

    # Parse arguments
    args = parser.parse_args()

    fability_score = args.s
    feats_file = args.f
    type = args.n
    top_num = args.t
    bottom_num = args.b
    out = args.o

    svmfile = feats_file.replace('.csv','.svm')
    feats = parse_files(fability_score, feats_file)
    save_insts_to_svm_file(feats, svmfile)

    bucketed_svm = even_classes(svmfile, [bottom_num,top_num], type, change_labels=False)
    out = 'res/' + str(threshold) + '_hi_level_res_' + type + '.csv'

    info_gain(labels, features, os.path.join(out, 'feat_res.csv'), type)

    # saves info about the buckets if you want to
    # outf = 'bucket_lists/bucket_list_'+type+'.csv'
    # bucket_list(bucketed_svm, outf)
