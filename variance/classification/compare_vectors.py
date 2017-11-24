'''
@author bekah
'''
import argparse
import math,sys, os
from sklearn import svm

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def class_variance(file, outfile):

    with open(file, "r") as f, open(outfile.replace(".txt","_intra.txt"), "w") as o1, open(outfile.replace(".txt","_inter.txt"), "w") as o2:
        # all classes holds the features for each class separately
        # number of classes x number of features x number of instances
        lines = []
        num_classes_finder = {}
        max_class_num = 0
        for line in f:
            lines.append(line.split("#")[0].strip())
            class_label = int(line.split(" ")[0])
            num_classes_finder[class_label] = 1
            if max_class_num < class_label:
                max_class_num = class_label
        f.close()

        num_classes = len(num_classes_finder)

        num_features = len(lines[0].split(" "))

        # num_instances = len(lines)

        print("num classes: "+str(num_classes))
        all_classes = []
        for i in range(0,max_class_num+1):
            # all features holds the features for each class
            all_features = []

            for j in range(0, num_features-1):
                all_features.append([])
            all_classes.append(all_features)

        for line in lines:
            split = line.split(" ")
            label = int(split[0])

            all_features = all_classes[label]
            feat_vals = split[1:len(split)]

            for i in range(0, len(feat_vals)):
                temp = all_features[i]

                temp.append(float(feat_vals[i].split(":")[1]))

                all_features[i] = temp

            all_classes[label] = all_features

        print("Intra-Class Variance")

        res = [long(0.0)]*len(all_classes[0])
        for k in range(0, len(all_classes)):
            all_features = all_classes[k]
            if all_features[0] == []:
                continue

            for i in range(0,len(res)):
                try:
                    res[i] += float(np.var(all_features[i]))
                except:
                    print("error: "+str(all_features[i]))
                    print("class: "+str(k))
                    print("inst: "+str(i))
                    quit()
        tot = 0
        n = 0
        for r in res:
            v = r/float(len(all_features))
            tot += v
            n += 1
            o1.write(str(v) + "\n")
        print("intra-class: "+str(tot/n))
        intra = tot/n

        print("Inter-Class Variance")

        # here we want to average the features for each class
        # then determine the variance between the features across classess

        res = [0.0] * num_features
        feats = []
        for i in range(0,num_features):
            feats.append([])
        for cur_class in range(0,len(all_classes)):
            all_features = all_classes[cur_class]
            # averages_for_cur_class = [0.0] * len(all_features)
            for i in range(0, len(all_features)):

                this_feature = all_features[i]
                if len(this_feature) == 0:
                    continue
                sum = 0
                for feature_val in this_feature:
                    sum += feature_val
                average = sum / len(this_feature)
                temp = feats[i]
                temp.append(average)
        for i in range(0,len(feats)-1):
            res[i] += np.var(feats[i])


        tot = 0
        n = 0
        for r in res:
            tot += r
            n += 1
            o2.write(str(r) + "\n")
        print("inter-class: " + str(tot / n))
        inter = tot/n
        f.close()
        o1.close()
        o2.close()
        return intra, inter



if __name__ == '__main__':

    # Parse arguments
    # note, you should scale/norm the features outside of this
    parser = argparse.ArgumentParser(description='Compute Feature Variances')
    parser.add_argument('-f', '--featurefile',
                        help="Filename with feature vectors (.svm file)")
    parser.add_argument('-o', '--output',
                        help="Output file location")


    # Parse arguments
    args = parser.parse_args()
    class_variance(args.f, args.o)



