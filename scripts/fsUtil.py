# coding: utf-8
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from future import standard_library

import os
import sys
import time
import gc

from pyHSICLasso import HSICLasso

import scipy.io as sio
import pandas as pd
import numpy as np

standard_library.install_aliases()

def saveDF(X, Y, Xcolnames, filename):
    df = pd.DataFrame(data=X, columns=Xcolnames)
    df.insert(loc=0, column="class", value=Y)
    df.to_csv(filename, index=False, header=True)

def saveFeatures(hsic_lasso, filename):
    columns = ["FeatureIdx", "FeatureName", "Importance_Score", "T1RF", "T1RS", "T2RF", "T2RS", "T3RF", "T3RS", "T4RF", "T4RS", "T5RF", "T5RS"]
    data = []
    featScores = hsic_lasso.get_index_score()
    featScores = featScores / featScores[0] # normalize by the most important feature
    featIndicesOri = hsic_lasso.get_index() # indice in the original dataset
    featNames = hsic_lasso.get_features()
    numFeat = len(featIndicesOri)

    for featIdxExt, featIdxOri, featName, featScore in zip(
            range(numFeat), featIndicesOri, featNames, featScores):
        record = [featIdxOri, featName, featScore]
        neighbors_names = hsic_lasso.get_features_neighbors(feat_index=featIdxExt)
        neighbors_scores = hsic_lasso.get_index_neighbors_score(feat_index=featIdxExt)
        for neighbor_name, neighbor_score in zip(neighbors_names, neighbors_scores):
            record.extend([neighbor_name, neighbor_score])

        data.append(record)
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(filename, index=False, header=True)


def getNumClass(labelName, value):
    if labelName == "Diagnosis":
        if value == "Cancer":
            return 1
        else:
            return -1
    elif labelName =="Class":
        if value == "Young Cancer":
            return 1
        elif value == "Young Clear":
            return 2
        elif value == "Young Polyp":
            return 3
        elif value == "Old Cancer":
            return 4
        elif value == "Old Clear":
            return 5
        else: # Old Polyp
            return 6
    elif labelName =="CancerAge":
        if value == "Young Cancer":
            return 1
        elif value == "Young Clear" or value == "Young Polyp":
            return 2
        elif value == "Old Cancer":
            return 3
        elif value == "Old Clear" or value == "Old Polyp":
            return 4
        else: # throw an error
            raise "Non-known value {} for CancerAge classification".format(value)
    elif labelName == "AgeGroup":
        if value == "Young":
            return 1
        else: # Old
            return -1
    else:
        raise "Unsupported task {}".format(labelName)

def dataProcess(filepath, labelName, selectAgeGender=False):
    #columns = ["Sample Title", "Age Group", "Diagnosis", "Class", "Age", "Gender"]
    #indices = [0, 3, 4, 5, 9, 10]
    #columns = ['class']
    if labelName == "Diagnosis": # Cancer, No-Cancer
        columns = ['class', 'AgeGroup']
    elif labelName == 'AgeGroup':
        columns = ['AgeGroup', 'class']
    else: # Class, CancerAge
        columns = ['class']	
    if selectAgeGender:
        columns.extend(["Age", "Gender"])

    records = []
    start_time = time.time()
    with open(filepath, encoding='ISO-8859-1') as fp: # UTF-8
        print("Processing the table header")
        line = fp.readline().strip()
        parts = line.split("\t")
        columns.extend(parts[12:])
        print("{} columns".format(len(columns)))

        print("Processing each record")
        line = fp.readline().strip()
        while(line):
            parts = line.split("\t")
            record = []
            
            if labelName == "Diagnosis": # Cancer, No-Cancer
                record = [getNumClass(labelName, parts[4]), parts[3]]
            elif labelName == "Class" or labelName == "CancerAge":
                # Class:
                #   Young Cancer, Young Clear, Young Polyp
                #   Old Cancer, Old Clear, Old Polyp
                # CancerAge:
                #   Young Cancer, {Young Clear, Young Polyp}
                #   Old Cancer, {Old Clear, Old Polyp}
                record = [getNumClass(labelName, parts[5])]
            else: # Age Group
                record = [getNumClass(labelName, parts[3]), parts[4]]
            if selectAgeGender:
                record.extend([float(parts[9]), parts[10]])

            record.extend([float(x) for x in parts[12:]])
            records.append(record)
            if len(records) % 25 == 0:
                print("{} records, elapsed time (s): {}.".format(
                    len(records), time.time()-start_time))

            line = fp.readline().strip()
            
        print("{} records, elapsed time (s): {}.".format(
            len(records), time.time()-start_time))
    #print("record cols: {} vs columns: {}".format(len(records[0]), len(columns)))

    df = pd.DataFrame(data=records, columns=columns)
    del records
    del columns
    return df

def extractFeatures(X, Y, num_feat_list, featname, resultDir, discrete_x=False):
    hsic_lasso = HSICLasso()
    for num_feat in num_feat_list:
        gc.collect()
        start_time = time.time()
        print("Selecting {} features.".format(num_feat))

        hsic_lasso.input(X,Y,featname=featname)
        hsic_lasso.classification(num_feat=num_feat, M=1, discrete_x=discrete_x)

        #Save parameters
        saveFeatFP=os.path.join(resultDir, str(num_feat)+"feats.csv")
        selected_featnames = hsic_lasso.get_features()
        Z = X[:, hsic_lasso.get_index()]
        saveDF(Z, Y, selected_featnames, saveFeatFP)
        featImptFP=os.path.join(resultDir, str(num_feat)+"feats_importance.csv")
        saveFeatures(hsic_lasso, featImptFP)
        print("Selecting top {} features costs {} seconds.".format(
            num_feat, time.time()-start_time))



