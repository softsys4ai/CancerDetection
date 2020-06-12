# coding: utf-8
# Usage:
# python <script> <original_data_filepath> <label_name>
#   label_name = "Diagnosis", "Class"

import os
import sys
import numpy as np

from fsUtil import saveDF, saveFeatures, getNumClass, dataProcess, extractFeatures

#num_feat_list = [5, 10, 30, 50, 70, 100, 150, 200, 400, 600, 800, 1000]

# Detect AgeGroup: maxmimum number of useful features is 95
num_feat_list = [15, 20, 25] #[5, 10, 30, 50, 70, 100]


def main(argv):
    original_data_filepath = argv[0]
    labelName = argv[1] #"Class" # "Diagnosis" # Class
    print("[Label is : {}]".format(labelName))


    df = dataProcess(original_data_filepath, labelName)
    if labelName == "Diagnosis":
        featname = list(df.columns)
        featname.remove("class")
        featname.remove("AgeGroup")

        print("[Process all age groups]")
        X = df.loc[:, (df.columns != "class") & (df.columns != "AgeGroup")].to_numpy(dtype=np.float32)
        Y = df['class'].to_numpy(dtype=np.float32)
        resultDir = "extFeat_"+labelName+"_All"
        if not os.path.exists(resultDir):
                os.makedirs(resultDir)
        extractFeatures(X, Y, num_feat_list, featname, resultDir)

        # process Young and Old groups exclusively
        ageGroups = ["Young", "Old"]
        for ageGroup in ageGroups:
            print("[Process {} Group]".format(ageGroup))
            X = df.loc[
                    df["AgeGroup"] == ageGroup,
                    (df.columns != "class") & (df.columns != "AgeGroup")
                    ].to_numpy(dtype=np.float32)
            Y = df.loc[
                    df["AgeGroup"] == ageGroup,
                    df.columns == "class"
                    ].to_numpy(dtype=np.float32).flatten() # before flattern, its shape is (nSamples, 1)
            print("[{}] X:{}, Y:{}, feat:{}".format(ageGroup, np.shape(X), np.shape(Y), len(featname)))
            resultDir = "extFeat_"+labelName+"_"+ageGroup
            if not os.path.exists(resultDir):
                    os.makedirs(resultDir)
            extractFeatures(X, Y, num_feat_list, featname, resultDir)
    elif labelName == "AgeGroup":
        featname = list(df.columns)
        featname.remove("class")
        featname.remove("AgeGroup")

        print("[Process all diagnosis groups]")
        X = df.loc[:, (df.columns != "class") & (df.columns != "AgeGroup")].to_numpy(dtype=np.float32)
        Y = df['AgeGroup'].to_numpy(dtype=np.float32)
        resultDir = "extFeat_"+labelName+"_All"
        if not os.path.exists(resultDir):
                os.makedirs(resultDir)
        extractFeatures(X, Y, num_feat_list, featname, resultDir)

        # process Cancer and Clear groups exclusively
        diagnosisGroups = ["Cancer", "No-Cancer"]
        for diagnosisGroup in diagnosisGroups:
            print("[Process {} Group]".format(diagnosisGroup))
            X = df.loc[
                    df["class"] == diagnosisGroup,
                    (df.columns != "class") & (df.columns != "AgeGroup")
                    ].to_numpy(dtype=np.float32)
            Y = df.loc[
                    df["class"] == diagnosisGroup,
                    df.columns == "AgeGroup"
                    ].to_numpy(dtype=np.float32).flatten() # before flattern, its shape is (nSamples, 1)
            print("[{}] X:{}, Y:{}, feat:{}".format(diagnosisGroup, np.shape(X), np.shape(Y), len(featname)))
            resultDir = "extFeat_"+labelName+"_"+diagnosisGroup
            if not os.path.exists(resultDir):
                    os.makedirs(resultDir)
            extractFeatures(X, Y, num_feat_list, featname, resultDir)


    else:
        X = df.loc[:, df.columns != "class"].to_numpy(dtype=np.float32)
        Y = df['class'].to_numpy(dtype=np.float32)
        featname = list(df.columns)
        featname.remove("class")
        resultDir = "extFeat_"+labelName+"_All"
        if not os.path.exists(resultDir):
                os.makedirs(resultDir)
        extractFeatures(X, Y, num_feat_list, featname, resultDir)

if __name__ == "__main__":
    main(sys.argv[1:])
