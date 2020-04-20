# coding: utf-8
# Usage:
# python <script> <original_data_filepath> <label_name>
#   label_name = "Diagnosis", "Class"

import os
import sys
import numpy as np

from fsUtil import saveDF, saveFeatures, getNumClass, dataProcess, extractFeatures

num_feat_list = [5, 10, 30, 50, 70, 100, 150, 200, 400, 600, 800, 1000]

def main(argv):
    original_data_filepath = argv[0]
    labelName = argv[1] #"Class" # "Diagnosis" # Class
    print("[Label is : {}]".format(labelName))


    df = dataProcess(original_data_filepath, labelName)
    if labelName == "Diagnosis":
        featname = list(df.columns)
        featname.remove("class")
        featname.remove("age group")

        print("[Process all age groups]")
        X = df.loc[:, (df.columns != "class") & (df.columns != "age group")].to_numpy(dtype=np.float32)
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
                    df["age group"] == ageGroup,
                    (df.columns != "class") & (df.columns != "age group")
                    ].to_numpy(dtype=np.float32)
            Y = df.loc[
                    df["age group"] == ageGroup,
                    df.columns == "class"
                    ].to_numpy(dtype=np.float32).flatten() # before flattern, its shape is (nSamples, 1)
            print("[{}] X:{}, Y:{}, feat:{}".format(ageGroup, np.shape(X), np.shape(Y), len(featname)))
            resultDir = "extFeat_"+labelName+"_"+ageGroup
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
