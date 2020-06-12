# Usage:
# python <script_name> <classifier_name> <label_name> <class_weight> <original_data_filepath>
#    classifier_name  = "SVM", "RF"
#    label_name =  "Diagnosis", "Class"
#    class_weight = "equal", "balanced", float
#    original_data_filepath: Optional. It is only for "Diagnosis" Task
#       e.g., Sample_Master_Source_and_Platform_batch_removed_Extra_Sample_Info.exported_for_AI.txt
#
# The script should be in the same directory level as directories where the extracted features files locate

import os
import sys
import time

import numpy as np
import pandas as pd

from BoUtil import mapConfig, createModel, fitCV, plotAcc, NP2CSV, evaluation 
from fsUtil import dataProcess

SVM_Domain = [
        {'name': 'C',      'type': 'continuous', 'domain': (1, 1000)},
        #{'name': 'kernel', 'type': 'categorical', 'domain': (0, 1)},
        #{'name': 'degree', 'type': 'discrete', 'domain': (3, 4, 5, 6, 7, 8)},
        {'name': 'gamma',  'type': 'continuous', 'domain': (1e-5, 0.02)}
        ]

RF_Domain = [
        {'name': 'bootstrap', 'type': 'categorical', 'domain': (0, 1)}, # 0 - False, 1 - True
        {'name': 'max_depth', 'type': 'discrete', 'domain': (5, 10, 20, 30, 40, 50, 60)},
        {'name': 'max_features', 'type': 'categorical', 'domain': (0, 1, 2)}, # 0 - None (all features), 1 - sqrt, 2 - log2
        {'name': 'min_samples_leaf', 'type': 'discrete', 'domain': (1 ,2 ,4)},
        {'name': 'min_samples_split', 'type': 'discrete', 'domain': (2 ,5 ,10)},
        {'name': 'n_estimators', 'type': 'discrete', 'domain': (100, 150, 200, 250, 300, 350, 400, 450, 500)}
        ]
domains = {"SVM": SVM_Domain, "RF": RF_Domain}
itersBO = 50
cv = 5 # folds in cross validation
resultFDPrex="TestRatio"
test_ratio= 0.4
dataDirPrefix = "extFeat"

#numFeats = [5, 10, 30, 50, 70, 100, 150, 200, 400, 600, 800, 1000]
#numFGs = len(numFeats)

def main(argv):
    global numFeats
    clf_name = argv[0] # "SVM", "RF"
    labelName = argv[1] # "Diagnosis", "Class", "AgeGroup"
    class_weight_label = argv[2] # equal, balanced, float
    if labelName == "Diagnosis":
        numFeats = [5, 10, 30, 50, 70, 100, 150, 200, 400, 600, 800, 1000]
    else: # currently only valid for "AgeGroup"
        numFeats = [5, 10, 30, 50, 70, 100]
    numFGs = len(numFeats)

    if class_weight_label == "balanced":
        class_weight = "balanced"
    elif class_weight_label == "equal":
        class_weight = None
    else:
        class_weight = {1 : float(class_weight_label)}
    print("class_weight: {}".format(class_weight))

    if labelName == "Diagnosis" or labelName == "AgeGroup":
        original_data_filepath = argv[3]
        oriDF = dataProcess(original_data_filepath, labelName)
    else:
        oriDF = None

    domain = domains[clf_name]
    saveFileNamePrefix = "{}_{}_{}FeatureGroups_{}TestRatio_{}BOIters".format(
            labelName, clf_name, numFGs, int(100*test_ratio), itersBO)

    if labelName == "Diagnosis":
        dataTags = ["All", "Young", "Old"]
        testDataTags = dataTags
    elif labelName == "AgeGroup":
        dataTags = ["All", "Cancer", "No-Cancer"]
        testDataTags = dataTags
    else: # Class, AgeGroup
        dataTags = ["All"]
        testDataTags = None
    numDataTags = len(dataTags)
    #scenarios = {
    #        "ALL": ["All", "Young", "Old"],
    #        "Young": ["All", "Young", "Old"],
    #        "Old": ["All", "Young", "Old"],
    #        }
    if testDataTags != None:
        crossDataAccs = np.zeros((numDataTags, numFGs, numDataTags))

    figID = 1
    #testDataTags = dataTags
    for dtIdx in range(numDataTags):
        sourceDataTag = dataTags[dtIdx]
        print("\nTraining a classifier with the source data: {}".format(sourceDataTag))
        transferAccsAllFGs = evaluation(
                dataDirPrefix, sourceDataTag, labelName,
                numFGs, numFeats, test_ratio, cv,
                domain, clf_name, itersBO,
                saveFileNamePrefix, figID,
                testDataTags=testDataTags, oriDF=oriDF,
                class_weight=class_weight)
        if testDataTags != None:
            crossDataAccs[dtIdx] = transferAccsAllFGs

        figID += 1

    if testDataTags != None:
        crossDataAccDir = "result_transferAcc"
        if not os.path.exists(crossDataAccDir):
                os.makedirs(crossDataAccDir)

        for nfIdx in range(numFGs):
            numFeat = numFeats[nfIdx]
            crossDataAccsDF = pd.DataFrame(data=crossDataAccs[:, nfIdx, :], columns=dataTags)
            crossDataAccsFP = os.path.join(
                    crossDataAccDir,
                    "{}_{}_{}Feat_{}TestRatio_{}BOIters_{}Weight_TransferAccs.csv".format(
                        labelName, clf_name, numFeat, int(100*test_ratio), itersBO, class_weight_label)
                    )
            crossDataAccsDF.to_csv(crossDataAccsFP, index=False)

if __name__ == "__main__":
    main(sys.argv[1:])