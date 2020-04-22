import os
import time
import json

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedShuffleSplit

import GPyOpt
import pickle
from functools import partial


def mapConfig(params, domain, clf_name="RF"):
    dict_params = dict(zip([el['name'] for el in domain], params))
    # map categorical vairables
    if clf_name == 'RF':
        if 'bootstrap' in dict_params:
            dict_params['bootstrap'] = True if dict_params['bootstrap'] == 1 else False
        if 'max_features' in dict_params:
            if dict_params['max_features'] == 0:
                dict_params['max_features'] = None
            elif dict_params['max_features'] == 1:
                dict_params['max_features'] = 'sqrt'
            else: # has a value of 2
                dict_params['max_features'] = 'log2'
        dict_params['n_estimators'] = int(dict_params['n_estimators'])
        dict_params['max_depth'] = int(dict_params['max_depth'])
        dict_params['min_samples_leaf'] = int(dict_params['min_samples_leaf'])
        dict_params['min_samples_split'] = int(dict_params['min_samples_split'])
    elif clf_name == "SVM":
        if 'kernel' in dict_params:
            dict_params["kernel"] = "rbf" if dict_params["kernel"] == 0 else "poly"
        else:
            dict_params["kernel"] = "rbf"
        if dict_params["kernel"] == "poly":
            if 'degree' in dict_params: # for 'poly' kernel
                dict_params['degree'] = int(dict_params['degree'])
            else: # use 5 as the degree for 'poly' kernel
                dict_params['degree'] = 5

    #print("[Mapped Params]:\n{}".format(dict_params))
    return dict_params

def createModel(params, clf_name, class_weight=None):
    if clf_name == "RF":
        model = RandomForestClassifier(class_weight=class_weight, **params)
    else: # "SVM"
        model = svm.SVC(class_weight=class_weight, **params)
    return model

def fitCV(configs, cv=None, X_train=None, y_train=None, domain=None, clf_name="RF", class_weight=None):
    fs = np.zeros((configs.shape[0], 1))
    for i, params in enumerate(configs):
        dict_params = mapConfig(params, domain, clf_name=clf_name)
        #print(dict_params)
        mdl = createModel(dict_params, clf_name, class_weight)
        # For minimization: negative validation accuracy averaged all folds
        fs[i] = -np.mean(cross_val_score(mdl, X_train, y_train, cv=cv))
    return fs


def plotAcc(figID, numFeats, accs, clf_name, test_ratio, resultFDName, itersBO):

    #[accs]: (numRepresentations, 2). First 2: train vs test. second 2: mean vs std

    plt.figure(figID)
    trainAcc = accs[:, 0]
    testAcc = accs[:, 1]

    plt.plot(numFeats, trainAcc, 'b--', label="Training Accuracy")
    plt.plot(numFeats, testAcc, 'r-o', label="Test Accuracy")

    plt.legend(loc="best")
    plt.title("{}: {}% Test Ratio".format(clf_name, int(100*test_ratio)))
    plt.xlabel("Number of Selected Top Features")
    plt.ylabel("Test Accuracy")
    figFP = os.path.join(
            resultFDName,
            "{}_{}FeatureGroups_{}TestRatio_{}BOIters_Accs.png".format(
                clf_name, len(numFeats), int(100*test_ratio), itersBO)
            )
    plt.savefig(figFP, dpi=600)

def NP2CSV(arr, npColumnNames, npRowName, npRowValues, fp=None):
    df = pd.DataFrame(data=arr, columns=npColumnNames)
    df.insert(loc=0, column=npRowName, value=npRowValues)
    if fp != None:
        df.to_csv(fp, index=False)
    return df

# Train and testing on one type of data.
# Potentionally testing on different types of data.
def evaluation(
        dataDirPrefix, sourceDataTag, labelName,
        numFGs, numFeats, test_ratio, cv,
        domain, clf_name, itersBO,
        saveFileNamePrefix, figID, testDataTags=None, oriDF=None, class_weight=None):

    numFGs = len(numFeats)

    if testDataTags != None:
        #if oriDF == None:
        #    raise "oriDF should not be None because testDataTags is not None"
        transferAccsAllFGs = np.zeros((numFGs, len(testDataTags)))

    if class_weight == None:
        weight_tag = "WeightEqual"
    elif class_weight == "balanced":
        weight_tag = "WeightBalanced"
    else:
        weight_tag = "Weight{}".format(class_weight[1])
    resultFDName = "result_"+labelName+"_"+sourceDataTag+"_"+weight_tag
    if not os.path.exists(resultFDName):
            os.makedirs(resultFDName)

    accs = np.zeros((numFGs, 2))
    optConfigs = {}
    for nfIdx in range(numFGs):
        start_time = time.time()
        numFeat = numFeats[nfIdx]
        print("\n[Using {} features]".format(numFeat))

        if testDataTags != None: # Cancer - Young and Old
            # get the feature indices in the original dataset
            feats_impt_fp = os.path.join(
                    dataDirPrefix+"_"+labelName+"_"+sourceDataTag,
                    str(numFeat)+"feats_importance.csv")
            feats_impt_df = pd.read_csv(feats_impt_fp)
            feats_names = feats_impt_df["FeatureName"].tolist()

            testDataBank = {}
            for dataTag in testDataTags:
                testDF = None
                if dataTag == "All":
                    #testDF = oriDF[oriDF.columns.isin(feats_names)]
                    testDF = oriDF[feats_names]
                    testDF.insert(loc=0, column='class', value=oriDF["class"])
                else: # Young or Old
                    testDF = oriDF.loc[oriDF["age group"]==dataTag, feats_names]
                    testDF.insert(loc=0, column='class', value=oriDF["class"])
                #dataDir=dataDirPrefix+"_"+labelName+"_"+dataTag
                #dataFP = os.path.join(dataDir, str(numFeat)+"feats.csv")
                #testDataBank[dataTag] = pd.read_csv(dataFP)
                testDataBank[dataTag] = testDF        

            sourceData = testDataBank[sourceDataTag]
        else:
            sourceDataDir=dataDirPrefix+"_"+labelName+"_"+sourceDataTag
            sourceDataFP = os.path.join(sourceDataDir, str(numFeat)+"feats.csv")
            sourceData = pd.read_csv(sourceDataFP)

        Y = sourceData['class'].to_numpy()
        X = sourceData.loc[:, sourceData.columns != "class"].to_numpy()

        train_index, test_index = next(StratifiedShuffleSplit(test_size=test_ratio, random_state=123).split(X,Y))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        opt = GPyOpt.methods.BayesianOptimization(
                f = partial(fitCV, X_train=X_train, y_train=y_train, cv=cv, domain=domain, clf_name=clf_name, class_weight=class_weight),  # function to optimize
                domain = domain,         # box-constrains of the problem
                acquisition_type ='EI')       # EI, MPI, LCB acquisition
        opt.run_optimization(max_iter=itersBO)
        #opt.plot_convergence()

        x_best = opt.x_opt
        best_params = mapConfig(x_best, domain, clf_name=clf_name)
        print("[Best Config]: {}".format(best_params))
        optModel = createModel(best_params, clf_name)
        optModel.fit(X_train, y_train)

        trainAcc = round(optModel.score(X_train, y_train), 4)
        testAcc = round(optModel.score(X_test, y_test), 4)
        print("[{}] Train score: {}, Test score: {}. ({} seconds)".format(
            numFeat, trainAcc, testAcc, time.time()-start_time) )
        accs[nfIdx, 0] = trainAcc
        accs[nfIdx, 1] = testAcc

        optConfigs["numFeat"+str(numFeat)] = {"modelParams": best_params, "testAcc": testAcc}
        bestModelFP = os.path.join(
                resultFDName,
                saveFileNamePrefix+"numFeat"+str(numFeat)+"_Model.sav")
        pickle.dump(optModel, open(bestModelFP, "wb"))

        # save test acc cross datasets
        if testDataTags:
            transferAccsPerFeatGroup = []
            for testDataTag in testDataTags:
                testData = testDataBank[testDataTag]
                teY = testData['class'].to_numpy()
                teX = testData.loc[:, sourceData.columns != "class"].to_numpy()
                transferAccsPerFeatGroup.append(round(optModel.score(teX, teY), 4))

            transferAccsAllFGs[nfIdx] = np.array(transferAccsPerFeatGroup)            

    # Save optimal configs
    json_str = json.dumps(optConfigs, indent=4)
    optConfigsFP = os.path.join(
            resultFDName,
            saveFileNamePrefix+"_OptConfigs.json")
    with open(optConfigsFP, "w") as fp:
        print(json_str, file=fp)

    # Save Training and Testing Accuracies
    accsFP = os.path.join(
            resultFDName,
            saveFileNamePrefix+"_Accs.csv")
    npColumnNames = ["TrainAcc", "TestAcc"]
    npRowName = "NumFeatures"
    npRowValues = numFeats 
    NP2CSV(accs, npColumnNames, npRowName, npRowValues, fp=accsFP)
    plotAcc(figID, numFeats, accs, clf_name, test_ratio, resultFDName, itersBO)

    return transferAccsAllFGs


