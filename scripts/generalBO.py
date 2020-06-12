import GPy
import GPyOpt
import numpy as np
import pandas as pd

from numpy.random import seed

from sklearn.model_selection import cross_val_score

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
        if dict_params["kernel"] == 0:
            dict_params["kernel"] = "rbf"
        else:
            dict_params["kernel"] = "poly"

    #print("[Mapped Params]:\n{}".format(dict_params))
    return dict_params

def createModel(params, clf_name):
    if clf_name == "RF":
        model = RandomForestClassifier(**params)
    else: # "SVM"
        model = svm.SVC(**params)
    return model

def fit_svc_val(configs, cv=None, X_train=None, y_train=None, domain=None, clf_name="RF"):
    fs = np.zeros((configs.shape[0], 1))
    for i, params in enumerate(configs):
        dict_params = mapConfig(params, domain, clf_name=clf_name)
        mdl = createModel(dict_params, clf_name)
        #mdl.set_params(**dict_params)
        # For minimization: negative validation accuracy averaged all folds
        fs[i] = -np.mean(cross_val_score(mdl, X_train, y_train, cv=cv))
    return fs

from sklearn.datasets import load_iris, make_blobs
X, y = make_blobs(centers=3, cluster_std=4, random_state=1234)

from sklearn.model_selection import StratifiedShuffleSplit
train_index, test_index = next(StratifiedShuffleSplit(test_size=.4, random_state=123).split(X,y))

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]


SVM_Domain = [
        {'name': 'C',      'type': 'continuous', 'domain': (0, 1000)},
        {'name': 'kernel', 'type': 'categorical', 'domain': (0, 1)},
        {'name': 'gamma',  'type': 'continuous', 'domain': (1e-5, 0.1)}
        ]

RF_Domain = [
        {'name': 'bootstrap', 'type': 'categorical', 'domain': (0, 1)}, # 0 - False, 1 - True
        {'name': 'max_depth', 'type': 'discrete', 'domain': (5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)},
        {'name': 'max_features', 'type': 'categorical', 'domain': (0, 1, 2)}, # 0 - None (all features), 1 - sqrt, 2 - log2
        {'name': 'min_samples_leaf', 'type': 'discrete', 'domain': (1 ,2 ,4)},
        {'name': 'min_samples_split', 'type': 'discrete', 'domain': (2 ,5 ,10)},
        {'name': 'n_estimators', 'type': 'discrete', 'domain': (100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000)}
        ]

cv = 5 
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from functools import partial
#mdlSVM = svm.SVC(kernel='rbf')
#mdlRF = RandomForestClassifier(random_state=127)
#mdl = mdlRF #mdlSVM #mdlRF
domain = RF_Domain #SVM_Domain #RF_Domain
clf_name = "RF" #"SVM" #"RF"

print("Starting tuning {} using BO".format(clf_name))
opt = GPyOpt.methods.BayesianOptimization(
        f = partial(fit_svc_val, X_train=X_train, y_train=y_train, cv=cv, domain=domain, clf_name=clf_name),  # function to optimize
        domain = domain,         # box-constrains of the problem
        acquisition_type ='LCB')       # EI, MPI, LCB acquisition
opt.run_optimization(max_iter=50)
#opt.plot_convergence()

#x_best = np.exp(opt.x_opt)
x_best = opt.x_opt
best_params = mapConfig(x_best, domain, clf_name=clf_name)
print("Best parameters extracted: %s" % best_params)

optModel = createModel(best_params, clf_name)

optModel.fit(X_train, y_train)
print("GPyOpt\nTrain score: %.3f" % optModel.score(X_train, y_train))
print("GPyOpt\nTest score: %.3f" % optModel.score(X_test, y_test))


