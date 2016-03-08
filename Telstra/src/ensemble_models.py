import numpy as np
import pandas as pd
import scipy as sp
import xgboost as xgb
import math
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

whole = pd.read_csv("../input/dataset.csv")

#----split the dataset into train and test
my_train = whole[whole['fault_severity']>-1]
my_test = whole[whole['fault_severity'] == -1]
labels = list(my_train['fault_severity'])
ids = list(my_test['id'])
del my_train['fault_severity']
del my_test['fault_severity']
del my_test['id']
del my_train['id']

#define evalution metics
def metrics(y_hat, y):
    logloss = 0.0
    for i in range(0, len(y)):
        for j in range(3):
            if y[i] == j:
                epsilon = 1e-15
                prob = sp.minimum(1-epsilon, y_hat[i][j])
                prob = sp.maximum(epsilon, prob)
                logloss += math.log(prob)
    logloss = -logloss/len(y)
    return logloss


#-----build the xgboost model
param = {}
param['silent'] = 1
param['objective'] = 'multi:softprob'
param['eta'] = 0.05
param['gamma'] = 0.5
param['max_depth'] = 8
param['subsample'] = 0.8
param['min_child_weight'] = 0.8
param['num_class'] = 3
param['eval_metric'] = 'mlogloss'
num_rounds = 800

pred_xgb = np.zeros((my_test.shape[0],3))
xgb_test = xgb.DMatrix(my_test)
#-------cross validation
skf = StratifiedKFold(labels, 5, shuffle=True)
best_score = []
for ind1, ind2 in skf:
    X_train, X_val, y_train, y_val = my_train.values[ind1], my_train.values[ind2], np.array(labels)[ind1], np.array(labels)[ind2]
    
    xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_val = xgb.DMatrix(X_val, y_val)
    
    evallist = [(xgb_train, 'train'), (xgb_val, 'eval')]
    bst = xgb.train(param, xgb_train, num_rounds, evals=evallist, early_stopping_rounds=100, verbose_eval=50)

    pred_xgb += (1-bst.best_score)*(bst.predict(xgb_test).reshape(my_test.shape[0], 3))
    best_score += [bst.best_score]

print "xgboost cv score:", sum(best_score)/len(best_score)
pred_xgb = pred_xgb/(1*len(best_score)-sum(best_score))

#-----build the randomforest model
pred_rf = np.zeros((my_test.shape[0],3))
skf = StratifiedKFold(labels, 5, shuffle=True)
best_score = []
for ind1, ind2 in skf:
    X_train, X_val, y_train, y_val = my_train.values[ind1], my_train.values[ind2], np.array(labels)[ind1], np.array(labels)[ind2]
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=15,criterion="entropy",max_features=0.6,min_samples_leaf=5)
    rf = rf.fit(X_train, y_train)

    y_val_prob = rf.predict_proba(X_val)
    score = metrics(y_val_prob, y_val)
    pred_rf += (1-score)*(rf.predict_proba(my_test))
    best_score += [score]
 
print "randomforest cv score:", sum(best_score)/5
pred_rf = pred_rf/(1*len(best_score)-sum(best_score))

#------make the submission
prob1 = (pred_rf[:,0]+pred_xgb[:,0]*4)/5
prob2 = (pred_rf[:,1]+pred_xgb[:,1]*4)/5
prob3 = (pred_rf[:,2]+pred_xgb[:,2]*4)/5

result = {'id': ids, 'predict_0':prob1 , 'predict_1':prob2, 'predict_2':prob3}
pd.DataFrame(result)[['id', 'predict_0','predict_1', 'predict_2']].to_csv("../output/prediction.csv", index = False)