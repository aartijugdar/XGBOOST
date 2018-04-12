# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 21:25:22 2018

@author: aarti jugdar
"""
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score


data=pd.read_csv("")
Count_Col=data.shape[1]
X = data.iloc[:,0:(Count_Col - 1)]
Y= data.iloc[:,(Count_Col - 1)]

#defining a function for xgboost
def xgmethod(X,Y):
  
    # split data into train and test sets
    seed = 7
    test_size = 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    scaler = preprocessing.StandardScaler().fit(X_train)
    scaler.transform(X_train) 
    # XGtrain matrix
    xgtrain = xgb.DMatrix(X_train, label=y_train)
    
   
    model = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=100)
    xgb_param = model.get_xgb_params()
    
    print ('Start cross validation')
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=500, nfold=10, metrics=['auc'],
     early_stopping_rounds=50, stratified=True, seed=1301)
    print('Best number of trees = {}'.format(cvresult.shape[0]))
    
    model.set_params(n_estimators=cvresult.shape[0])
    print('Fit on the trainingsdata')
    model.fit(X_train, y_train, eval_metric='auc')
   
    pred = model.predict(X_test, ntree_limit=cvresult.shape[0])
    
  
    # make predictions for test data
    predictions = [round(value) for value in pred]
   
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    return accuracy