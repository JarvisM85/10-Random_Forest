# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:14:13 2024

@author: sahil
"""

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import warnings
warnings.filterwarnings('ignore')
#read csv file
loan_data=pd.read_csv('income.csv')
loan_data.columns
loan_data.head()

#Let us split the data into ip and op
X=loan_data.iloc[:,0:6]
y=loan_data.iloc[:,6]

#Split the dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
#Create adaboost classifier
ada_model=AdaBoostClassifier(n_estimators=100,learning_rate=1)
#n_estimators= no. of weak learners
#Learnimg_rate it contributes weights of weak learners, bydefault 1
#Train the model

model=ada_model.fit(X_train,y_train)
#Predict the results

y_pred=model.predict(X_test)
print('accuracy',metrics.accuracy_score(y_test,y_pred))
#Let us try for another base model

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

#Here base model is changed
Ada_model=AdaBoostClassifier(n_estimators=50,base_estimator=lr,learning_rate=1)
model=ada_model.fit(X_train,y_train)

y_pred=model.predict(X_test)
print('accuracy',metrics.accuracy_score(y_test,y_pred))