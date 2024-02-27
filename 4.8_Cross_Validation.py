# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:43:55 2024

@author: sahil
"""

import pandas as pd

data = pd.read_csv("C:/DS2/4_Random_Forest/puma_diabetes.csv")

df = pd.DataFrame(data)
df.head()
df.isnull().sum()
df.describe()
df.Outcome.value_counts()

X = df.drop("Outcome",axis='columns')
y = df["Outcome"]


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled[:3]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,stratify=y,random_state=10)

X_train.shape
X_test.shape
y_train.value_counts()

y_test.value_counts()

# Train using stand alone model
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

scores = cross_val_score(DecisionTreeClassifier(), X,y,cv=5)
scores
scores.mean()

#Train using Bagging
from sklearn.ensemble import BaggingClassifier

bag_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0)

bag_model.fit(X_train, y_train)
bag_model.oob_score

bag_model.score(X_test, y_test)

bag_model = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    max_samples=0.8,
    oob_score=True,
    random_state=0)

scores = cross_val_score(bag_model, X,y, cv=5)
scores
scores.mean()


#Train using Random Forest
from sklearn.ensemble import RandomForestClassifier

scores = cross_val_score(RandomForestClassifier(n_estimators=50),X,y,cv=5)
scores.mean()


