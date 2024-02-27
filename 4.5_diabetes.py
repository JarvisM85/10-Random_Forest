# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:54:09 2024

@author: sahil
"""

import pandas as pd
import numpy as np
data = pd.read_csv("C:/DS2/4_Random_Forest/diabetes.csv")

df = pd.DataFrame(data)
df.head()

X = df.drop("Outcome",axis='columns')
y = df["Outcome"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

model.fit(X_train,y_train)

model.score(X_test,y_test)
y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm


#Matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')



























