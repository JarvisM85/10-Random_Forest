# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:19:30 2024

@author: sahil
"""

import pandas as pd
import numpy as np
data = pd.read_csv("C:/DS2/4_Random_Forest/Fraud_check.csv")
dir(data)

df = pd.DataFrame(data)
df.head()

df['target'] = data["Taxable.Income"]



from sklearn.preprocessing import LabelEncoder
ug = LabelEncoder()
ms = LabelEncoder()
urban = LabelEncoder()
tax_income = LabelEncoder()

data['ug_n'] = ug.fit_transform(data['Undergrad'])
data['ms_n'] = ms.fit_transform(data['Marital.Status'])
data['urban_n'] = urban.fit_transform(data['Urban'])
data_n = data.drop(['Undergrad','Marital.Status','Urban'],axis='columns')

data_n
df["tax_income_n"]= tax_income.fit_transform(data["Taxable.Income"]<=30000)
df= df.drop("target",axis='columns')

df

X = data_n.drop("Taxable.Income",axis='columns')
y = df["tax_income_n"]



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













