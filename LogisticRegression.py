# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 19:52:29 2019

@author: sundar.p.jayaraman
"""

import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./titanic-dataset-full.csv")
print(df.columns)

X = pd.DataFrame()
X['sex'] = df['sex']
X['age'] = df['age']
X['pclass'] = df['pclass']
X['sibsp'] = df['sibsp']
X['parch'] = df['parch']
X['survived'] = df['survived']

X["age"].fillna(X.age.mean(), inplace=True)
X = X.dropna(axis=0)


#survived will be my dependent variable, y.   I'll assign it to y and remove it from X
y = X['survived']
X = X.drop(['survived'], axis=1)

X['sex'] = pd.get_dummies(X.sex)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model = LogisticRegression()
model.fit(X_train, y_train)

print ("Logistic accuracy is %2.2f" % accuracy_score(y_test,model.predict(X_test)))

print(model.coef_)


