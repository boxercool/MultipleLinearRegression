#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 23:03:36 2019

@author: reenajoshan
"""

import pandas as pd
import numpy as np

# np.zeros(3)
pf = pd.read_csv('hiring.csv')
print(pf)
# experience
# c
# print(pf['experience'])

pf['test_score'].fillna(int(pf['test_score'].mean()), inplace=True)

X = pf.iloc[:, :3]


# print(pf['test_score'])
def convert_int(word):
    dic_word = {'one': 1,
                'two': 2,
                'three': 3,
                'four': 4,
                'five': 5,
                'six': 6,
                'seven': 7,
                'eight': 8,
                'nine': 9,
                'ten': 10,
                'zero': 0,
                'eleven': 11,
                0: 0}
    #  print("welxome",dic_word)
    return dic_word[word]


X = X.fillna(0)
X['experience'] = X['experience'].apply(lambda x: convert_int(x))

y = pf.iloc[:, -1]

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X, y)
regressor.coef_
regressor.intercept_
"""print(regressor.intercept_)
import pickle
with open('linearmodel','wb') as f:
   pickle.dump(regressor,f)
print(regressor.predict([[0,8,9]]))
with open('linearmodel','rb')as fb:
    model=pickle.load(fb)

    print(int(model.predict([[0,8,9]])))
"""

# print(pf)
from sklearn.externals import joblib

joblib.dump(regressor, 'model_joblib1')
mjoblib = joblib.load('model_joblib1')
print(int(mjoblib.predict([[0, 8, 9]])))
