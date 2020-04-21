# -*- coding: utf-8 -*-
"""
Created on Mon May 20 13:23:54 2019
@author: Niloofar-Z
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\Niloofar-Z\Desktop\MetroCo\datamining\120-years-of-olympic')

df = pd.read_csv("cleaned-olympic.csv")
df.columns.values
df.info()  # good one

####initial model
#
X = df[['Age', 'Weight', 'Year']].values
y = df[['Height']].values

# scale the values
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(len(y), 1)).reshape(len(y))

# note that we did not split X and y here

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

Listfeatures = ['Age', 'Weight', 'Year']

adj_R2 = []
feature_set = []
max_adj_R2_so_far = 0
n = len(X)
k = len(X[0])
for i in range(1, k + 1):
    selector = RFE(LinearRegression(), i, verbose=1)
    selector = selector.fit(X, y)
    current_R2 = selector.score(X, y)
    current_adj_R2 = 1 - (n - 1) * (1 - current_R2) / (n - i - 1)
    adj_R2.append(current_adj_R2)
    feature_set.append(selector.support_)

    # loop starts with 1 feature and I put the threshold of 0.005
    # to ignore not very important features
    if current_adj_R2 - max_adj_R2_so_far > 0.005:
        max_adj_R2_so_far = current_adj_R2
        selected_features = selector.support_
        final_ranking = list(selector.ranking_)
    print('End of iteration no. {}'.format(i))
# the result shows that age and year are not that important
# details of ranking
DicOfRank = {}
for i in range(0, len(Listfeatures)):
    case = {final_ranking[i]: Listfeatures[i]}
    DicOfRank.update(case)
print('ranking info: ', DicOfRank)

# for this case since RFE chose all three, no need to use X_sub
X_sub = X[:, selected_features]

##############10fold R2 value#####################

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
scores = []
max_score = 0
from sklearn.model_selection import KFold

kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    current_model = LinearRegression()
    # train the model
    current_model.fit(X_train, y_train)
    # see performance score
    current_score = current_model.score(X_test, y_test)
    scores.append(current_score)
    if max_score < current_score:
        max_score = current_score
        best_model = current_model
import statistics

print('all scores: {}'.format(scores))
print('mean score: ', statistics.mean(scores))

y_train.shape
X_train.shape
best_model.intercept_
best_model.coef_
# height=0.0009-0.0195Age+0.7888weight+0.0163year

X_test.shape
y_test.shape
type(X)
from sklearn.model_selection import cross_val_score, cross_val_predict

list_R2 = cross_val_score(LinearRegression(), X, y, cv=5, scoring='neg_mean_squared_error')
list_R2.mean()

##########################Extra###############
####correlation of number of Athletes vs year
df[(df.Medal == 'Gold')]
df.groupby(['NOC', 'Medal']).count()
df.head()

MenOverTime = df[(df.Sex == 'M') & (df.Season_Winter == 0)]
WomenOverTime = df[(df.Sex == 'F') & (df.Season_Winter == 0)]

"""
part = MenOverTime.groupby('Year')['Sex'].value_counts()
plt.figure(figsize=(20, 10))
part.loc[:,'M'].plot()
plt.title('Variation of Male Athletes over time')
"""

part1 = MenOverTime.groupby('Year')['Sex'].value_counts()
part2 = WomenOverTime.groupby('Year')['Sex'].value_counts()
m = part1.loc[:, 'M']
f = part2.loc[:, 'F']
female = f.reset_index(level=0)
male = m.reset_index(level=0)
female = female.rename(columns={"Sex": "F"})
male = male.rename(columns={"Sex": "M"})
merged = pd.merge(female, male, how='outer')
merged['F'].fillna(0, inplace=True)

# f.reset_index(drop=True, inplace=True)#it drops index which is year


plt.scatter(merged.Year, merged.F)
plt.xlabel('year')
plt.ylabel('women_count')
plt.savefig('foo', dpi=600, bbox_inches='tight')
plt.scatter(merged.Year, merged.M)
plt.xlabel('year')
plt.ylabel('men_count')
plt.savefig('foo', dpi=600, bbox_inches='tight')

y = merged['Year']
X = merged.drop(columns=['Year'])
merged.corr()  # highly correlated. you can run it as a test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

# R2 score
model.score(X_test, y_test)
y_pred = model.predict(X)
b0 = model.intercept_
model.coef_
b1 = model.coef_[0]
b2 = model.coef_[1]
# it does not help for 3D model
# x1=np.array([X.F.min(),X.F.max()])
# x2=np.array([X.M.min(),X.M.max()])
# y_pred=b0+b1*x1+b2*x2

##predicted_year=1897+0.01women+0.006men


from mpl_toolkits import mplot3d

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(X['F'].values, X['M'].values, y_pred, 'gray')
# ax.plot3D(X['F'].values, X['M'].values, y,'red')
ax.scatter3D(X['F'].values, X['M'].values, y, c=y, cmap='seismic_r')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.savefig('foo.pdf', bbox_inches='tight')

from sklearn.datasets import make_friedman1
from sklearn.feature_selection import RFE
from sklearn.svm import SVR

X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
estimator = SVR(kernel="linear")
selector = RFE(estimator, 5, step=1)
selector = selector.fit(X, y)
selector.support_
l = list(selector.ranking_)
type(list(l))
