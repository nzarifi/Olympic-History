# -*- coding: utf-8 -*-
"""
Created on Thu May 16 21:09:46 2019

@author: Niloofar-Z
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\Niloofar-Z\Desktop\MetroCo\datamining\120-years-of-olympic')


df = pd.read_csv("cleaned-olympic.csv")        
df.columns.values


####initial model
#
X=df[['Age', 'Weight','Year']].values
y=df[['Height']].values
type(y)
#X=df[['Age', 'Weight','Year']].values
X.dtype #.values convert pandas series to array format
from sklearn.preprocessing import StandardScaler

"""
The following format is problematic later for inverse_transform
and cause 
###NotFittedError: This StandardScaler instance is 
not fitted yet. Call 'fit' with appropriate arguments before 
using this method.###
X_scaled = StandardScaler().fit_transform(X)
y_scaled = StandardScaler().fit_transform(y.reshape(len(y),1)).reshape(len(y))
"""
sc_X = StandardScaler()
sc_y = StandardScaler()
X_scaled = sc_X.fit_transform(X)
y_scaled = sc_y.fit_transform(y.reshape(len(y),1)).reshape(len(y))



from sklearn.model_selection import train_test_split

X_train, X_test,y_train,y_test=train_test_split(X_scaled,y_scaled,test_size=0.3,random_state=0)

from sklearn.linear_model  import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
#r-squared
model.score(X_test,y_test)


y_pred=model.predict(X_test)
y_pred.shape
##sc_y = StandardScaler() This line here is the source of problem
##not allowed to define or even use StandardScaler() right before inverse

y_pred = sc_y.inverse_transform(y_pred.reshape(len(y_pred),1)).reshape(len(y_pred))
y_test = sc_y.inverse_transform(y_test.reshape(len(y_test),1)).reshape(len(y_test))

##r2_score and model.score are the same
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)  


from sklearn.metrics import mean_squared_error
meansquare=mean_squared_error(y_test,y_pred)
RMSE = np.sqrt(meansquare)
print("Root mean squared error: {}".format(RMSE))


import statsmodels.api as sm

X_modified=sm.add_constant(X_train)
lin_reg=sm.OLS(y_train,X_modified)
result=lin_reg.fit()
print(result.summary())

#height=-0.0009-0.0194Age+0.7909weight+0.0169year

X_modified = np.delete(X_modified,3,axis=1)
lin_reg = sm.OLS(y_train,X_modified)
result = lin_reg.fit()
print(result.summary())
print('Parameters: ', result.params)
print('R2: ', result.rsquared)


#let have height and weight 
X=df[['Weight']]
y=df[['Height']]
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
model = LinearRegression()
model.fit(X_train,y_train)
#r-squared
model.score(X_test,y_test)

b0=model.intercept_
model.coef_
b1=model.coef_[0]
x=np.array([X.Weight.min(),X.Weight.max()])
y_pred=b0+b1*x
#height=0.58weight+134

plt.plot(x,y_pred)
plt.scatter(X,y,c='red')
plt.xlabel('weight')
plt.ylabel('hight')
plt.savefig('foo',dpi=600,bbox_inches='tight')







