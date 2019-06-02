# -*- coding: utf-8 -*-
"""
Created on Sun May 19 16:21:39 2019

@author: Niloofar-Z
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

os.chdir(r'C:\Users\Niloofar-Z\Desktop\MetroCo\datamining\120-years-of-olympic')


df = pd.read_csv("athlete_events.csv")        
df.columns.values
#let's drop ID first
df.drop(columns='ID',axis=1, inplace=True)

#quick visualization to get idea
#a quick look
df.hist()
plt.savefig('foo',dpi=600,bbox_inches='tight')

#plot is from pandas
df.plot(kind='density', subplots=True, layout=(2,2), grid=True, sharex=False)
#[ax.legend(loc=1) for x in plt.gcf().axes]
plt.legend(bbox_to_anchor=(0.95, 0.95), loc=1, borderaxespad=0.)
plt.savefig('foo',dpi=600,bbox_inches='tight')
plt.show()




df.ax.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.savefig('foo',dpi=600,bbox_inches='tight')
df['Age'].describe()
df['Weight'].describe()
df['Height'].describe()

""" using matshow
correlations = df.corr()
# plot correlation matrix
names=['Age', 'Height', 'Weight','Year']
#names=list(df.columns.values)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1.0)
fig.colorbar(cax)
ticks = np.arange(0,4,1) 
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.savefig('foo',dpi=600,bbox_inches='tight')
plt.show()
"""

from pandas.plotting import scatter_matrix
scatter_matrix(df)
plt.savefig('foo',dpi=600,bbox_inches='tight')
plt.show()
#look into individual vars 
#plt.scatter(df.Age,df.Year)
#plt.scatter(df.Height,df.Weight)

import seaborn as sns

fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.savefig('foo',dpi=600,bbox_inches='tight')
plt.show()


#############cleaning data################
df.isnull().sum()
df.size #columnxrow
df.shape
"""
check the ratio of missing

Age         9474  3.4%
Height     60171  22.2%
Weight     62875  23.2%
Medal     231333  85%
(271116, 14)
"""
#replace the ave value for Age, Height and Weight, forget Medal

df['Age'].fillna(25,inplace=True)
df.Height.mean()
df['Height'].fillna(175,inplace=True)
df.Weight.mean()
df['Weight'].fillna(70,inplace=True)

######dummies########


df=pd.get_dummies(df,columns=['Season'],drop_first=True)#in dummies drop the first col

df.to_csv("cleaned-olympic.csv",index=False)
#








