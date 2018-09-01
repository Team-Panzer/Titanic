# Titanic
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

df=pd.read_csv('train.csv')
df1=pd.read_csv('test_x.csv')
df2=pd.read_csv('test_y.csv')
df.Sex=df.Sex.replace('male',1).replace('female',0)


import math
mean_age=math.floor(df.Age.median())
mean_cost=math.floor(df.Cost.median())
df.Age=df1.Age.fillna(mean_age)
df.cost=df1.Fare.fillna(mean_cost)

reg=linear_model.LogisticRegression()
reg.fit(df1[['Pclass','Sex','Age','Cost']],df1.Survived)

mean=math.floor(df1.Age.median())
mean_f=math.floor(df1.Cost.median())
df1.Age=df1.Age.fillna(mean)
df1.Cost=df1.Cost.fillna(mean_f)


a=np.array(reg.predict(df1[['Pclass','Sex','Age','Cost']]))
df3=pd.DataFrame(df1['PassengerId'])

df3['Survived']=a

df3.to_csv('dead_alive.csv')

y=pd.read_csv("test_Y.csv")
print("The test data score is")
print(reg.score(x_test[['Pclass','Sex','Age','Cost']],y[['Survived']]))
print("The train data score is")
print(reg.score(df[['Pclass','Sex','Age','Cost']],df[['Survived']]))
