import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df=pd.read_csv("train.csv")
df1=pd.DataFrame(df[:])
df1.Sex=df1.Sex.replace('male',1).replace('female',0)

import math
mean_age=math.floor(df1.Age.median())
mean_fare=math.floor(df1.Fare.median())
df1.Age=df1.Age.fillna(mean_age)
df1.Fare=df1.Fare.fillna(mean_fare)

reg=linear_model.LogisticRegression()
reg.fit(df1[['Pclass','Sex','Age','Fare']],df1.Survived)

df2=pd.read_csv("test_X.csv")
df3=pd.DataFrame(df2[:])

df3.Sex=df3['Sex'].replace('male',1).replace('female',0)

mean=math.floor(df3.Age.median())
mean_f=math.floor(df3.Fare.median())
df3.Age=df3.Age.fillna(mean)
df3.Fare=df3.Fare.fillna(mean_f)

a=np.array(reg.predict(df3[['Pclass','Sex','Age','Fare']]))
a1=np.round(a)
a1=np.absolute(a1)
df4=pd.DataFrame(df3['PassengerId'])

df4['Survived']=a1

df4.to_csv('dead_alive.csv')

df5=pd.read_csv("test_Y.csv")
print("Train data score")
print(reg.score(df1[['Pclass','Sex','Age','Fare']],df1.Survived))
print("Test data score")
print(reg.score(df3[['Pclass','Sex','Age','Fare']],df5.Survived))
