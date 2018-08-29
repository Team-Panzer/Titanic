import pandas as pd
import numpy as np
from sklearn import linear_model
df=pd.read_csv("train.csv")
df1=df['Sex'].replace(['male'], 1).replace(['female'], 0)
df.Sex=df1
df['Age']=df['Age'].replace(np.nan, 29.7)
reg=linear_model.LogisticRegression()
reg.fit(df[['Pclass','Sex','Age','Fare']],df.Survived)
x_test=pd.read_csv("test_X.csv")
x_test['Sex']=x_test['Sex'].replace(['male'],1).replace(['female'],0)
x_test['Age']=x_test['Age'].replace(np.nan, 27)
x_test['Fare']=x_test['Fare'].replace(np.nan, 14.1083)
a=reg.predict(x_test[['Pclass','Sex','Age','Fare']])
b=a.round()
a=np.absolute(b)
a
y=pd.read_csv("test_Y.csv")
print("The test data score is:")
print(reg.score(x_test[['Pclass','Sex','Age','Fare']],y[['Survived']]))
print("The train data score is:")
print(reg.score(df[['Pclass','Sex','Age','Fare']],df[['Survived']]))
