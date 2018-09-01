import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv('train.csv')
data1=pd.read_csv('test_X.csv')
data2=pd.read_csv('test_y.csv')
data.Sex=data.Sex.replace('male',1).replace('female',0)
data1.Sex=data1.Sex.replace('male',1).replace('female',0)
data.Embarked=data.Embarked.replace('S',1).replace('C',0).replace('Q',2)
data1.Embarked=data1.Embarked.replace('S',1).replace('C',0).replace('Q',2)



data['Cabin']=data['Cabin'].fillna(0)
count=data['Cabin'].value_counts().to_dict()
data=data.replace({"Cabin":count})

data1['Cabin']=data1['Cabin'].fillna(0)
count1=data1['Cabin'].value_counts().to_dict()
data1=data1.replace({"Cabin":count1})

train1=data.drop(['PassengerId','Survived','Name','Ticket','Fare'],axis=1)
test1=data1.drop(['PassengerId','Name','Ticket','Fare'],axis=1)
test2=data2.drop(['PassengerId'],axis=1)



from sklearn.linear_model import LogisticRegression

#train1= train1.Cabin(pd.to_numeric)
data=data.fillna(data.mean())
data1=data1.fillna(data1.mean())
train1=train1.fillna(train1.mean())
test1=test1.fillna(test1.mean())

reg=LogisticRegression()
reg.fit(train1,data['Survived'])

print("Train data score")
print(reg.score(train1,data['Survived']))
print("Test data score")
print(reg.score(test1,test2))
