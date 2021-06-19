# -*- coding: utf-8 -*-
"""
Created on Tue May 25 23:16:15 2021

@author: DELL
"""

#Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
company = pd.read_csv("Company_Data.csv")
company.head()

#Converting the categorical into numeric
company['High'] = company.Sales.map(lambda x: 1 if x>8 else 0)

company['ShelveLoc']=company['ShelveLoc'].astype('category')
company['Urban']=company['Urban'].astype('category')
company['US']=company['US'].astype('category')

#label encoding to convert categorical into numerical
company['ShelveLoc']=company['ShelveLoc'].cat.codes
company['Urban']=company['Urban'].cat.codes
company['US']=company['US'].cat.codes

#Setting x (feature) and y (target)
feature_col = ["CompPrice","Income","Advertising","Population","Price","ShelveLoc","Age","Education","Urban","US"]
#x = company.drop(['Sales','High'],axis=1)

x=company[feature_col]
y=company.High
print(x)
print(y)

#train test split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

#Model building using DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(criterion = 'entropy')
model1=model.fit(x_train,y_train)
pred = model1.predict(x_test)
type(pred)

#evaluating the model(by using crosstab or confusion matrix)
pd.crosstab(y_test,pred)
#or by using confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,pred))

#Accuracy
from sklearn.metrics import classification_report
print(classification_report(y_test,pred))
#Here, we will consider f1-score for accuracy i.e 78%

#plot
from sklearn import tree
tree.plot_tree(model1,filled=True)
tree.plot_tree

















