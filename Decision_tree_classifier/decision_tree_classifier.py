import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sklearn.metrics

data = pd.read_csv('health.csv')
data = data.dropna()
print(data.describe())

predictors = data[['BIO_SEX','HISPANIC','WHITE','BLACK','NAMERICAN','ASIAN',
'age','ALCEVR1','ALCPROBS1','marever1','cocever1','inhever1','cigavail','DEP1',
'ESTEEM1','VIOL1','PASSIST','DEVIANT1','SCHCONN1','GPA1','EXPEL1','FAMCONCT','PARACTV',
'PARPRES']]

target = data.TREG1

prediction_train, prediction_test, target_train, target_test = train_test_split(predictors,target,test_size=.4)

print(prediction_train.shape)
print(prediction_test.shape)
print(target_test.shape)


classifier = DecisionTreeClassifier()
classifier = classifier.fit(prediction_train,target_train)

predictions = classifier.predict(prediction_test)

print(sklearn.metrics.accuracy_score(target_test,predictions))

#to plot decision tree
from sklearn import tree
from io import StringIO
from IPython.display import Image
with open('fruit_classifier.txt','w') as f:
    f = tree.export_graphviz(classifier, out_file=f)