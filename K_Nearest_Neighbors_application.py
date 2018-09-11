import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.fillna(55,inplace=True)
df.replace('?', -9999, inplace=True)
df.drop(['id'], 1, inplace=True)
#print(df.head())

# x is features
X = np.array(df.drop(['class'],1))
# y is for labels
Y = np.array(df['class'])

X_train,X_test,Y_train,Y_test = cross_validation.train_test_split(X,Y,test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train,Y_train)

accuracy = clf.score(X_test,Y_test)
print(accuracy)

examples_measure = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,4,5,3,2,1],[4,2,5,1,4,2,1,2,1]])
examples_measure = examples_measure.reshape(len(examples_measure),-1)

prediction = clf.predict(examples_measure)
print(prediction)