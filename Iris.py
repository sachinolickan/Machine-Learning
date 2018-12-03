from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  accuracy_score
import pandas as pd
iris=datasets.load_iris()
x=iris.data
y=iris.target

columns=iris.target_names
# print(x)
# print(y)


clf=KNeighborsClassifier()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30)

clf.fit(x_train,y_train)

y_predict=clf.predict(x_test)

print(x_test)
print(y_predict)
print("accuracy score:",accuracy_score(y_test,y_predict))



