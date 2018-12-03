import sys
import csv
import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from  sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import gender_guesser.detector as gender
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

def read_datasets():
    """ Reads users profile from csv files """
    genuine_users = pd.read_csv("data/users.csv")
    fake_users = pd.read_csv("data/fusers.csv")
    x=pd.concat([genuine_users,fake_users])   
    y=len(genuine_users)*[1] + len(fake_users)*[0]
    return x,y
def predict_gender(name):
    testname=str(name)
    first_name=testname.split(" ")[0]
    gender_code=0 
    listdata=["i" ,"a","y"]
    if testname[len(testname)-1] in listdata:
        gender_code=1
    return gender_code
def gender_guard_code(name,guard_code):
    d = gender.Detector()
    sex=predict_gender(name)
    gender_guard_code=0
    if sex==1  and guard_code==1:
        gender_guard_code=1
    return gender_guard_code



def extract_features(x):
    feature_columns_to_use = ['memory','checkin','profilecount','googleprofile','safeguard','instagram','taggedpost','postcount','commentcount','intro','reviews','events','friendsversary']
    x=x.loc[:,feature_columns_to_use]
    print(x)
    return x
    
def randomforest(X_train,y_train):
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    return clf

def naivebayes(X_train,y_train):
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    return clf
def knn(X_train,y_train):
    clf=KNeighborsClassifier()
    clf.fit(X_train, y_train)
    return clf

def svmclassifier(X_train,y_train):
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    return clf

    
def decisiontreeclassifier(X_train,y_train):
    
    clf= DecisionTreeClassifier(criterion = "gini", random_state = 100,
                               max_depth=3, min_samples_leaf=5)
    clf.fit(X_train, y_train)
    return clf


print ("reading datasets.....\n")
x,y=read_datasets()
print(x)
print(y)
x.describe()

print ("extracting featues.....\n")
x=extract_features(x)
x.describe()
x=np.nan_to_num(x)


###split to training and testing dataset
X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state=44)


###classification using random forest
print("classification using random forest.........")
trained_model = randomforest(X_train, y_train)
print ("Trained model :: ", trained_model)
##making predeicting

y_pred = trained_model.predict(X_test)
###calulating accuracy score
print( 'Classification Accuracy on Test dataset: ' ,accuracy_score(y_test, y_pred))

###confusion matrix

print (" Confusion matrix ", confusion_matrix(y_test, y_pred))
print("........................................................")


print("claasification using naive bayes..............")
trained_model = naivebayes(X_train, y_train)
print ("Trained model :: ", trained_model)
##making predeicting

y_pred = trained_model.predict(X_test)
###calulating accuracy score
print( 'Classification Accuracy on Test dataset: ' ,accuracy_score(y_test, y_pred))

###confusion matrix

print (" Confusion matrix ", (y_test, y_pred))
print("........................................................")


print("claasification using decistion tree classifier..............")
trained_model = decisiontreeclassifier(X_train, y_train)
print ("Trained model :: ", trained_model)
##making predeicting

y_pred = trained_model.predict(X_test)
###calulating accuracy score
print( 'Classification Accuracy on Test dataset: ' ,accuracy_score(y_test, y_pred))

###confusion matrix

print (" Confusion matrix ", confusion_matrix(y_test, y_pred))
print("........................................................")



print("classification using Knn .........")
trained_model = knn(X_train, y_train)
print ("Trained model :: ", trained_model)
##making predeicting

y_pred = trained_model.predict(X_test)
###calulating accuracy score
print( 'Classification Accuracy on Test dataset: ' ,accuracy_score(y_test, y_pred))

###confusion matrix

print (" Confusion matrix ", confusion_matrix(y_test, y_pred))
print("........................................................")



print("classification using svm .........")
trained_model = svmclassifier(X_train, y_train)
print ("Trained model :: ", trained_model)
##making predeicting

y_pred = trained_model.predict(X_test)
###calulating accuracy score
print( 'Classification Accuracy on Test dataset: ' ,accuracy_score(y_test, y_pred))

###confusion matrix

print (" Confusion matrix ", confusion_matrix(y_test, y_pred))
print("........................................................")
