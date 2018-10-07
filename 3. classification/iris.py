import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,precision_score,recall_score
from itertools import product
iris = datasets.load_iris()

x = iris["data"]
y = iris["target"]

from sklearn.preprocessing import StandardScaler 
sc = StandardScaler() 
x_scaled = sc.fit_transform(x)

from sklearn.cross_validation import train_test_split #dataseti test ve train olarak ikiye böler
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.33, random_state=42) 

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=42)
lr.fit(x_train, y_train)

y_pred_lr = lr.predict(x_test)

#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred_lr)
print("LR")
print(cm)
print("Accuracy = " + str(accuracy_score(y_test, y_pred_lr)))
print("Precision = " + str(precision_score(y_test, y_pred_lr, average="weighted")))
print("Recall = " + str(recall_score(y_test, y_pred_lr, average="weighted")))
print("F1 = " + str(f1_score(y_test, y_pred_lr, average="weighted")))

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

y_pred_knn = knn.predict(x_test)

#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred_knn)
print("KNN")
print(cm)
print("Accuracy = " + str(accuracy_score(y_test, y_pred_knn)))
print("Precision = " + str(precision_score(y_test, y_pred_knn, average="weighted")))
print("Recall = " + str(recall_score(y_test, y_pred_knn, average="weighted")))
print("F1 = " + str(f1_score(y_test, y_pred_knn, average="weighted")))

from sklearn.svm import SVC
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(x_train, y_train)

y_pred_svm_rbf = svm_rbf.predict(x_test)

#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred_svm_rbf)
print("SVM RBF")
print(cm)
print("Accuracy = " + str(accuracy_score(y_test, y_pred_svm_rbf)))
print("Precision = " + str(precision_score(y_test, y_pred_svm_rbf, average="weighted")))
print("Recall = " + str(recall_score(y_test, y_pred_svm_rbf, average="weighted")))
print("F1 = " + str(f1_score(y_test, y_pred_svm_rbf, average="weighted")))

svm_poly = SVC(kernel='poly', degree=5)
svm_poly.fit(x_train, y_train)

y_pred_svm_poly = svm_poly.predict(x_test)

#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred_svm_poly)
print("SVM POLY")
print(cm)
print("Accuracy = " + str(accuracy_score(y_test, y_pred_svm_poly)))
print("Precision = " + str(precision_score(y_test, y_pred_svm_poly, average="weighted")))
print("Recall = " + str(recall_score(y_test, y_pred_svm_poly, average="weighted")))
print("F1 = " + str(f1_score(y_test, y_pred_svm_poly, average="weighted")))

svm_sigmoid = SVC(kernel='sigmoid')
svm_sigmoid.fit(x_train, y_train)

y_pred_svm_sigmoid = svm_sigmoid.predict(x_test)

#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred_svm_sigmoid)
print("SVM SİGMOİD")
print(cm)
print("Accuracy = " + str(accuracy_score(y_test, y_pred_svm_sigmoid)))
print("Precision = " + str(precision_score(y_test, y_pred_svm_sigmoid, average="weighted")))
print("Recall = " + str(recall_score(y_test, y_pred_svm_sigmoid, average="weighted")))
print("F1 = " + str(f1_score(y_test, y_pred_svm_sigmoid, average="weighted")))

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(x_train, y_train)

y_pred_gnb = gnb.predict(x_test)

#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred_gnb)
print("GNB")
print(cm)
print("Accuracy = " + str(accuracy_score(y_test, y_pred_gnb)))
print("Precision = " + str(precision_score(y_test, y_pred_gnb, average="weighted")))
print("Recall = " + str(recall_score(y_test, y_pred_gnb, average="weighted")))
print("F1 = " + str(f1_score(y_test, y_pred_gnb, average="weighted")))

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion="entropy")
dtc.fit(x_train, y_train)

y_pred_dtc = dtc.predict(x_test)

#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred_dtc)
print("DTC")
print(cm)
print("Accuracy = " + str(accuracy_score(y_test, y_pred_dtc)))
print("Precision = " + str(precision_score(y_test, y_pred_dtc, average="weighted")))
print("Recall = " + str(recall_score(y_test, y_pred_dtc, average="weighted")))
print("F1 = " + str(f1_score(y_test, y_pred_dtc, average="weighted")))

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(criterion="entropy", n_estimators=10)
rfc.fit(x_train, y_train)

y_pred_rfc = rfc.predict(x_test)

#CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred_rfc)
print("RFC")
print(cm)
print("Accuracy = " + str(accuracy_score(y_test, y_pred_rfc)))
print("Precision = " + str(precision_score(y_test, y_pred_rfc, average="weighted")))
print("Recall = " + str(recall_score(y_test, y_pred_rfc, average="weighted")))
print("F1 = " + str(f1_score(y_test, y_pred_rfc, average="weighted")))
