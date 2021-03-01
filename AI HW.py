import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

dataset = np.loadtxt('C:/Users/hml76/OneDrive/바탕 화면/2019학년도 4학년1학기/인공지능/과제/과제2/pima-indians-diabetes.data_utf.csv', delimiter=',', encoding='cp949')

x_train = dataset[:,0:8]
y_train = dataset[:,8]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.4, random_state=0)
model = LogisticRegression()
model.fit(x_train, y_train)

# make probability predictions with the model
predictions_X = model.predict(x_test)
predictions_Y = model.predict(x_test)

for i in range(len(predictions_X)):
    if predictions_X[i] > 0.5:
       predictions_Y[i] = 1
    else:
        predictions_Y[i] = 0

tp = 0
tn = 0
fp = 0
fn = 0
n = 0

for j in range(len(predictions_X)):
    if predictions_Y[j] == 1:
        if predictions_Y[j] == y_test[j]:
            tp = tp + 1
        else:
            fp = fp + 1
    elif predictions_Y[j] == 0 :
        n = n + 1
        if predictions_Y[j] == y_test[j]:
            tn = tn + 1
        else:
            fn = fn + 1

print('total parameters of confusion matrix(tp,tn,fp,fn) = ' ,len(predictions_X))
print('tp : ',tp,'\ntn : ', tn,'\nfp : ', fp,'\nfn : ', fn)
print('Accuracy(calculated by confusion matrix)       : %.2f' % (100*(tp + tn)/(tp + tn + fp + fn)) + '%')
print('Accuracy(calculated by accuracy_score library) : %.2f' % (100*metrics.accuracy_score(y_test,predictions_Y)) + '%')
