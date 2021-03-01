import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


def plot_reg(X, y, beta):
    '''
    function to plot decision boundary
    '''
    # labelled observations
    x_0 = x_test[np.where(predictions_Y == 0.0)]
    x_1 = x_test[np.where(predictions_Y == 1.0)]

    # plotting points with diff color for diff label
    plt.scatter([x_0[:, 1]], [x_0[:, 2]], c='b', label='y = 0')
    plt.scatter([x_1[:, 1]], [x_1[:, 2]], c='r', label='y = 1')

    # plotting decision boundary
    x1 = np.arange(0, 1, 0.1)
 #   x2 = -(beta[0, 0] + beta[0, 1] * x1) / beta[0, 2]
#    plt.plot(x1, x2, c='k', label='reg line')

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

dataset = np.loadtxt('C:/Users/hml76/OneDrive/바탕 화면/2019학년도 4학년1학기/인공지능/과제/과제2/pima-indians-diabetes.data_utf.csv', delimiter=',', encoding='cp949')

x_train = dataset[:,0:8]
y_train = dataset[:,8]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=0)
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

print(len(predictions_X))
print('tp : ',tp,'\ntn : ', tn,'\nfp : ', fp,'\nfn : ', fn)
print('Accuracy(calcuated by confusion matrix) : ', (tp + tn)/(tp + tn + fp + fn))
print('ACC : ', metrics.accuracy_score(y_test,predictions_Y))
beta = 0
plot_reg(x_test, predictions_Y, beta)

#graph
'''
plt.figure(figsize=(15,15))

plt.plot()
plt.scatter(x_test,predictions_Y)
#plt.plot(epochs, acc, label='acc')
#plt.plot(epochs, loss, label='loss')
plt.title('Logistic Regression')
#plt.legend()
plt.xlabel('')
plt.ylabel('')

plt.show()
'''