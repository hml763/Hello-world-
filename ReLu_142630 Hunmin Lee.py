import numpy as np
from numpy import loadtxt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, precision_recall_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score

# load the dataset
dataset = loadtxt('C:/Users/hml76/OneDrive/바탕 화면/2019학년도 4학년1학기/지능시스템/과제/pima-indians-diabetes.data_utf.csv', delimiter=',', encoding='cp949')

# split into input (X) and output (y) variables
x_train = dataset[:,0:8]
y_train = dataset[:,8]

print(x_train)
print(y_train)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=0)  #x,y = train data, X,Y = Test data, test data : 10%, train data : 90%

# define the keras model
model = Sequential()
#model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(12, input_dim=8, activation='relu'))
#model.add(Dropout(0.99))        #99%정도를 dropout함.
#model.add(Dense(64, kernel_initializer='random_normal', activation='relu'))
model.add(Dense(50, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dense(40, kernel_initializer='random_normal', activation='relu'))
#model.add(Dropout(0.99))
model.add(Dense(50, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dense(40, kernel_initializer='random_normal',activation='relu'))
#model.add(Dropout(0.99))
model.add(Dense(50, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dense(40, kernel_initializer='random_normal', activation='relu'))
#model.add(Dropout(0.99))
model.add(Dense(50, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dense(30, kernel_initializer='random_normal', activation='relu'))
#model.add(Dropout(0.99))
model.add(Dense(50, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dense(20, kernel_initializer='random_normal', activation='relu'))
#model.add(Dropout(0.00))
model.add(Dense(50, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dense(10, kernel_initializer='random_normal', activation='relu'))
#model.add(Dropout(0.99))
model.add(Dense(50, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(50, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(15, activation='relu'))
#model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
#model.add(Dense(1, kernel_initializer='glorot_normal', activation='sigmoid'))
#he_uniform, he_normal, random_uniform, random_normal

model.summary()

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# fit the keras model on the dataset
history = model.fit(x_train, y_train, epochs=300, batch_size=50, validation_split=0.15)#batch_size=700) # batch_size는 학습될때 50씩 증가한다.

# evaluate the keras model
_, train_accuracy = model.evaluate(x_train, y_train)        #y_train으로 정확도랑 loss뽑아낸다
_, test_accuracy = model.evaluate(x_test,y_test)             #y_test로 정확도 뽑아냄, 출력

print('train_accuracy: %.2f' % (train_accuracy*100) + '%')
print('test_accuracy : %.2f' % (test_accuracy*100) + '%')

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

print('\ntotal parameters of confusion matrix(tp,tn,fp,fn) = ' ,len(predictions_X))
print('tp : ',tp,'\ntn : ', tn,'\nfp : ', fp,'\nfn : ', fn,'\n')
# round predictions
print('Precision                      : %.2f' % (100*tp/(tp + tn)))
print('Recall                         : %.2f' % (100*tp/(tp + fn)))
print('Specificity                    : %.2f' % (100*tn/n))
print('False positive rate            : %.2f' % (100*fp/n))
print("AUC: Area Under Curve          : {}".format(100*roc_auc_score(y_test, predictions_Y)))
print('Average precision-recall curve : {}'.format(100*average_precision_score(y_test, predictions_Y)),'\n')

rounded = [round(x_train[0]) for x_train in predictions_X]

#graph
acc = history.history['acc']
loss = history.history['loss']

val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.figure(figsize=(15,15))

plt.subplot(2,2,1)
plt.plot(epochs, acc, label='acc')
plt.plot(epochs, loss, label='loss')
#plt.plot(epochs, val_acc, label='val_acc')
#plt.plot(epochs, val_loss, label='val_loss')
plt.title('RELU _ Training and validation loss')
#plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')

print('Roc curve 꺾이는 점              : ', roc_curve(y_test, predictions_Y)) #roc 커브에서 어느 부분에서 꺾이는 지 알수 있음.
print('Precision recall curve 꺾이는 점 : ', precision_recall_curve(y_test, predictions_Y)) #precision recall 커브에서 어느 부분에서 꺾이는 지 알수 있음.

#plt.subplot(1,1,1)
fpr, tpr, _ = roc_curve(y_test, predictions_Y)
plt.subplot(2, 2, 2)
#f.set_size_inches((8, 4))
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')

rec, prec, _ = precision_recall_curve(y_test, predictions_Y)
plt.subplot(2, 2, 3)
plt.plot(rec, prec)
plt.plot([0, 1], [1, 0], linestyle='--')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall curve')

plt.show()
