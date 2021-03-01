from numpy import loadtxt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.optimizers import adam

# load the dataset
dataset = loadtxt('C:/Users/hml76/OneDrive/바탕 화면/2019학년도 4학년1학기/지능시스템/과제/pima-indians-diabetes.data_utf.csv', delimiter=',', encoding='cp949')

# split into input (X) and output (y) variables
x = dataset[:,0:8]
y = dataset[:,8]

print(x)
print(y)

x, X, y, Y = train_test_split(x, y, test_size=0.3, random_state=0)

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(16, activation='sigmoid'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(4, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(2, activation='sigmoid'))   #random seed => random 값 초기값이 똑같게 해주는 함수
model.add(Dense(1, activation='sigmoid'))
model.summary()

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

# fit the keras model on the dataset
history = model.fit(x, y, epochs=300, batch_size=20)

# evaluate the keras model
_, accuracy = model.evaluate(x, y)
for i in model.layers:
    print(i.get_weights())
print('Accuracy: %.2f' % (accuracy*100))

# make probability predictions with the model
predictions = model.predict(X)
# round predictions
rounded = [round(x[0]) for x in predictions]


#graph
acc = history.history['acc']
loss = history.history['loss']
#val_acc = history.history['val_acc']
#val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.figure(figsize=(15,15))
plt.plot(epochs, acc, label='acc')
plt.plot(epochs, loss, label='loss')
#plt.plot(epochs, val_acc, label='val_acc')
#plt.plot(epochs, val_loss, label='val_loss')
plt.title('Sigmoid _ Training and validation loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

#activation 함수를 sigmoid 로 바꿨을때 성능 비교, Vanishing gradient가 실제로 되는지(hidden layer추가 -> 8,9개), 젤 앞단 레이어의 출력, 여기서 weight값 출력하는 부분이 없음. 그부분만 추가
