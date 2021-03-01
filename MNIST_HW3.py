#1. convolution layer의 feature map의 크기는 4x4, 개수는 16개 사용
#2. activation function은 relu
#3 maxpooling을 사용하고, pooling filter size는 2x2

from __future__ import print_function           #필요한 라이브러리들
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K

batch_size = 128            #설정할 배치크기, 클래스 갯수, Epoch 횟수
num_classes = 10
epochs = 5


img_rows, img_cols = 28, 28   # 인풋 이미지 크기
(x_train, y_train), (x_test, y_test) = mnist.load_data()    #mnist에 필요한 데이터를 트레이닝 셋과 테스트 셋에 집어넣는다.
#print(x_train = x_train.reshape(5,5))

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1) # 패라미터 : 이미지 개수, width, height, channel 로 구성
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')     #데이터 타입 변경
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')        #샘플 갯수 출력
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)  #10개 값을 one-hot encoding  해주는 함수
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_test)   #보면 한 row당 하나의 1만 있는 것을 볼 수 있다.

model = Sequential()        # 딥러닝 모델 Sequential 하게 생성. 각 레이어를 더하는 방식으로 구현.

model.add(Conv2D(16, kernel_size=(2, 2),                         # 필터로 특징을 뽑아주는 컨볼루션(Convolution) 레이어, 16개의 필터, 커널 크기의 행렬
 #                padding=0,
                 activation='relu',                              #활성화 함수 : Relu
                 input_shape=input_shape))                       #kernel_size = filter_size 2,2 5,5 => 지나치면 이미지는 27*27 크기로 변한다.
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)     #shape 확인(디버깅 용)
model.add(MaxPooling2D(pool_size=(3, 3)))                              #pooling layer size : 3*3, 이미지는 9*9 크기로 변한다.
model.add(Dropout(0.25))                                               #Dropout 크기 : 25%
model.add(Conv2D(16, (2, 2), activation='relu'))                        #두 번째 컨볼루션 레이어, 이미지 크기는 8*8로 변한다.
model.add(MaxPooling2D(pool_size=(2, 2)))                               #풀링 레이어를 거쳐서 이미지 크기는 4*4로 변한다.
model.add(Dropout(0.25))

model.summary()                                                         #모델 경과를 나타내 줌.
model.add(Flatten())                                                    #인풋 이미지를 일차원 배열로 펴줌.
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,               #loss율 측정 변수, 옵티마이저 Adadelta, 정확도 관련 변수
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])


model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test)) #모델 학습, 위에서 선언해준 변수들 학습 실시
score = model.evaluate(x_test, y_test, verbose=0)                                                               #모엘 평가
print('Test loss:', score[0])
print('Test accuracy:', score[1])
n=39
print('The Answer is ', model.predict_classes(x_test[n].reshape((1, 28, 28, 1))))                               #x_test에 있는 레이블 값 나타내줌

plt.imshow(x_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')                                    #해당 값 근처에 있는 그림을 다시 살려서 출력해줌
plt.show()


import random

predicted_result = model.predict(x_test)                                                                        #예측 실패한 값들 나타내 주는 함수
predicted_labels = np.argmax(predicted_result, axis=1)
test_labels = np.argmax(y_test, axis=1)
wrong_result = []

for n in range(0, len(test_labels)):
    if predicted_labels[n] != test_labels[n]:
        wrong_result.append(n)

samples = random.choices(population=wrong_result, k=16)
count = 0
nrows = ncols = 4

plt.figure(figsize=(12,8))

for n in samples:
    count += 1
    plt.subplot(nrows, ncols, count)
    plt.imshow(x_test[n].reshape(28, 28), cmap='Greys', interpolation='nearest')
    tmp = "Label:" + str(test_labels[n]) + ", Prediction:" + str(predicted_labels[n])
    plt.title(tmp)

plt.tight_layout()
plt.show()
