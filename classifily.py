#!/usr/bin/env python
# encoding=utf-8

import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import RMSprop

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

X_train_gray = np.zeros(shape=(X_train.shape[:-1]))
for num, data in enumerate(X_train):
    im = Image.fromarray(np.uint8(data)).convert('L')
    X_train_gray[num] = np.array(im)


X_test_gray = np.zeros(shape=(X_train.shape[:-1]))
for num, data in enumerate(X_test):
    im = Image.fromarray(np.uint8(data)).convert('L')
    X_test_gray[num] = np.array(im)


X_train = X_train_gray.reshape(X_train.shape[0], -1) / 255
X_test = X_test_gray.reshape(X_test.shape[0], -1) / 255
Y_train = np_utils.to_categorical(Y_train, nb_classes=10)
Y_test = np_utils.to_categorical(Y_test, nb_classes=10)

model = Sequential()

model.add(Dense(input_dim=1024, output_dim=32))
model.add(Activation('relu'))

model.add(Dense(output_dim=32))
model.add(Activation('relu'))

model.add(Dense(output_dim=32))
model.add(Activation('relu'))

model.add(Dense(output_dim=32))
model.add(Activation('relu'))

model.add(Dense(output_dim=32))
model.add(Activation('relu'))

model.add(Dense(output_dim=32))
model.add(Activation('relu'))

model.add(Dense(output_dim=32))
model.add(Activation('relu'))

model.add(Dense(output_dim=32))
model.add(Activation('relu'))

model.add(Dense(output_dim=32))
model.add(Activation('relu'))

model.add(Dense(output_dim=32))
model.add(Activation('relu'))


model.add(Dense(output_dim=10))
model.add(Activation('softmax'))

rmsprop = RMSprop(lr=0.05)

model.compile(loss='categorical_crossentropy',
    optimizer=rmsprop,
    metrics=['accuracy'])

model.fit(X_train, Y_train, nb_epoch=20, batch_size=32)

loss, accuracy = model.evaluate(X_test, Y_test)
print('loss: ', loss)
print('accuracy: ', accuracy)