#!/usr/bin/env python
# coding: utf-8

# In[0]:


import numpy as np
import cv2
import csv
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard


# In[2]:


LETTERSTR = "0123456789"


def toonehot(text):
    labellist = []
    for letter in text:
        onehot = [0 for _ in range(10)]
        num = LETTERSTR.find(letter)
        onehot[num] = 1
        labellist.append(onehot)
    return labellist


# In[3]:


# Create CNN Model
print("Creating CNN model...")
tensor_in = Input((32, 120, 3))
tensor_out = tensor_in
tensor_out = Conv2D(filters=32, kernel_size=(
    3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = Conv2D(filters=32, kernel_size=(3, 3),
                    activation='relu')(tensor_out)
tensor_out = BatchNormalization()(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Conv2D(filters=64, kernel_size=(
    3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = Conv2D(filters=64, kernel_size=(3, 3),
                    activation='relu')(tensor_out)
tensor_out = BatchNormalization()(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Conv2D(filters=128, kernel_size=(
    3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = Conv2D(filters=128, kernel_size=(3, 3),
                    activation='relu')(tensor_out)
tensor_out = BatchNormalization()(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Conv2D(filters=256, kernel_size=(
    3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = Conv2D(filters=256, kernel_size=(
    3, 3), padding='same', activation='relu')(tensor_out)
tensor_out = BatchNormalization()(tensor_out)
tensor_out = MaxPooling2D(pool_size=(2, 2))(tensor_out)
tensor_out = Flatten()(tensor_out)
tensor_out = Dropout(0.5)(tensor_out)
tensor_out = [Dense(10, name='digit1', activation='softmax')(tensor_out),
              Dense(10, name='digit2', activation='softmax')(tensor_out),
              Dense(10, name='digit3', activation='softmax')(tensor_out),
              Dense(10, name='digit4', activation='softmax')(tensor_out),
              Dense(10, name='digit5', activation='softmax')(tensor_out),
              Dense(10, name='digit6', activation='softmax')(tensor_out)]
model = Model(inputs=tensor_in, outputs=tensor_out)
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()


# In[4]:


print("Reading training data...")
traincsv = open('./img/train/labeled.csv', 'r', encoding='utf8')
train_data = np.stack([np.array(cv2.imread(
    "./img/train/" + row[0] + ".png"))/255.0 for row in csv.reader(traincsv)])
traincsv = open('./img/train/labeled.csv', 'r', encoding='utf8')
read_label = [toonehot(row[1]) for row in csv.reader(traincsv)]
train_label = [[] for _ in range(6)]
for arr in read_label:
    for index in range(6):
        train_label[index].append(arr[index])
train_label = [arr for arr in np.asarray(train_label)]
print("Shape of train data:", train_data.shape)


# In[5]:


print("Reading validation data...")
valicsv = open('./img/valid/labeled.csv', 'r', encoding='utf8')
vali_data = np.stack([np.array(cv2.imread(
    "./img/valid/" + row[0] + ".png"))/255.0 for row in csv.reader(valicsv)])
valicsv = open('./img/valid/labeled.csv', 'r', encoding='utf8')
read_label = [toonehot(row[1]) for row in csv.reader(valicsv)]
vali_label = [[] for _ in range(6)]
for arr in read_label:
    for index in range(6):
        vali_label[index].append(arr[index])
vali_label = [arr for arr in np.asarray(vali_label)]
print("Shape of valid data:", vali_data.shape)


# In[6]:


filepath = "./model/elecos_model.h5"
checkpoint = ModelCheckpoint(
    filepath, monitor='val_digit6_acc', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_digit6_acc',
                          patience=5, verbose=1, mode='auto')
tensorBoard = TensorBoard(log_dir="./logs", histogram_freq=1)
callbacks_list = [checkpoint, earlystop, tensorBoard]
model.fit(train_data, train_label, batch_size=400, epochs=100, verbose=2,
          validation_data=(vali_data, vali_label), callbacks=callbacks_list)
