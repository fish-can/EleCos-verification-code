#!/usr/bin/env python
# coding: utf-8

# In[0]:


import numpy as np
import os
import csv
import cv2
from keras.models import load_model
from keras.models import Model
from keras import backend as K


# In[1]:


LETTERSTR = "0123456789"


# In[2]:


print("Loading test data...")
testcsv = open('./img/test/labeled.csv', 'r', encoding='utf8')
test_data = np.stack([np.array(cv2.imread(
    "./img/test/" + row[0] + ".png"))/255.0 for row in csv.reader(testcsv)])
testcsv = open('./img/test/labeled.csv', 'r', encoding='utf8')
test_label = [row[1] for row in csv.reader(testcsv)]
print("Loading model...")
K.clear_session()
model = load_model("./model/elecos_model.h5")
print("Predicting...")
prediction = model.predict(test_data)


# In[3]:


count = 0
total = 100
for i in range(total):
    res = ""
    for j in range(6):
        index = 0
        maxi = 0
        for k in range(10):
            if prediction[j][i][k] > maxi:
                maxi = prediction[j][i][k]
                index = k
        res += str(index)
    if test_label[i] == res:
        count += 1


# In[4]


print("Correct rate: "+str(count/total))
