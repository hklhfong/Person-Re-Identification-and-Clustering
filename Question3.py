# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:37:30 2020



@author: user
"""
import glob
import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from keras.layers import Dense, Conv2D, MaxPool2D , Flatten, GlobalAveragePooling2D,Dropout

import numpy
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras.applications import Xception 
from keras.layers import Input
from keras.models import Model

from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder

csv = pd.read_csv(r"C:\Users\user\Downloads\Assignment_1B_Data\Data\Q3\Q3\Train_Data\Train.csv")
csv_test = pd.read_csv(r"C:\Users\user\Downloads\Assignment_1B_Data\Data\Q3\Q3\Test_Data\Test.csv")
print(csv.head())
print(csv.describe())
print(csv_test.head())
print(csv_test.describe())

binary_maps = r"C:\Users\user\Downloads\Assignment_1B_Data\Data\Q3\Q3\Train_Data\Binary_Maps"
originals = r"C:\Users\user\Downloads\Assignment_1B_Data\Data\Q3\Q3\Train_Data\Originals\*.png"
parsed = r"C:\Users\user\Downloads\Assignment_1B_Data\Data\Q3\Q3\Train_Data\Parsed\*.png"
originals_test = r"C:\Users\user\Downloads\Assignment_1B_Data\Data\Q3\Q3\Test_Data\Originals\*.png"

X = csv["filename"]
y = csv[['gender', 'pose', 'tortyp', 'torcol', 'torcol2', 'torcol3',
       'tortex', 'legtyp', 'legcol', 'legcol2', 'legcol3', 'legtex',
       'luggage']]
X_test = csv_test["filename"]
y_test = csv_test[['gender', 'pose', 'tortyp', 'torcol', 'torcol2', 'torcol3',
       'tortex', 'legtyp', 'legcol', 'legcol2', 'legcol3', 'legtex',
       'luggage']]

csv_total = pd.concat([csv, csv_test])
data_binary_maps = []
data_parsed = []
y_total = csv_total[['gender', 'pose', 'tortyp', 'torcol', 'torcol2', 'torcol3',
       'tortex', 'legtyp', 'legcol', 'legcol2', 'legcol3', 'legtex',
       'luggage']]

ohe = OrdinalEncoder()
ohe.fit(y_total)
y = ohe.transform(y)
y_test = ohe.transform(y_test)


files = glob.glob(originals)
files.sort()
data_originals = []
for f in files:
    d = {}
    head, tail = os.path.split(f)
    d['label'] = tail
    image = cv2.imread(f)
    d['image'] = cv2.resize(image, (256,128), interpolation=cv2.INTER_CUBIC)
    data_originals.append(d)
X = np.array([d['image'] for d in data_originals])/255

files = glob.glob(originals_test)
files.sort()
data_originals_test = []
for f in files:
    d = {}
    head, tail = os.path.split(f)
    d['label'] = tail
    image = cv2.imread(f)
    d['image'] = cv2.resize(image, (256,128), interpolation=cv2.INTER_CUBIC)
    data_originals_test.append(d)
X_test = np.array([d['image'] for d in data_originals_test])/255

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)

#separeate in each columns
y_train = np.hsplit(y_train, 13)
y_val  = np.hsplit(y_val,13)
y_test = np.hsplit(y_test,13)

loss_list = {'gender': 'sparse_categorical_crossentropy',
                'pose': 'sparse_categorical_crossentropy',
                'tortyp': 'sparse_categorical_crossentropy',
                'torcol': 'sparse_categorical_crossentropy',
                'torcol2': 'sparse_categorical_crossentropy',
                'torcol3': 'sparse_categorical_crossentropy',
                'tortex': 'sparse_categorical_crossentropy',
                'legtyp': 'sparse_categorical_crossentropy',
                'legcol': 'sparse_categorical_crossentropy',
                'legcol2': 'sparse_categorical_crossentropy',
                'legcol3': 'sparse_categorical_crossentropy',
                'legtex': 'sparse_categorical_crossentropy',
                'luggage': 'sparse_categorical_crossentropy'}

test_metrics = {'gender': 'accuracy',
                'pose': 'accuracy',
                'tortyp': 'accuracy',
                'torcol': 'accuracy',
                'torcol2': 'accuracy',
                'torcol3': 'accuracy',
                'tortex': 'accuracy',
                'legtyp': 'accuracy',
                'legcol': 'accuracy',
                'legcol2': 'accuracy',
                'legcol3': 'accuracy',
                'legtex': 'accuracy',
                'luggage': 'accuracy'}
dd = 0.1

def multi_model(loss_list,test_metrics,dd):
    
    base_model = Xception(weights='imagenet', include_top=False)

    #Speed up the training time by freezing model
    for layer in base_model.layers[:]:
       layer.trainable = False

    
    model_input = Input(shape=(128, 256, 3))
    x = base_model(model_input)
    x = GlobalAveragePooling2D()(x)
    
    x = Dense(256, activation='relu')(x)
    x = Dropout(dd)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(dd)(x)

    y1 = Dense(128, activation='relu')(x)
    y1 = Dropout(dd)(y1)
    y1 = Dense(64, activation='relu')(y1)
    y1 = Dropout(dd)(y1)
    
    y2 = Dense(128, activation='relu')(x)
    y2 = Dropout(dd)(y2)
    y2 = Dense(64, activation='relu')(y2)
    y2 = Dropout(dd)(y2)
    
    y3 = Dense(128, activation='relu')(x)
    y3 = Dropout(dd)(y3)
    y3 = Dense(64, activation='relu')(y3)
    y3 = Dropout(dd)(y3)

    y4 = Dense(128, activation='relu')(x)
    y4 = Dropout(dd)(y4)
    y4 = Dense(64, activation='relu')(y4)
    y4 = Dropout(dd)(y4)
    
    y5 = Dense(128, activation='relu')(x)
    y5 = Dropout(dd)(y5)
    y5 = Dense(64, activation='relu')(y5)
    y5 = Dropout(dd)(y5)
    
    y6 = Dense(128, activation='relu')(x)
    y6 = Dropout(dd)(y6)
    y6 = Dense(64, activation='relu')(y6)
    y6 = Dropout(dd)(y6)    
    
    y7 = Dense(128, activation='relu')(x)
    y7 = Dropout(dd)(y7)
    y7 = Dense(64, activation='relu')(y7)
    y7 = Dropout(dd)(y7) 
    
    y8 = Dense(128, activation='relu')(x)
    y8 = Dropout(dd)(y8)
    y8 = Dense(64, activation='relu')(y8)
    y8 = Dropout(dd)(y8) 
    
    y9 = Dense(128, activation='relu')(x)
    y9 = Dropout(dd)(y9)
    y9 = Dense(64, activation='relu')(y9)
    y9 = Dropout(dd)(y9) 
    
    y10 = Dense(128, activation='relu')(x)
    y10 = Dropout(dd)(y10)
    y10 = Dense(64, activation='relu')(y10)
    y10 = Dropout(dd)(y10) 
    
    y11 = Dense(128, activation='relu')(x)
    y11 = Dropout(dd)(y11)
    y11 = Dense(64, activation='relu')(y11)
    y11 = Dropout(dd)(y11) 
    
    y12 = Dense(128, activation='relu')(x)
    y12 = Dropout(dd)(y12)
    y12 = Dense(64, activation='relu')(y12)
    y12 = Dropout(dd)(y12) 
    
    y13 = Dense(128, activation='relu')(x)
    y13 = Dropout(dd)(y13)
    y13 = Dense(64, activation='relu')(y13)
    y13 = Dropout(dd)(y13) 
    
    #connect all the heads to their final output layers
    y1 = Dense(4, activation='softmax',name= 'gender')(y1)
    y2 = Dense(6, activation='softmax',name= 'pose')(y2)
    y3 = Dense(4, activation='softmax',name= 'tortyp')(y3)
    y4 = Dense(12, activation='softmax',name= 'torcol')(y4)
    y5 = Dense(12, activation='softmax',name= 'torcol2')(y5)
    y6 = Dense(12, activation='softmax',name= 'torcol3')(y6)
    y7 = Dense(9, activation='softmax',name= 'tortex')(y7)
    y8 = Dense(4, activation='softmax',name= 'legtyp')(y8)
    y9 = Dense(12, activation='softmax',name= 'legcol')(y9)
    y10 = Dense(12, activation='softmax',name= 'legcol2')(y10)
    y11 = Dense(12, activation='softmax',name= 'legcol3')(y11)
    y12 = Dense(9, activation='softmax',name= 'legtex')(y12)
    y13 = Dense(4, activation='softmax',name= 'luggage')(y13)
    
    
    model = Model(inputs=model_input, outputs=[ y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13])    
    model.compile(loss=loss_list, optimizer='Adam', metrics=test_metrics)
    return model

multi_model = multi_model(loss_list,test_metrics,dd)
multi_model.summary()

callback = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)]
multi_model.fit(X_train,y_train,batch_size = 32,epochs = 10, verbose = 2, callbacks = callback,validation_data=(X_val, y_val) )
score = multi_model.evaluate(X_test,y_test,verbose = 0)
print(dict(zip(multi_model.metrics_names, score)))