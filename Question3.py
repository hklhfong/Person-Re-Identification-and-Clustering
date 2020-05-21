# -*- coding: utf-8 -*-
"""
Created on Wed May 20 16:37:30 2020



@author: user
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
import matplotlib.pylab as plt
from collections import Counter 

import glob
import os
import cv2


train_y = pd.read_csv(r"C:\Users\user\Downloads\Assignment_1B_Data\Data\Q3\Q3\Train_Data\Train.csv")
binary_maps = r"C:\Users\user\Downloads\Assignment_1B_Data\Data\Q3\Q3\Train_Data\Binary_Maps\*.png"
originals = r"C:\Users\user\Downloads\Assignment_1B_Data\Data\Q3\Q3\Train_Data\Originals\*.png"
parsed = r"C:\Users\user\Downloads\Assignment_1B_Data\Data\Q3\Q3\Train_Data\Parsed\*.png"
data_binary_maps = []
data_originals = []
data_parsed = []

def read_file(file_name, data):
    files = glob.glob(file_name)
    files.sort()
    for f in files:
        d = {}
        head, tail = os.path.split(f)
        parts = tail.split('_')
        if (len(parts) == 2):
            d['label'] = int(parts[0])
            d['image'] = cv2.imread(f)
            data.append(d)
        else:
            print('Could not load: ' + f + '! Incorrectly formatted filename')
            
read_file(binary_maps, data_binary_maps)
# read_file(originals, data_originals)
# read_file(parsed, data_parsed)


