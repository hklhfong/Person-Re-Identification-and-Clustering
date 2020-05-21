# -*- coding: utf-8 -*-
"""
Created on Fri May 15 15:24:24 2020

@author: hofonglaw
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

movie_data = pd.read_csv(r"C:\Users\user\Downloads\Assignment_1B_Data\Data\Q2\movies.csv")
ratings_data = pd.read_csv(r"C:\Users\user\Downloads\Assignment_1B_Data\Data\Q2\ratings.csv")
ratings_data = ratings_data.drop(columns='timestamp')
#use len of movie data index could bring the fastest speed process
#Finding all genres set
genre = movie_data['genres'][0]
x = set(genre.split("|"))
for i in range(len(movie_data.index)):
    temp_genre = movie_data['genres'][i]
    tempX = set(temp_genre.split("|"))
    if not tempX.issubset(x):
        for val in tempX:
            x.add(val)

list_x = list(x)
for i in range(len(list_x)):
    movie_data[list_x[i]] = 0 

#Change value the match the column 
for i in range(len(movie_data.index)):
    temp_genre = movie_data['genres'][i]
    tempX = temp_genre.split("|")
    for temp_X_item in tempX:
        movie_data.loc[i, movie_data.columns.str.contains(temp_X_item)] = 1
       

modify_movie_data = movie_data.drop(columns = ['title','genres'])         
ratings_42 = ratings_data.loc[ratings_data['userId'] == 42]
ratings_42 = pd.merge(ratings_42,modify_movie_data)
for i in range (ratings_42.iloc[:,2].size):
    ratings_42.iloc[i,3:] = ratings_42.iloc[i,2] * ratings_42.iloc[i,3:]
ratings_42_5stars = ratings_42.loc[ratings_42['rating'] == 5]
ratings_42_5stars = ratings_42_5stars.drop(columns = ['userId','movieId','rating'])
ratings_42 = ratings_42.drop(columns = ['userId','movieId','rating'])

ratings_314 = ratings_data.loc[ratings_data['userId'] == 314]
ratings_314 = pd.merge(ratings_314,modify_movie_data)
for i in range (ratings_314.iloc[:,2].size):
    ratings_314.iloc[i,3:] = ratings_314.iloc[i,2] * ratings_314.iloc[i,3:]
ratings_314_5stars = ratings_314.loc[ratings_314['rating'] == 5]
ratings_314_5stars = ratings_314_5stars.drop(columns = ['userId','movieId','rating'])
ratings_314 = ratings_314.drop(columns = ['userId','movieId','rating'])

ratings_444 = ratings_data.loc[ratings_data['userId'] == 444]
ratings_444 = pd.merge(ratings_444,modify_movie_data)
for i in range (ratings_444.iloc[:,2].size):
    ratings_444.iloc[i,3:] = ratings_444.iloc[i,2] * ratings_444.iloc[i,3:]
ratings_444_5stars = ratings_444.loc[ratings_444['rating'] == 5]
ratings_444_5stars = ratings_444_5stars.drop(columns = ['userId','movieId','rating'])
ratings_444 = ratings_444.drop(columns = ['userId','movieId','rating'])

col_titile = ratings_42.columns.values


def Plotbics (ratings):
    bics = []
    for i in range (25):
        gmm = GaussianMixture(i+1, random_state=4)
        gmm.fit(ratings)
        bics.append(gmm.bic(ratings))
        
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(bics)
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('BIC');

def PlotGmm(ratings,ratings_5stars, numCluster):
    gmm = GaussianMixture(numCluster, random_state=4,covariance_type = 'full')
    gmm.fit(ratings)
    scores = gmm.score_samples(ratings)
    labels = gmm.predict(ratings)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(scores)

    five_star_list = gmm.predict(ratings_5stars);
    unique, counts = np.unique(five_star_list, return_counts=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    plt.bar(unique, counts)
    ax.set_xlabel('Clusters Group Index')
    ax.set_ylabel('Amount of 5 stars review have been predicted');
    dictuniqueFiveStars = Counter(dict(zip(unique, counts)))  
    Top3 = dictuniqueFiveStars.most_common(3)  
    y_pos = np.arange(len(col_titile))
    for i in Top3:
            fig = plt.figure(figsize=[30, 25])
            performance = gmm.means_[i[0]]
            plt.bar(y_pos, performance, align='center', alpha=0.5)
            plt.xticks(y_pos, col_titile)
            plt.ylabel('Ratings')
            plt.title('Recommended movie genres with cluster ' + str(i[0])+ " with " + str(i[1]) + " amount of 5 star reviews support")
            plt.show()
            
Plotbics(ratings_42)
PlotGmm(ratings_42,ratings_42_5stars,12)
Plotbics(ratings_314)
PlotGmm(ratings_314,ratings_314_5stars,4)
Plotbics(ratings_444)
PlotGmm(ratings_444,ratings_444_5stars,3)