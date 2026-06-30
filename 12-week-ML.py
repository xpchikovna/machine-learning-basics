#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 10:31:01 2020

@author: farshad.toosi
"""
#from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold

from sklearn.model_selection import ShuffleSplit
from sklearn import datasets
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import math 
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import KBinsDiscretizer

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVR
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier

from scipy.stats import pearsonr 

from sklearn.cluster import KMeans

from sklearn import datasets
from sklearn.decomposition import PCA


def Q1():
    
   

    def preprocess_data(data, features):
        """Preprocess data for clustering"""
        # Select features and drop missing values
       
        
        cluster_data = data[features].copy()
        cluster_data = cluster_data.dropna()
        
        # Convert Sex to numerical (male=0, female=1)
        if 'Sex' in features:
            cluster_data['Sex'] = cluster_data['Sex'].map({'male': 0, 'female': 1})
        
        # Standardize the features
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(cluster_data)
       
        return scaled_data
        
    def plot_clusters(data, labels, k):
        """Plot clusters using PCA for 2D visualization"""
        # Reduce to 2D for visualization
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(data)
        
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(principal_components[:, 0], 
                             principal_components[:, 1], 
                            c=labels, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('2D visualization '+ str(k)+ ' Clusters')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f} variance)')
        plt.show()
        
        # Elbow method function
    def plot_elbow_method(data, max_k=10):
        """Plot elbow method to find optimal k"""
        inertias = []
        k_range = range(1, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(data)
            inertias.append(kmeans.inertia_)
            plot_clusters(scaled_data, kmeans.labels_, k)
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertias, 'bo-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.xticks(k_range)
        plt.grid(True)
        plt.show()
        
    data = pd.read_csv("titanic.csv",encoding = "ISO-8859-1")
    features = ['Survived', 'Sex', 'Age', 'Fare']
    scaled_data = preprocess_data(data, features)
    #print(scaled_data)
    
    plot_elbow_method(scaled_data, 10)
        
#Q1()


def Q2():
    flt = pd.read_csv('movie_metadata.csv', encoding = "ISO-8859-1")
    df = flt[['budget', 'imdb_score', 'duration']].copy()
    
    df['budget'] = df['budget'].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['budget'])
   



    df['imdb_score'] = df['imdb_score'].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['imdb_score'])
   
   

    df['duration'] = df['duration'].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['duration'])
    



    df = df.dropna()

    scalingObj = preprocessing.MinMaxScaler()
    newFLT2 = scalingObj.fit_transform(df)
    
    kmeans = KMeans(n_clusters=3).fit(newFLT2)   
    
    print('2 attrs"  ',kmeans.inertia_) 
    pca = PCA(n_components=2)
    
    principalComponents = pca.fit_transform(newFLT2)

    principalDf = pd.DataFrame(data = principalComponents
    , columns = ['a1', 'a2'])
    
   

    plt.scatter(principalDf['a1'], principalDf['a2'], c=kmeans.labels_ )
    
    plt.show()

#Q2()






def Q3():
    flt = pd.read_csv('Student_Performance.csv', encoding = "ISO-8859-1")
    flt = flt[['Performance Index',  'Hours Studied',	'Previous Scores',	'Extracurricular Activities',	'Sleep Hours',	'Sample Question Papers Practiced']].copy()
    
    flt = flt.dropna()
       
    print(flt)
    
    
    X = flt[[   'Hours Studied',	'Previous Scores',	'Sleep Hours',	'Sample Question Papers Practiced']]
    y = flt[['Performance Index']]
    #print(y)
    model = LinearRegression()
    #Note for regression we should use the below function for cross validation. The test would be 1/5 in the below case.
    shuffle_split = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    
    
    results = cross_validate(model, X, y, cv=shuffle_split, scoring='accuracy',
                        return_train_score=True)  # This gets training scores
    
    
    # The accuracy is above 90% and so we do visualization
    print(np.mean(results['test_score']))
    print(np.mean(results['train_score']))
    
    
    #Visualizations needs two features x and y... all ordinary features became X and class attribute acts as Y
    pca = PCA(n_components=1)
    
    principalComponents = pca.fit_transform(X)
    plt.scatter(principalComponents, y )
    plt.show()

#Q3()


def Q4():
    
    
    flt = pd.read_csv('Student_Performance.csv', encoding = "ISO-8859-1")
    flt = flt[['Performance Index',  'Hours Studied',	'Previous Scores',	'Extracurricular Activities',	'Sleep Hours',	'Sample Question Papers Practiced']].copy()
    
    flt['Extracurricular Activities'] = flt['Extracurricular Activities'].astype('category')
    flt['Extracurricular Activities'] = flt['Extracurricular Activities'].cat.codes
    
    # Fit the model
    X = flt[[  'Hours Studied',	'Previous Scores',	'Extracurricular Activities',	'Sleep Hours',	'Sample Question Papers Practiced']]
    y = flt['Performance Index']
    model = LinearRegression()
    
    
    shuffle_split = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    
    
    
    resultsD = cross_validate(model, X, y, cv=shuffle_split, scoring='r2',
                         return_train_score=True)  # This gets training scores
    
    train_scoresD = np.mean(resultsD['train_score'])
    test_scoresD = np.mean(resultsD['test_score'])
    
   
    
    #print(flt['Extracurricular Activities'])
    
    
    print("R^2 Score Training:", train_scoresD)
    print("R^2 Score Test:", test_scoresD)
    


#Q4()


def Q5():
    
    
    flt = pd.read_csv('titanic.csv', encoding = "ISO-8859-1")
    flt['Sex'] = flt['Sex'].map({'male': 0, 'female': 1})
    flt = flt.dropna()
    # Fit the model
    X = flt [['Survived', 'Sex', 'Age', 'Pclass']]
    y = flt['Fare']
    model = LinearRegression()
    
    
    shuffle_split = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    
    
    
    resultsD = cross_validate(model, X, y, cv=shuffle_split, scoring='r2',
                         return_train_score=True)  # This gets training scores
    
    train_scoresD = np.mean(resultsD['train_score'])
    test_scoresD = np.mean(resultsD['test_score'])
    
   
    
    #print(flt['Extracurricular Activities'])
    
    
    print("R^2 Score Training:", train_scoresD)
    print("R^2 Score Test:", test_scoresD)
    
    for f in X.columns:
        r, p_value = pearsonr(X[f], y) 
        print(f)
        print(r, p_value)


#Q5()


def Q6():
    flt = pd.read_csv('movie_metadata.csv', encoding = "ISO-8859-1")
    df = flt[['budget', 'imdb_score', 'gross', 	'aspect_ratio'	, 'movie_facebook_likes', 'director_facebook_likes']].copy()
    
    df = df.dropna()
    
    r, p_value = pearsonr(df['movie_facebook_likes'], df['budget']) 
    print(r, p_value)
    
    r, p_value = pearsonr(df['movie_facebook_likes'], df['gross']) 
    print(r, p_value)
                
    r, p_value = pearsonr(df['movie_facebook_likes'], df['director_facebook_likes']) 
    print(r, p_value)

Q6()