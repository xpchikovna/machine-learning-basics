from sklearn import datasets
from sklearn import tree

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import math 

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import warnings


import seaborn as sns

# Below line would ignore the warning messages in console.
warnings.filterwarnings("ignore")

def Q1():
    df = pd.read_csv("titanic.csv",encoding = "ISO-8859-1")
    
    df = df.dropna()
    group1 = df[df['Sex'] == 'female']
    group2 = df[df['Sex'] == 'male']
    
    
    
    
    X1 = group1[['Age',  'Pclass']]
    y1 = group1[['Survived']]
    tree_clf = tree.DecisionTreeClassifier()
    
    tree_clf.fit(X1, y1)   
    print(accuracy_score(tree_clf.predict(X1), y1))
    
    
    X1 = group2[['Age',  'Pclass']]
    y1 = group2[['Survived']]
    tree_clf = tree.DecisionTreeClassifier()
    
    tree_clf.fit(X1, y1)   
    print(accuracy_score(tree_clf.predict(X1), y1))
    

    
    
#Q1()

def Q2():
    df = pd.read_csv("titanic.csv",encoding = "ISO-8859-1")
    dict = {'female': 1, 'male':2}
    df['Sex'] = df['Sex'] .map(dict)
    df['Sex'] = df['Sex'].astype(int)
    
    
    
    flt = df [['Survived', 'Pclass', 'Age', 'Sex']]
  
    """
    filling the empty cells with the average of the exisiting values in each column.
    """
    flt = flt.dropna()
    
    X = (flt[['Pclass', 'Age']])
    
    y = flt[['Survived']]
    tree_clf = tree.DecisionTreeClassifier( )
    
    tree_clf.fit(X, y)
    print(tree_clf.score(X, y))
    
    
    X = (flt[['Pclass', 'Age', 'Sex']])
    
    y = flt[['Survived']]
    tree_clf = tree.DecisionTreeClassifier()
    tree_clf.fit(X, y)
    print(tree_clf.score(X, y))
    

   
#Q2()


def Q3():
    df = pd.read_csv("attacks.csv",encoding = "ISO-8859-1")
    
    flt = df [['Activity', 'Age', 'Fatal']]
    
    flt['Age'] = flt['Age'].apply(pd.to_numeric, errors='coerce')
    
    flt = flt [['Activity', 'Age', 'Fatal']].dropna()
   
  
    
    # Convert to categorical and use category codes
    flt['Activity'] = flt['Activity'].astype('category')
    flt['Activity'] = flt['Activity'].cat.codes + 1  # +1 to start from 1 instead of 0
        
   
    
    
    dict1 = {"Y":1, "N":2}
    flt['Fatal'] = flt['Fatal'] .map(dict1)
    flt['Fatal'] = flt['Fatal'].astype(str)
    flt['Fatal'] = flt['Fatal'].apply(pd.to_numeric, errors='coerce')
    flt = flt.dropna()

    X = (flt[['Activity', 'Age']])
    
    y = flt[['Fatal']]
    # print(np.shape(X), np.shape(y))
    
    tree_clf = tree.DecisionTreeClassifier( )
    tree_clf.fit(X, y)
    print(tree_clf.score(X, y))
    
#Q3()
    
def Q4():
    df = pd.read_csv("attacks.csv",encoding = "ISO-8859-1")
    
    countries = df.groupby('Country')
    #the below line has an array selector that returns the list of country with more than 100 enteries.
    countries100 = countries.size()[countries.size()>200]
    #print(countries100)
    
    for country in countries100.index:
        
        flt = df[df['Country']==country]
        
        
        # Convert to categorical and use category codes
        flt['Activity'] = flt['Activity'].astype('category')
        flt['Activity'] = flt['Activity'].cat.codes + 1  # +1 to start from 1 instead of 0
        
        dict1 = {'F': 1, 'M':2}
        flt['Sex '] = flt['Sex '] .map(dict1)
        
        
        allFatals = np.unique(flt['Fatal'].astype(str))
        
        dict1 = {"Y":1, "N":2}
        flt['Fatal'] = flt['Fatal'] .map(dict1)
        
        flt['Fatal'] = flt['Fatal'].apply(pd.to_numeric, errors='coerce')
        flt['Age'] = flt['Age'].apply(pd.to_numeric, errors='coerce')
        flt = flt.dropna()
        
        flt['Case Number'] = flt['Case Number'].str.replace('.[a-zA-Z]', '', regex=True)
        flt['Case Number'] = flt['Case Number'].str.replace(' ', '')
        flt['Case Number'] = flt['Case Number'].str.replace('\"', '')
        flt['Case Number'] = flt['Case Number'].apply(pd.to_datetime, errors='coerce')
        #The below line helps you to convert a series into a sepecific format e.g., 2024.08.14
        
        flt['Case Number'] = pd.to_datetime(flt['Case Number'], format="%Y.%m.%d")
        # 2022/10/01
        # 2022/01/10
        # 10/12/2022
        if len(flt) > 200:
            X = (flt[['Activity', 'Age', 'Sex ']])
        
            y = flt[['Fatal']]
            tree_clf = tree.DecisionTreeClassifier()
            tree_clf.fit(X, y)
            # Below is to extract only the year
            print(flt['Case Number'].astype(str).str.split("-").str[0].value_counts().index[0])
            print(country, tree_clf.score(X, y), len(flt))
        
        
    
#Q4()
def Q5():
    df = pd.read_csv("titanic.csv",encoding = "ISO-8859-1")
   
    
    flt = df [['Survived', 'Pclass', 'Age', 'Fare']]
  
    """
    filling the empty cells with the average of the exisiting values in each column.
    """
    flt['Fare'] = flt['Fare'].apply(pd.to_numeric, errors='coerce')
    flt['Fare'] = flt['Fare'].fillna(np.mean(flt['Fare']))
    
    flt['Age'] = flt['Age'].apply(pd.to_numeric, errors='coerce')
    flt['Age'] = flt['Age'].fillna(np.mean(flt['Age']))
    
    flt = flt.dropna()
    
        
    
    X = (flt[['Pclass', 'Age', 'Fare']])
    
    
    y = flt[['Survived']]
    tree_clf = tree.DecisionTreeClassifier()
    tree_clf.fit(X, y)    
    print(tree_clf.score(X, y))
    
    importance = tree_clf.feature_importances_
    
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
#Q5()
    

def Q6():
    df = pd.read_csv("titanic.csv",encoding = "ISO-8859-1")
    # Note that you dont have to always pass the entire df to the seaborn function, you can call the features separately like below:
    # Note that below helps you to manage the border of the figure...
    plt.figure(figsize=(20, 12))    
    sns.barplot( x=df["Pclass"].astype(float), y=df["Fare"].astype(float), estimator=sum)
    plt.show()

Q6()