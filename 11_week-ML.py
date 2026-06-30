from sklearn import datasets
from sklearn import tree

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import math 
from sklearn.model_selection import train_test_split
from pandas import read_csv
import numpy as np 
from sklearn.model_selection import cross_val_score, KFold

from sklearn.model_selection import ShuffleSplit

from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from io import StringIO
from sklearn import linear_model
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans

from sklearn import preprocessing  


from sklearn import datasets
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")


def Q1():
    df = pd.read_csv("attacks.csv",encoding = "ISO-8859-1")
    
    flt = df [['Activity','Age', 'Fatal']].copy()
    
    flt['Age'] = flt['Age'].apply(pd.to_numeric, errors='coerce')
    flt['Age'] = flt['Age'].fillna(flt['Age'].mean())
    
    flt.loc[flt['Activity'].str.lower().str.contains('surf', na=False), 'Activity'] = 0
    flt.loc[flt['Activity'].str.lower().str.contains('swimm', na=False), 'Activity'] = 1
    flt.loc[flt['Activity'].str.lower().str.contains('scuba', na=False), 'Activity'] = 2
    
    flt['Activity'] = flt['Activity'].apply(pd.to_numeric, errors='coerce')
    
    
    flt = flt.dropna()
    
    
    flt['Fatal']= flt['Fatal'].str.replace(' ','')
    flt['Fatal']= flt['Fatal'].str.replace('UNKNOWN','N')
    flt['Fatal']= flt['Fatal'].str.replace('F','N')
    flt['Fatal']= flt['Fatal'].str.replace('#VALUE!','N')
    flt['Fatal']= flt['Fatal'].str.replace('n','N')
    flt['Fatal']= flt['Fatal'].str.replace('2017','N')
    dic = {'N':1, 'Y':2}
    flt['Fatal'] = flt['Fatal'].map(dic)
    
    print(np.unique(flt['Fatal']))
    flt = flt.dropna()
    X = (flt[['Activity', 'Age']])
    
    y = flt[['Fatal']]
    
    
    models = []
   
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('DTC', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    models.append(('RFS',RandomForestClassifier()))
    # evaluate each model in turn
    names = []
    results = {}
    
    
    AllModels = {}
    
    
    for name, model in models:
        
        kfold = StratifiedShuffleSplit(n_splits=5,test_size=0.1)
        
        # one of the below
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        shuffle_split = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
        
        
        results = cross_validate(model, X, y, cv=shuffle_split, scoring='accuracy',
                            return_train_score=True)  # This gets training scores
        
        
        AllModels[name] = results
        train_scores = np.mean(results['train_score'])
        test_scores = np.mean(results['test_score'])
        #print(name)
        
        #print(train_scores)
        #print(test_scores)
    return AllModels
            
        
    

#Q1()

def Q2():
    result = Q1()
    sResults = pd.Series(result['KNN'])
    print(np.mean(sResults['test_score']))
    print(np.mean(sResults['train_score']))
    fig, (vis1, vis2, vis3) = plt.subplots(1, 3)
    vis1.bar(range(1, len(sResults['test_score'])+1), sResults['test_score'])
    vis2.bar(range(1, len(sResults['train_score'])+1), sResults['train_score'])
    vis3.bar(range(1, len(sResults['train_score'])+1), sResults['train_score']-sResults['test_score'])
    
    # Note the below how the scale of the axis are set.
    vis1.set_ylim(0.7,0.8)
    vis2.set_ylim(0.7,0.8)
    vis3.set_ylim(0.0,0.08)
    labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for ax in [vis1, vis2, vis3]:
        ax.set_xticks(range(1, len(labels)+1))  # Set positions 0-9
        ax.set_xticklabels(labels)
       
    # Add titles and labels
    vis1.set_title('Training Accuracy ')
    vis1.set_ylabel('Accuracy')
    vis1.set_xlabel('All folds')
    
    vis2.set_title('Test Accuracy') 
    vis2.set_ylabel('Accuracy')
    vis2.set_xlabel('All folds')
    
    vis3.set_title('Difference Train vs Test') 
    vis3.set_ylabel('Accuracy')
    vis3.set_xlabel('All folds')
    
    # Note the line below... it helps better clearaity 
    plt.tight_layout()
    plt.show()
#
#Q2()




def Q3_1(depth):
    df = pd.read_csv("titanic.csv",encoding = "ISO-8859-1")
    
    dict = {'female': 1, 'male':2}
    df['Sex'] = df['Sex'] .map(dict)
    df['Sex'] = df['Sex'].astype(int)
    X = df [['Sex',  'Age']]
    y = df['Survived']
    model = tree.DecisionTreeClassifier(max_depth=depth)
    shuffle_split = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    
    
    results = cross_validate(model, X, y, cv=shuffle_split, scoring='accuracy',
                        return_train_score=True)  # This gets training scores
   
    print(np.mean(results['test_score']))
    print(np.mean(results['train_score']))
    
def Q3_2(depth):
    df = pd.read_csv("titanic.csv",encoding = "ISO-8859-1")
    
    dict = {'female': 1, 'male':2}
    df['Sex'] = df['Sex'] .map(dict)
    df['Sex'] = df['Sex'].astype(int)
    
    buckets = pd.cut(df['Age'], 
                     bins=[-np.inf, 25, 60, np.inf], 
                     labels=list(range(1, 4)),
                     duplicates='drop')
    
    #print(buckets)
    df['Age'] = buckets
    
    X = df [['Sex',  'Age']]
    y = df['Survived']
    
    model = tree.DecisionTreeClassifier(max_depth=depth)
    shuffle_split = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    
    
    results = cross_validate(model, X, y, cv=shuffle_split, scoring='accuracy',
                        return_train_score=True)  # This gets training scores
   
    print(np.mean(results['test_score']))
    print(np.mean(results['train_score']))

#Q3_1(5)

#Q3_2(5)
    

def Q4():
    
    
    def category2number(col):
        col = col.astype('category')
        newCol = col.cat.codes

        return newCol
    
    df = pd.read_csv("Student_Performance.csv",encoding = "ISO-8859-1")
    
    newDF = pd.DataFrame()
    
    for feature in df.columns:
        if df[feature].dtype != np.float64 and df[feature].dtype != np.int64:
            newDF[feature] = category2number(df[feature])
        else:
            newDF[feature] = df[feature]
    
        
    
    buckets = pd.cut(newDF['Performance Index'], 
                     bins=[-np.inf, 33, 50, 75, np.inf], 
                     labels=list(range(1, 5)),
                     duplicates='drop')
    
    newDF['Performance Index'] = buckets
    y = newDF['Performance Index']
   
    
    X = newDF[['Hours Studied',	'Previous Scores',	'Extracurricular Activities',	'Sleep Hours',	'Sample Question Papers Practiced']]
    
    model = tree.DecisionTreeClassifier()
    shuffle_split = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    
    
    results = cross_validate(model, X, y, cv=shuffle_split, scoring='accuracy',
                        return_train_score=True)  # This gets training scores
   
    print(np.mean(results['test_score']))
    print(np.mean(results['train_score']))
    
#Q4()
    
    
    
    
    
    
    

