import numpy as np
from collections import Counter
from math import log2
import math
import pandas as pd

from sklearn.preprocessing import LabelEncoder

#calculate the entropy of the target variable
def entropy(y):
    n = len(y)
    counts = np.bincount(y)
    probs = counts[np.nonzero(counts)] / n
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

#calculate the gain of the feature
def information_gain(X, y, feature_idx):
    total_entropy = entropy(y)

    values, counts = np.unique(X[:, feature_idx], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / len(y)) * entropy(y[X[:, feature_idx] == values[i]]) for i in range(len(values))])

    gain = total_entropy - weighted_entropy
    return gain

#finds the root node with the highest information gain
def find_root_node(X, y):
    num_features = X.shape[1]
    gains = [information_gain(X, y, i) for i in range(num_features)]
    print(gains)
    root_feature_idx = np.argmax(gains)
    return root_feature_idx


df = pd.read_csv("C:/Users/vijay/Downloads/weatherAUS.csv/weatherAUS.csv")
df=pd.DataFrame(df)

#merging 2 features in one array as base set
def removeNull(Class):
    iterate=df[Class]
    y=[]
    mean=df[Class].mean(axis=0)
    for i in iterate:
        if(math.isnan(i)):
            y.append(mean)
        else:
            y.append(i)
    y=pd.DataFrame(y)
    df[Class]=y
removeNull('MinTemp')
removeNull('MaxTemp')
def labelencode(Class):
    removeNullCategorical(Class)
    temp=df[Class]
    temp=temp.to_numpy()
    temp=temp.flatten()
    LE=LabelEncoder()
    LE.fit(temp)
    array=LE.transform(temp)
    array=pd.DataFrame(array)
    df[Class]=array
    
def isNaN(string):
    return string != string
def removeNullCategorical(Class):
    iterate=df[Class]
    y=[]
    mostfreq=df[Class].value_counts().idxmax()
    for i in iterate:
        if(isNaN(i)):
            y.append(mostfreq)
        else:
            y.append(i)
    y=pd.DataFrame(y)
    df[Class]=y
    return y

removeNullCategorical('RainToday')
labelencode('RainToday')
y=df['RainToday'].to_numpy()
X=df[['MinTemp','MaxTemp']].to_numpy()

root_feature_idx = find_root_node(X,y)
print("Root node feature index:", root_feature_idx)
