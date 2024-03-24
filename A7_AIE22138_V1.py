#importing panda to access the dataset/plotting
import pandas as pd
#importing numpy to access the array functions
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn import svm


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
removeNullCategorical('RainToday')
labelencode('RainToday')
y=df['RainToday'].to_numpy()
y=y[:600]
X=df[['MinTemp','MaxTemp']].to_numpy()
X=X[:600]

#Perceptron
per = Perceptron(tol=1e-3, random_state=0)
per.fit(X,y)
per.score(X, y)
#initialzing n_neighbors
parameters={'alpha':uniform(loc=0, scale=4)}
#call randomized Search CV to tune the hyperparameter
clf = RandomizedSearchCV(per, param_distributions=parameters,random_state=0)
search=clf.fit(X,y)
print("best K value:",search.best_params_)

#MLP classifier
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlp.fit(X, y)
parameters={'alpha':uniform(loc=0, scale=4)}
#call randomized Search CV to tune the hyperparameter
clf = RandomizedSearchCV(mlp, param_distributions=parameters,random_state=0)
search=clf.fit(X,y)
print("best K value:",search.best_params_)

#SVM
svmclass=svm.SVC()
svmclass.fit(X,y)
clf = RandomizedSearchCV(svmclass, param_distributions=parameters,random_state=0)
search=clf.fit(X,y)
print("best K value:",search.best_params_)









