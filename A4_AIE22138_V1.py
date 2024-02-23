#importing panda to access the dataset/plotting
import pandas as pd
#importing numpy to access the array functions
import numpy as np
#importing matplotlib to plot the histogram
import matplotlib.pyplot as mpl
#importing sklearn to remove Nan values, split the train and test set, confusion matrix, f1 score, recall
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

#importing scipy to calculate minkowski distance
from scipy.spatial import distance

#A1
#convert the execel contents to dataframe
df = pd.read_csv("C:/Users/vijay/Downloads/weatherAUS.csv/weatherAUS.csv")
df=pd.DataFrame(df)

#we take classes MinTemp and MaxTemp
def mean(Class):
    mean_class=df[Class].mean(axis=0)
    return mean_class

print("mean of the class MaxTemp:",mean('MaxTemp'))
print("mean of the class MinTemp:",mean('MinTemp'))

#finding the standard deviation
def standard_deviation(Class):
    std_deviate=df[Class].std(axis=0)
    return std_deviate  
print("mean of the standard deviation of MaxTemp:",standard_deviation('MaxTemp'))
print("mean of the standard deviation of MinTemp:",standard_deviation('MinTemp'))

#finding the distance between mean vectors
def Distance():
    df1=pd.DataFrame(df)

    #converting datafram to array
    Array=df1.to_numpy()
    distance=np.linalg.norm(mean('MaxTemp')-mean('MinTemp'))
    return distance
print("distance between mean vectors:",Distance())

#A2

#to find variance
def variance(feature):
    variance=df[feature].var(axis=0)
    return variance    

#to plot the histogram
def histogram(feature):
    plot=df[feature].hist(bins=2)
    mpl.xlabel('windspeed')
    mpl.ylabel('no. of data')
    mpl.title("histogram")
    mpl.show()

#printing mean and variance
print("mean:",mean('WindGustSpeed'),"variance:",variance('WindGustSpeed'))
histogram('WindGustSpeed')


#A4
#merging 2 features in one array as base set
X=pd.concat([df['MinTemp'],df['MaxTemp']],axis=1)

#features
#removing the Nan values with mean of the columns
imp=SimpleImputer(strategy="mean")
X=imp.fit_transform(X)

#prediction labels
#removing the missing values with most frequent label
imp2=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
y=[df['RainTomorrow']]
y=imp2.fit_transform(y)
#converting into 1D array
y=y.flatten()

#A3
#printing the minkowski distance
print("Minksowki distance:",distance.minkowski(X[0],X[1]))

#A4 conti....
#splitting the data into train and set
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)

#A5
#applying KNN of K=3
neigh=KNeighborsClassifier(n_neighbors=3)

#training the model
neigh.fit(X_train,y_train)

#A6
#returning the accuracy of the model with respect to testing model
accuracy=neigh.score(X_test,y_test)
print(accuracy)

#A7
#returning the prediction vector predicted by the model
test_vector=neigh.predict(X_test)
print(test_vector[0:100]) #sample tes_vector

#A9
#returning the confusion matrix and the report
print(confusion_matrix(y_test, test_vector))
print(classification_report(y_test,test_vector))


#A8
#comparing the accuracy of KNN with K=1,K=3
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)
neigh=KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train,y_train)
accuracy=neigh.score(X_test,y_test)
print("K=1 accuracy:",accuracy)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)
neigh=KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train,y_train)
accuracy=neigh.score(X_test,y_test)
print("K=3 accuracy:",accuracy)

#initialzing the accuracy list
accuracy=[]
#finding the accuracy from K=1 to K=11
for i in range(1,12):
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)
    neigh=KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,y_train)
    accuracy.append(neigh.score(X_test,y_test))
print(accuracy)
s = pd.Series(accuracy)

#plotting the accuracy
s.plot.line()
mpl.xlabel('')
mpl.ylabel('accuracy')
mpl.title("Accuracy plot")
mpl.show()

