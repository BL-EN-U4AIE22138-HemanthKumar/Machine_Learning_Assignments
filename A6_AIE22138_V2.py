import matplotlib.pyplot as mp
#importing panda to access the dataset/plotting
import pandas as pd
#importing numpy to access the array functions
import numpy as np
import math
from sklearn.neural_network import MLPClassifier

w1= 0.2
w2= -0.75
b=10
alpha=0.05
A=[0,0,1,1]
B=[0,1,0,1]
Z=[0,0,0,1]

def stepactivation(Yin):
    if(Yin >= 0):
        return 1
    else:
        return 0

def bipolarstep(Yin):
    if(Yin > 0):
        return 1
    elif(Yin == 0):
        return 0
    else:
        return -1

def RelU(Yin):
    if(Yin > 0):
        return Yin
    else:
        return 0
    
def sigmoid(Yin):
    if(Yin > 0):
        return 1
    elif(Yin == 0):
        return 0
    else:
        return -1

epochs=[]
errors=[]
epoch=1
for _ in range(0,10):
    err=0
    sum=0
    print("------------------------------------------------------------------------")
    print("epoch:",epoch)
    print("------------------------------------------------------------------------")
    epochs.append(epoch)
    epoch += 1
    for i in range(0,len(Z)):
        x1=A[i]
        x2=B[i]
        t=Z[i]
        yin= b + x1*w1 +x2*w2
        y=stepactivation(yin)

        err=t-y
        sum += err

        print("y value:",y)
        
        if(y != t):
            print(y,"!=",t)
            print("b old:",b,"w1 old:",w1,"w2 old:",w2)
            print("update weights:")
            b=alpha*t
            w1=w1+alpha*t*x1
            w2=w2+alpha*t*x2
            print("b new:",b,"w1 new:",w1,"w2 new:",w2)
        else:
            print(y,"=",t)
            print("dont update weight")
            print("b new:",b,"w1 new:",w1,"w2 new:",w2)
    errors.append(sum)

mp.title('Step activation functions')
mp.xlabel('epochs')
mp.ylabel('errors')
mp.plot(epochs,errors)
mp.show()
print("weights",b,w1,w2)


#Bi polar step funtion
epochs=[]
errors=[]

epoch=1
for _ in range(0,10):
    err=0
    sum=0
    epochs.append(epoch)
    epoch += 1
    for i in range(0,len(Z)):
        x1=A[i]
        x2=B[i]
        t=Z[i]
        yin= b + x1*w1 +x2*w2
        y=bipolarstep(yin)

        err=t-y
        sum += err
        if(y != t):
            b=alpha*t
            w1=w1+alpha*t*x1
            w2=w2+alpha*t*x2
        else:
            continue
    errors.append(sum)
mp.title('bipolar step function')
mp.xlabel('epochs')
mp.ylabel('errors')
mp.plot(epochs,errors)
mp.show()


#ReLU activation function
epochs=[]
errors=[]

epoch=1
for _ in range(0,10):
    err=0
    sum=0
    epochs.append(epoch)
    epoch += 1
    for i in range(0,len(Z)):
        x1=A[i]
        x2=B[i]
        t=Z[i]
        yin= b + x1*w1 +x2*w2
        y=RelU(yin)

        err=t-y
        sum += err
        
        if(y != t):
            b=alpha*t
            w1=w1+alpha*t*x1
            w2=w2+alpha*t*x2
        else:
            continue
    errors.append(sum)

mp.title('ReLu')
mp.xlabel('epochs')
mp.ylabel('errors')
mp.plot(epochs,errors)
mp.show()


#Sigmoid
epochs=[]
errors=[]
epoch=1
for _ in range(0,10):
    err=0
    sum=0
    epochs.append(epoch)
    epoch += 1
    for i in range(0,len(Z)):
        x1=A[i]
        x2=B[i]
        t=Z[i]
        yin= b + x1*w1 +x2*w2
        y=sigmoid(yin)

        err=t-y
        sum += err       
        if(y != t):
            b=alpha*t
            w1=w1+alpha*t*x1
            w2=w2+alpha*t*x2
        else:
            continue
    errors.append(sum)

mp.title('sigmoid function')
mp.xlabel('epochs')
mp.ylabel('errors')
mp.plot(epochs,errors)
mp.show()


#A3
w1= 0.2
w2= -0.75
b=10
alpha=0
A=[0,0,1,1]
B=[0,1,0,1]
Z=[0,0,0,1]

epochs=[]
errors=[]
epoch=1
alphas=[]
for k in range(0,10):
    alpha=alpha+0.1
    for _ in range(0,10):
        epoch += 1
        for i in range(0,len(Z)):
            x1=A[i]
            x2=B[i]
            t=Z[i]
            yin= b + x1*w1 +x2*w2
            y=stepactivation(yin)
            if(y != t):
                b=alpha*t
                w1=w1+alpha*t*x1
                w2=w2+alpha*t*x2
            else:
                continue
    alphas.append(alpha)
    epochs.append(epoch)
mp.title('iterations - learning rates PLOT')
mp.xlabel('iterations')
mp.ylabel('alpha')
mp.plot(epochs,alphas)
mp.show()



#A11
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
removeNullCategorical('RainToday')
y=df['RainToday'].to_numpy()
y=y[:20]
X=df[['MinTemp','MaxTemp']].to_numpy()
X=X[:20]
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlp.fit(X, y)
