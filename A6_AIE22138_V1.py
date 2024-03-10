import numpy as np

w1=0.2
w2=-0.75
b=-1
alpha=0.05
A=[0,0,1,1]
B=[0,1,0,1]
Z=[0,0,0,1]
sum=0
y=0
for _ in range(0,200):
    for i in range(0,len(Z)):
        x1=A[i]
        x2=B[i]
        
        sum= x1*w1 + x2*w2 + b
        if(sum+b >= 0):
            y=1
        elif(sum+b == 0):
            y=0.5
        elif(sum+b < 0):
            y=0

            
        if (y != Z[i]):
            b = b + alpha*Z[i]
            w1= w1 + alpha * x1
            w2 = w2 + alpha * x2
        else:
            w1=w1
            w2=w2        
print("updated weights:",w1,w2,b)
    
    
    
    
    
    
