import pandas as pd
import numpy as np

#convert the execel contents to dataframe
df = pd.read_excel("C:/Users/vijay/Downloads/Lab Session1 Data.xlsx", sheet_name='Purchase data')
df=pd.DataFrame(df)
print(df)

#convert datafram to array
Array=df.to_numpy()
X=Array[:,:1]
A=Array[:,1:4]
C=Array[:,4:5]
print(A)
print(X)
print(C)

#typecast the elements to float
A=np.float64(A)

#return the rank
print("rank of matrix A:",np.linalg.matrix_rank(A))

#return the pseudoinverse
print("pseudo inverse:",np.linalg.pinv(A))

#categorise the customers to RICH and POOR if purchase exceeds 200
C=C.flatten()
category=[]
for i in range(len(C)):
    if(C[i]>=200):
        category.append("RICH")
    else:
        category.append("POOR")

#inserting to the dataframe
df.insert(1, "category", category)
print(df)

