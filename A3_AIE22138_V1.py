import pandas as pd
import numpy as np
import statistics as st
import seaborn as sns
import matplotlib.pyplot as plt


#read the excel contents and convert to dataframe and then to array
df = pd.read_excel("C:/Users/vijay/Downloads/Lab Session1 Data.xlsx", sheet_name='IRCTC Stock Price')
df=pd.DataFrame(df)
Array=df.to_numpy()

#fetch according to the column, price column = D
D=Array[:,3:4]
D=D.flatten()

#mean and variance
print("Mean:",st.mean(D))
print("Variance:",st.variance(D))

#fetch according to the column, day column = C
C=Array[:,2:3]
C=C.flatten()
sample=[]
for i in range(0,len(D)):

    #append the price according to the day
    if(C[i]=="Wed"):
        sample.append(D[i])
print("sample mean of price data for all Wednesdays:",st.mean(sample))

#fetch according to the column, month column = B
B=Array[:,1:2]
B=B.flatten()
sample=[]
for i in range(0,len(D)):
    if(B[i]=="Apr"):

        #append the price according to the month
        sample.append(D[i])
print("sample mean price data for the month of April:",st.mean(sample))

chg=Array[:,8:9]
chg=chg.flatten()
# Marking all the values which have been marked below zero.
l2 = list(map(lambda i: i < 0, chg))

l2_false = [value for value in l2 if value is False]
probability = (len(l2_false) / len(l2))
l0 = 1 - probability
print("Probability: ", probability)

wednesday_chg=[]
for i in range(0,len(D)):
    if(C[i]=="Wed"):
        wednesday_chg.append(chg[i])

# caluculating the profits done on wednesday.
l3 = list(map(lambda i: i > 0, wednesday_chg))
l3_True = [value for value in l3 if value is True]
probability_wed = (len(l3_True) / len(l3))

# conditional probability  of a profit given that it was made on Wednesday
conditional_probability = probability_wed * l0
print("profits on wednesday:", probability_wed)
print("conditional probability:", conditional_probability)

# Create a scatter plot of 'Chg%' data against the day of the week
sns.scatterplot(x='Day', y='Chg%', data=df)

# Display the plot
plt.show()
