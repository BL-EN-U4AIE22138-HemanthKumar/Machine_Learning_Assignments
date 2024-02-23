Python 3.12.1 (tags/v3.12.1:2305ca5, Dec  7 2023, 22:03:25) [MSC v.1937 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> 
= RESTART: C:\Users\vijay\AppData\Local\Programs\Python\Python312\A4_AIE22138_V1.py

Warning (from warnings module):
  File "C:\Users\vijay\AppData\Local\Programs\Python\Python312\A4_AIE22138_V1.py", line 2
    import pandas as pd
DeprecationWarning: 
Pyarrow will become a required dependency of pandas in the next major release of pandas (pandas 3.0),
(to allow more performant data types, such as the Arrow string type, and better interoperability with other libraries)
but was not found to be installed on your system.
If this would cause problems for you,
please provide us feedback at https://github.com/pandas-dev/pandas/issues/54466
        
mean of the class MaxTemp: 23.226784191272355
mean of the class MinTemp: 12.186399728729265
mean of the standard deviation of MaxTemp: 7.117618141018133
mean of the standard deviation of MinTemp: 6.403282674671353
distance between mean vectors: 11.04038446254309
mean: 39.98429165757619 variance: 184.65550624412074
Minksowki distance: 6.390618123468184
0.7464731282613893
['No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'Yes' 'No'
 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No'
 'No' 'No' 'No' 'No' 'Yes' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No'
 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'Yes' 'No' 'No'
 'No' 'No' 'No' 'No' 'No' 'No' 'Yes' 'Yes' 'No' 'No' 'No' 'No' 'No' 'Yes'
 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No'
 'No' 'No' 'No' 'No' 'Yes' 'Yes' 'No' 'No' 'No' 'No' 'No' 'No' 'No' 'No'
 'No' 'No']
[[48362  6745]
 [11280  4710]]
              precision    recall  f1-score   support

          No       0.81      0.88      0.84     55107
         Yes       0.41      0.29      0.34     15990

    accuracy                           0.75     71097
   macro avg       0.61      0.59      0.59     71097
weighted avg       0.72      0.75      0.73     71097

K=1 accuracy: 0.7030817052758906
K=3 accuracy: 0.7490330112381676
[0.7066542892105152, 0.7663473845591234, 0.7482312896465393, 0.7736894665035093, 0.7650111819064096, 0.7773183116024586, 0.7702294048975343, 0.7798078681238308, 0.7766713082127235, 0.783000689199263, 0.7796250193397752]
