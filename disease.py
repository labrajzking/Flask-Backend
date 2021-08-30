import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(r"C:\Users\oussema\Downloads\Flask-Back-End\disease.py")

l=["male","BPMeds","prevalentHyp","diabetes","sysBP","cigsPerDay","age","TenYearCHD"]
#"BPMeds","prevalentHyp","diabetes","sysBP","cigsPerDay","age"
df1=data[l]
df1=df1.dropna()


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

X=df1.drop('TenYearCHD',axis=1).values
y=df1['TenYearCHD'].values.reshape(-1,1)
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
# Create the classifier: logreg
logreg = LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=10000)
# Fit the classifier to the training data
logreg.fit(X_train,y_train)
# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(logreg, open('disease.pkl','wb'))




