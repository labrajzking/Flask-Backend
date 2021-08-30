# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
data=pd.read_csv(r"C:\Users\oussema\Downloads\Flask-Back-End\data\diabetes_data.csv")


target_1 = data[data["Outcome"]==1]

target_0 = data[data["Outcome"]==0]

from sklearn.utils import resample

# Downsample majority and combine with minority
diabet_downsampled = resample(target_0, replace = False,  n_samples = len(target_1), random_state = 123)
downsampled = pd.concat([diabet_downsampled, target_1])


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
X=downsampled.drop('Outcome',axis=1).values
y=downsampled['Outcome'].values.reshape(-1,1)

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
# Create the classifier: logreg
logreg = LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=10000)
# Fit the classifier to the training data
logreg.fit(X_train,y_train)


# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(logreg, open('diabets.pkl','wb'))


