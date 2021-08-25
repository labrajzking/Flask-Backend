

import numpy as np
import pandas as pd



data=pd.read_csv(r"C:\Users\AYA LAJILI\Desktop\projet1\data\healthcare-dataset-stroke-data.csv")

data['bmi'] = data['bmi'].fillna(data['bmi'].mean())

data = data.drop(columns ='id')
data['gender'] = data['gender'].replace('Other', list(data.gender.mode().values)[0])
data["bmi"] = pd.to_numeric(data["bmi"])
data["bmi"] = data["bmi"].apply(lambda x: 50 if x>50 else x)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

data['gender'] = le.fit_transform(data['gender'])
data['ever_married'] = le.fit_transform(data['ever_married'])
data['Residence_type'] = le.fit_transform(data['Residence_type'])
data['work_type'] = le.fit_transform(data['work_type'])
data['smoking_status'] = le.fit_transform(data['smoking_status'])

data = data.drop(['ever_married'], axis = 1)


#target_1 = data[data["stroke"]==1]
#target_0 = data[data["stroke"]==0]
#from sklearn.utils import resample
#diabet_downsampled = resample(target_1, replace = True,  n_samples = len(target_0), random_state = 123)
#downsampled = pd.concat([diabet_downsampled, target_0])



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
X=data.drop('stroke',axis=1).values
y=data['stroke'].values.reshape(-1,1)
# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
# Create the classifier: logreg
logreg = LogisticRegression(solver='lbfgs',class_weight='balanced', max_iter=10000)
# Fit the classifier to the training data
logreg.fit(X_train,y_train)
# Saving model to current directory
# Pickle serializes objects so they can be saved to a file, and loaded in a program again later on.
pickle.dump(logreg, open('stroke.pkl','wb'))




