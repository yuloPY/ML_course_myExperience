# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Encoding the categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Training the multi regression model on the training set
from sklearn.linear_model import LinearRegression
multiRegresor = LinearRegression()
multiRegresor.fit(X_train,y_train)


#Predicting the test set results
y_pred = multiRegresor.predict(X_train)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

#Making an single prediction
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

#Getting the final linear regression equation with the values of the coefficients
print(regressor.coef_)
print(regressor.intercept_)

