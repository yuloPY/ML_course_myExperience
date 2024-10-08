# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1].values
y = dataset.iloc[:, -1].values

#X variable to the 2D array
X = X.reshape(-1, 1)

#Training the linear regression model on the whole dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Training the polynomial regression model on the whole dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)

#Visualising the Linear Regression Results
plt.scatter(X,y,color="red")
plt.plot(X,lin_reg.predict(X),color="blue")
plt.title("Truth or Bluf(Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Visualising the Polynomial Regression Results
plt.scatter(X,y,color="red")
plt.plot(X,lin_reg2.predict(X_poly),color="blue")
plt.title("Truth or Bluf(Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
