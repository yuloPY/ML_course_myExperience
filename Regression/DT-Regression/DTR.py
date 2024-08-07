# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

#Training the Decision Tree Regression Model on the whole dataset
from sklearn.tree import DecisionTreeRegressor
DTR =DecisionTreeRegressor(random_state=0)
DTR.fit(X,y)

#Predicting specific value
predicted_value = DTR.predict([[6.5]])

print(predicted_value)

#Visualising the Decision Tree Regression Model
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((-1, 1))
plt.scatter(X,y, color = 'red')
plt.plot(X_grid,DTR.predict(X_grid).reshape((-1,1)), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regresion)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()