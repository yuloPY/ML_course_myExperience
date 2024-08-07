# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values  # Use 1:2 to keep X as a 2D array
y = dataset.iloc[:, -1].values

# Feature scaling
from sklearn.preprocessing import StandardScaler
# We create two separate StandardScaler instances for sc_X and sc_y because
# X and y should have their own mean and standard deviation values.
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = y.reshape(-1, 1)  # First reshape y to 2D
y = sc_y.fit_transform(y).ravel()  
# Then apply fit_transform and ravel to convert y to 1D

# Training the SVR model
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)
print("MODEL IS READY.")

# Predicting new results
predicted_value = sc_y.inverse_transform(regressor.predict(
    sc_X.transform([[6.5]])).reshape(-1,1))

print("6.5 =>", predicted_value)

# Visualizing the SVR results
reserved_X = sc_X.inverse_transform(X)
reserved_y = sc_y.inverse_transform(y.reshape(-1, 1))
# Get model predictions
predictions = regressor.predict(X)

# Reshape the predictions
reshaped_predictions = predictions.reshape(-1, 1)

# Inverse transform the predictions to original scale
original_scale_predictions = sc_y.inverse_transform(reshaped_predictions)

plt.scatter(reserved_X, reserved_y, color="red")
plt.plot(reserved_X, original_scale_predictions, color="blue")
plt.title("Truth or Bluff (Support Vector Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

#Visualising the SVR results(for higher resolution and smoother curve)

X_grid = np.arange(min(reserved_X), max(reserved_X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(reserved_X, reserved_y, color = 'red')
plt.plot(X_grid,sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (Support Vector Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()