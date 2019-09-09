import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
# Register datetime converter for a matplotlib plotting method
pd.plotting.register_matplotlib_converters()


# Path to the CSV dataset file:
path_to_csv_file = './AAPL.csv'
# Load dataset from CSV
dataset = pd.read_csv(path_to_csv_file)

# Convert dataset['Date'] values to type "datetime".
dataset['Date'] = pd.to_datetime(dataset['Date'])
# Convert dataset['Date'] values to day counts from the date 01/01/01
dataset['Date'] = dataset['Date'].map(datetime.toordinal)


# Split the dataset into the Training set and Test set
# 99% of the dataset we will use as Training Set, and 1% to test models.
split_index = int(len(dataset)*0.99)

# Training Set
X_train = dataset.iloc[:split_index, :1].values
y_train = dataset.iloc[:split_index, 5].values

# Test Set
X_test = dataset.iloc[split_index:, :1].values
y_test = dataset.iloc[split_index:, 5].values



# Plot the Training Set
plt.scatter(X_train,y_train)
plt.title('Training Set')
plt.xlabel('Time (ordial format)')
plt.ylabel('Adjusted Closing Price')
plt.show()

# Plot the Test Set
plt.scatter(X_test,y_test)
plt.title('Test Set')
plt.xlabel('Time (ordial format)')
plt.ylabel('Adjusted Closing Price')
plt.show()

# * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# [1] Simple Linear Regression Model
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Fitting Simple Linear Regression to the Training Set
lin_regressor = LinearRegression()
lin_regressor.fit(X_train, y_train)

# The model has learnt correlations in the training set and now ready to make predictions
# Create a vector of predicted values of the dependent variable
y_pred = lin_regressor.predict(X_test)

# Visualizing Performance of the Linear Model on the Training Set
plt.scatter(X_train, y_train, color = 'red', label='real training data') # plotting the real observation from the training set
plt.plot(X_train, lin_regressor.predict(X_train), color = 'blue', label='predictions') # plotting the predicted values for the TRAINING set
plt.legend(loc='lower right', shadow=True)
plt.title('Linear Regression on Training Dataset')
plt.xlabel('Time (ordial format)')
plt.ylabel('AAPL - Adjusted Closing Price')
plt.show()

# Visualizing Performance of the Linear Model on the Test Set
plt.scatter(X_test, y_test, color = 'red', label='real test data') # plotting the real observation from the test set
plt.plot(X_test, lin_regressor.predict(X_test), color = 'blue', label='predictions') # plotting the predicted values for the TEST set
plt.legend(loc='center right', shadow=True)
plt.title('Linear Regression on Test Dataset')
plt.xlabel('Time (ordial format)')
plt.ylabel('AAPL - Adjusted Closing Price')
plt.show()



# * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# [2] Quadratic Polynomial Regression Model 
# Degree of Polynomial Features = 2
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Fitting Polynomial Regression to the Training Set
poly2_features = PolynomialFeatures(degree = 2)
X_train_poly2 = poly2_features.fit_transform(X_train)

# Fit the Polynomial Features into Linear Regression model
lin_reg_poly2 = LinearRegression()
lin_reg_poly2.fit(X_train_poly2, y_train)

# Predicting on Training Set
y_train_predicted = lin_reg_poly2.predict(X_train_poly2)

# Predicting on Test Set
y_test_predicted = lin_reg_poly2.predict(poly2_features.fit_transform(X_test))


# Visualizing Performance of the Quadratic Polynomial Regression Model
# on the Training Set
plt.scatter(X_train, y_train, color = 'red', label='real training set')
## Plotting the predicted points
plt.scatter(X_train, y_train_predicted, color = 'green', label='predicted')
plt.title('Quadratic Polynomial Regression on Training Dataset')
plt.legend(loc='lower right', shadow=True)
plt.xlabel('Time (ordial format)')
plt.ylabel('AAPL - Adjusted Closing Price')
plt.show()


# Visualizing Performance of the Quadratic Polynomial Regression Model
## Plotting the true Test observation points
plt.scatter(X_test, y_test, color = 'red', label='real test data')
## Plotting the predicted Test points
plt.scatter(X_test, y_test_predicted, color = 'green', label='predicted')
plt.title('Quadratic Polynomial Regression on Test Dataset')
plt.legend(loc='center right', shadow=True)
plt.xlabel('Time (ordial format)')
plt.ylabel('AAPL - Adjusted Closing Price')
plt.show()



# * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# [3] K-Nearest Neighbor (K-NN) Model 
# k = 5
# - - - - - - - - - - - - - - - - - - - - - - - - - - - -
from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)

# Predicting on Training Set
y_train_predicted = knn_reg.predict(X_train)

# Predicting on Test Set
y_test_predicted = knn_reg.predict(X_test)


# Visualizing Performance of the K-NN Regression Model on Training Set
plt.scatter(X_train, y_train, color = 'red', label='real training set')
plt.scatter(X_train, y_train_predicted, color = 'green', label='predicted')
plt.legend(loc='lower right', shadow=True)
plt.title('K-NN Regression on Training Dataset')
plt.xlabel('Time (ordial format)')
plt.ylabel('AAPL - Adjusted Closing Price')
plt.show()

# Visualizing Performance of the K-NN Regression Model on Test Set
plt.scatter(X_test, y_test, color = 'red', label='real test set')
plt.scatter(X_test, y_test_predicted, color = 'green', label='predicted')
plt.legend(loc='lower right', shadow=True)
plt.title('K-NN Regression on Test Dataset')
plt.xlabel('Time (ordial format)')
plt.ylabel('AAPL - Adjusted Closing Price')
plt.show()

