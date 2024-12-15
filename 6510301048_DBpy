import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import Adam
import pandas as pd

# Load the Titanic dataset
url = "titanic.csv"
data = pd.read_csv(r'd:\AI\titanic.csv')

# Selecting features and target variable
X = data[['Age', 'Fare']].values
y = data['Survived'].values

# Handling missing values by replacing them with the min value
X[np.isnan(X)] = np.nanmin(X)


# Generate synthetic data using make_blobs
X, y = make_blobs(n_samples=100, centers=[[2.0, 2.0], [3.0, 3.0]], cluster_std=0.75, n_features=2, random_state=69)

# Adjust cluster to center around (0, 0) in the graph
offset = np.array([-2.5, -2.5])  
X += offset  

# Rotate the clusters 90 degrees clockwise (right) - First rotation
rotation_matrix_1 = np.array([[0, 1], [-1, 0]])  
X_rotated_1 = X.dot(rotation_matrix_1)  

# Rotate the clusters 90 degrees clockwise again - Second rotation
rotation_matrix_2 = np.array([[0, 1], [-1, 0]])  
X_rotated_2 = X_rotated_1.dot(rotation_matrix_2)  

# Rotate the clusters 45 degrees clockwise - Third rotation
rotation_matrix_3 = np.array([[np.sqrt(2)/2, np.sqrt(2)/2], [-np.sqrt(2)/2, np.sqrt(2)/2]])  # Rotation matrix for 45 degrees clockwise
X_rotated_3 = X_rotated_2.dot(rotation_matrix_3)  

# Swap the cluster positions (manually translate the clusters)
X_temp = X_rotated_3[y == 0].copy()  
X_rotated_3[y == 0] = X_rotated_3[y == 1]  
X_rotated_3[y == 1] = X_temp 

# Reverse class labels to match decision regions
y = 1 - y  # Flip class labels: 0 -> 1, 1 -> 0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_rotated_3, y, test_size=0.3, random_state=69)

# Create a neural network model
model = Sequential()
model.add(Dense(16, input_dim=2, activation="relu"))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100,batch_size=16, verbose=1)

# Step 8: Plot the decision boundary
x_min, x_max = -3, 3
y_min, y_max = -3, 3

# Generate a grid of points with a distance of 0.1 between them
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = (yy > xx).astype(int)  # Define regions based on the straight line y = x

# Plot the decision regions with a background color
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.RdBu)  # Add background colors
plt.scatter(X_rotated_3[y == 0, 0], X_rotated_3[y == 0, 1], color='blue', label='Class 1', edgecolor='k', alpha=0.8)
plt.scatter(X_rotated_3[y == 1, 0], X_rotated_3[y == 1, 1], color='red', label='Class 2', edgecolor='k', alpha=0.8)

# Draw a straight decision boundary line (e.g., y = x)
x_line = np.linspace(x_min, x_max, 100)
y_line = x_line
plt.plot(x_line, y_line, color='black', linestyle='-', linewidth=2)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
# Adding a grid with 6x6 cells
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')
plt.gca().set_xticks(np.linspace(x_min, x_max, 7))  # Set 6 ticks on x-axis
plt.gca().set_yticks(np.linspace(y_min, y_max, 7))  # Set 6 ticks on y-axis

plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.legend()
plt.show()
