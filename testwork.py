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

# Step 1: Generate synthetic data using make_blobs
X, y = make_blobs(n_samples=200, centers=[[2.0, 2.0], [3.0, 3.0]], cluster_std=0.75, n_features=2, random_state=69)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)

# Step 3: Create a neural network model
model = Sequential()
model.add(Dense(16, input_dim=2, activation="relu"))
model.add(Dense(1, activation='sigmoid'))
optimizer = Adam(learning_rate=0.01)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Step 4: Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

# Step 5: Plot the decision boundary
# Set graph boundaries to -3 to 3
x_min, x_max = -3, 3
y_min, y_max = -3, 3

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid)
Z = (Z > 0.5).astype(int).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.RdBu)  # Decision regions
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.RdBu)  # Data points
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# Adding a grid with 6x6 cells
plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')
plt.gca().set_xticks(np.linspace(x_min, x_max, 7))  # Set 6 ticks on x-axis
plt.gca().set_yticks(np.linspace(y_min, y_max, 7))  # Set 6 ticks on y-axis

plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.title('Centered Decision Boundary with 6x6 Grid')
plt.show()
