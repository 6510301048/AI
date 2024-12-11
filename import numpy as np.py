import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

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
# Adjust the graph to center the data
x_center = (X[:, 0].max() + X[:, 0].min()) / 2
y_center = (X[:, 1].max() + X[:, 1].min()) / 2
x_range = max(abs(X[:, 0].max() - x_center), abs(X[:, 0].min() - x_center))
y_range = max(abs(X[:, 1].max() - y_center), abs(X[:, 1].min() - y_center))
margin = 1.0  # Add some margin to the graph
x_min, x_max = x_center - (x_range + margin), x_center + (x_range + margin)
y_min, y_max = y_center - (y_range + margin), y_center + (y_range + margin)

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
plt.gca().set_xticks(np.linspace(x_min, x_max, 6))  # Set 6 ticks on x-axis
plt.gca().set_yticks(np.linspace(y_min, y_max, 6))  # Set 6 ticks on y-axis

plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.title('Centered Decision Boundary with 6x6 Grid')
plt.show()
