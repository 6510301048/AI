import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import Perceptron

# สร้างข้อมูลสองกลุ่มที่สมมาตร
x1, y1 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(2, 2),  # กลุ่ม 1 อยู่ที่ (-1, -1)
                    cluster_std=0.25,
                    random_state=69)

x2, y2 = make_blobs(n_samples=100,
                    n_features=2,
                    centers=1,
                    center_box=(3, 3),  # กลุ่ม 2 อยู่ที่ (1, 1)
                    cluster_std=0.25,
                    random_state=69)

# รวมข้อมูลและกำหนด label
X = np.vstack((x1, x2))
y = np.hstack((np.zeros(x1.shape[0]), np.ones(x2.shape[0])))

# ปรับข้อมูลให้อยู่ในช่วงที่สมมาตร
X -= np.mean(X, axis=0)

# สร้าง Perceptron และฝึกโมเดล
model = Perceptron()
model.fit(X, y)

# สร้าง Grid สำหรับ Decision Regions
x1_range = np.linspace(-3.0, 3.0, 500)
x2_range = np.linspace(-3.0, 3.0, 500)
x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
g_values = model.decision_function(grid).reshape(x1_grid.shape)

# แสดงค่า Weights และ Bias
weights = np.round(model.coef_[0], 2)
bias = np.round(model.intercept_[0], 2)
print(f"Weights: {weights}")
print(f"Bias: {bias}")

# คำนวณ Accuracy
accuracy = model.score(X, y)
print(f"Accuracy: {accuracy:.2f}")

# วาด Decision Regions และ Boundary
plt.figure(figsize=(8, 6))  # ปรับขนาดให้สมมาตร
plt.contourf(x1_grid, x2_grid, g_values, levels=[-np.inf, 0, np.inf], colors=['red', 'blue'], alpha=0.4)
plt.contour(x1_grid, x2_grid, g_values, levels=[0], colors='black', linewidths=2)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='purple', label='Class 1', edgecolor='k', alpha=0.8)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='yellow', label='Class 2', edgecolor='k', alpha=0.8)

# ตั้งค่าช่วงแกนและจำนวนช่องในตาราง
plt.xlim(-3, 3)  # กำหนดช่วงแกน x
plt.ylim(-3, 3)  # กำหนดช่วงแกน y
plt.xticks(np.linspace(-3.0, 3.0, 7))  # แบ่งช่องแกน x
plt.yticks(np.linspace(-3.0, 3.0, 7))  # แบ่งช่องแกน y

# ตั้งค่าให้กราฟสมมาตร
plt.gca().set_aspect('equal', adjustable='box')  # ทำให้กราฟมีความสมมาตร

plt.xlabel('Feature x1')
plt.ylabel('Feature x2')
plt.title('Decision Plane')
plt.legend()
plt.grid(True)
plt.show()
