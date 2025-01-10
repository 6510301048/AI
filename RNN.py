import keras.api.models as mod
import keras.api.layers as lay
import numpy as np
import matplotlib.pyplot as plt

#สร้างโมเดล RNN
model = mod.Sequential()
model.add(lay.SimpleRNN(units=1,
                        input_shape=(1, 1),
                        activation='relu'))

model.summary()
model.save("RNN.h5")

#กำหนดพารามิเตอร์และฟังก์ชันสร้างข้อมูล
pitch = 20
step = 1
N = 500
n_train = int(N * 0.7)

def gen_data(x):
  return (x % pitch) / pitch

#สร้างข้อมูลตัวอย่าง
t = np.arange(1, N + 1)
y = np.sin(0.05 * t * 10) + 0.8 * np.random.rand(N)
y = np.array(y)

#แสดงกราฟข้อมูล
plt.figure()
plt.plot(y)
plt.show()

#ฟังก์ชันแปลงข้อมูลเป็น Matrix
def convertToMatrix(data, step=2):
  X, Y = [], []
  for i in range(len(data) - step):
     d = i + step
     X.append(data[i:d, ])
     Y.append(data[d])
  return np.array(X), np.array(Y)

#แยกข้อมูลเป็นชุด train และ test
train, test = y[0:n_train], y[n_train:N]
x_train, y_train = convertToMatrix(train, step)
x_test, y_test = convertToMatrix(test, step)

print("Dimension (Before)", train.shape, test.shape)
print("Dimension (After)", x_train.shape, x_test.shape)

#ฟังก์ชันแสดงผลการทำนาย
def plot_predictions(y_true, y_pred, title="Actual vs. Predicted"):
  plt.figure(figsize=(10, 6))
  plt.plot(y_true, label='Original', color='blue')
  plt.plot(y_pred, label='Predict', linestyle='--', color='red')
  plt.title(title)
  plt.xlabel('Time')
  plt.ylabel('Value')
  plt.legend()
  plt.grid(True)
  plt.show()

#ทำนายผลและแสดงกราฟ
predictions = model.predict(x_test)
plot_predictions(y_test, predictions, title="RNN")