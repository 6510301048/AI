import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    "Day": ["22nd (Mon.)", "23rd (Tues.)", "24th (Wed.)", "25th (Thurs.)", "26th (Fri.)"],
    "High_temp_C": [29, 28, 34, 31, 25],
    "Iced_tea_orders": [77, 62, 93, 84, 59]
}

# แปลงข้อมูลเป็น DataFrame
df = pd.DataFrame(data)

# เตรียมข้อมูลสำหรับโมเดลถดถอยเชิงเส้น
X = np.array(df["High_temp_C"]).reshape(-1, 1)  # ตัวแปรอิสระ (x)
y = np.array(df["Iced_tea_orders"])            # ตัวแปรตาม (y)

# สร้างโมเดลถดถอยเชิงเส้น
model = LinearRegression()
model.fit(X, y)

# คำนวณสมการถดถอย
slope = model.coef_[0]  # ความชัน
intercept = model.intercept_  # จุดตัดแกน y
print(f"RE: y = {slope:.2f}x  {intercept:.2f}")

# สร้างกราฟ
plt.figure(figsize=(8, 6))
plt.scatter(df["High_temp_C"], df["Iced_tea_orders"], color="blue", label="Data points")
plt.plot(df["High_temp_C"], model.predict(X), color="red", label="Regression line")
plt.title("Linear Regression: High Temperature vs Iced Tea Orders")
plt.xlabel("High Temperature (°C)")
plt.ylabel("Iced Tea Orders")
plt.legend()
plt.grid(True)

# บันทึกกราฟเป็นไฟล์ .png
plt.savefig("regression_graph.png")
plt.show()
