import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ใข้ค่า BMI นน กับ สส มาหาค่าที่เสี่ยงเป็นโรคอ้วน โดย 0 = ไม่เสี่ยง , 1 = เสี่ยง ,เสี่ยงคือ = BMI >= 25
x = np.array([[70, 175], [55,168], [85,180], [92, 178], [65,172], [73,174], [88,182], [95,180], [68,176], [76,177],
                [82,180], [89,179], [78,173], [90,181], [67,170], [80,176], [87,183], [93,178], [72,171], [84,179],
                [77,175], [91,180], [69,172], [83,178], [88,181], [70,173], [86,179], [79,177], [94,182], [75,174]])

y = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
     1, 1, 1, 1, 0, 1, 1, 1, 0, 1,
     1, 1, 0, 1, 1, 0, 1, 1, 1, 0]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .2,random_state=222)
print("x_test = \n", x_test)
print("y_test = ", y_test)
print('-----------------')
print("x_train = \n", x_train)
print("y_train = ", y_train)

model = LogisticRegression(solver='lbfgs')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("X_test = ",y_pred)
y_pred1 = model.predict(np.array([[60,170],[90,180]]))
print("y_predict = ",y_pred1)
