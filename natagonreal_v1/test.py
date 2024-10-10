import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

x = np.array ([[100], [90], [80],[70],[60],[50],[40],[30],[20],[10],[5]])
print(x)
y = [1,1,1,1,1,1,0,0,0,0,0]
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state=42)
print("x_test = ", x_test)
print("y_test = ", y_test)
print('-----------------')
print("x_train = ", x_train)
print("y_train = ", y_train)

model = LogisticRegression(solver='lbfgs')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print(y_pred)
y_pred1 = model.predict(np.array([[16],[65]]))
print(y_pred1)
