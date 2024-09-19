import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

cancer = datasets.load_breast_cancer()
x = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.DataFrame(cancer.target, columns=['cancer'])
y['cancer'] = y.cancer.apply(lambda x:1 if x == 0 else 0)
df = pd.concat([x,y], axis=1)
scaler = StandardScaler()
scaler.fit(x)
x_sca = scaler.transform(x)
x_sca = pd.DataFrame(x_sca, columns=cancer.feature_names)

sns.pairplot(df, vars=['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'cancer'], hue="cancer")
# sns.show()
# sns.distplot(df, kde=True, rug=False)
plt.show()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_sca, y.values.ravel(), train_size=.8, random_state=101)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver='lbfgs')
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
