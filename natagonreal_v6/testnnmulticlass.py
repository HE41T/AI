from sklearn.metrics import consensus_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from neuralnetworkmulticlass import nnmulticlass

def train_test_split(df,test_size):
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df

data = pd.read_csv("glass.txt", sep="\t", header=None)
del data[11]

data.head()

train_data, test_data = train_test_split(data, test_size=0.3)

x_train = train_data.iloc[:,0:10].T
y_train = pd.get_dummies(train_data.iloc[:,-1],prefix='y').T

x_test = test_data.iloc[:,0:10].T
y_test = pd.get_dummies(test_data.iloc[:,-1],prefix='y').T

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

nnm = nnmulticlass()
nnm.fit(x_train, y_train, 10000)

# forward_ch = nnm.forward_propagation(x_test, nnm.parameters)
y_pre = nnm.predict(x_test)
y_true = np.argmax(np.array(y_test), 0)

print("X_test :", x_test)
print("Y_predict :", y_pre)
print("y_test  :", y_true)

conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pre)
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Predictions', fontsize=18)
plt.show()
