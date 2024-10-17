import cv2
import numpy as np

from natagonreal_v8.trainMulNn import nnm
from neuralnetworkmulticlass import nnmulticlass
from myfunction import convIm

x_test = np.zeros((300 * 300, 1))
y_test = []

nameft = 'data/konata/k05.png'
im0 = cv2.imread(nameft, cv2.IMREAD_GRAYSCALE) / 255
im = cv2.resize(im0, (300, 300))
im = np.array(im)
imA = convIm(im).T
x_test[:, 0] = imA
y_test.append([False, True])

x_test = np.array(x_test)
y_test = np.array(y_test)
y_test = y_test.T

y_pre = nnm.predict(x_test)
y_true = np.argmax(np.array(y_test), 0)

print("X_test :", x_test)
print("Y_predict :", y_pre)
print("y_test :", y_true)