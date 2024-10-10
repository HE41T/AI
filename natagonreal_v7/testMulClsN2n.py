import glob
import cv2
import numpy as np
from neuralnetworkmulticlass import nnmulticlass


def convIm(im):
    imarr = np.zeros((im.shape[0] * im.shape[1]))

    cnt = 0
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            imarr[cnt] = im[row, col]
            cnt = cnt + 1
    return imarr


lstFolder = ['wow3', 'wow4']
numclass = len(lstFolder)
lstN = []
for cntC in range(numclass):
    lstName = glob.glob("data/" + lstFolder[cntC] + "/*")
    numfile = len(lstName)
    lstNs = []

    for cntf in range(numfile):
        lstNs.append(lstName[cntf])

    lstN.append(lstNs)
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

# print(lstN[0])
# print(lstN[1])

namef = ''
Y_train = []
X_train = np.zeros((300 * 300, len(lstN[0]), len(lstN[1])))

numf = 0
for cntc in range(numclass):
    for cntf in range(len(lstN[cntc])):
        namef = lstN[cntc][cntf]

        print(namef)
        if cntc != 0:
            Y_train.append([False, True])
        else:
            Y_train.append([True, False])

        im0 = cv2.imread(namef, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(im0, (300, 300))
        im = np.array(im / 255)

        imA = convIm(im).T
        X_train[:, numf] = imA
        numf = numf + 1

X_train = np.array(X_train)
Y_train = np.array(Y_train)
Y_train = Y_train.T

# Assuming nnmulticlass is your custom neural network class
nnm = nnmulticlass()
nnm.fit(X_train, Y_train, 3000)

# ///////////////////////////////////////
x_test = np.zeros((300 * 300, 1))
y_test = []

nameft = 'data/B1.png'
im0 = cv2.imread(namef, cv2.IMREAD_GRAYSCALE) / 255
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
