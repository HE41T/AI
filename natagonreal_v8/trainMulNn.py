import glob
import cv2
import numpy as np
from neuralnetworkmulticlass import nnmulticlass
from myfunction import convIm

lstFolder = ['wow3', 'wow4', 'konata']
numclass = len(lstFolder)
totalfile = 0
lstN = []
for cntc in range(numclass):
    lstname = glob.glob("data/" + lstFolder[cntc] + "/*")
    numfile = len(lstname)
    lstNs = []
    for cntf in range(numfile):
        lstNs.append(lstname[cntf])
        totalfile = totalfile + 1
    lstN.append(lstNs)

# ------------------Train------------------
y_train = []
x_train = np.zeros((300 * 300, totalfile))
namef = ''
numf = 0
for cntc in range(numclass):
    for cntf in range(len(lstN[cntc])):
        namef = lstN[cntc][cntf]
        print(namef)

        if cntc == 2:
            y_train.append([False, False, True])
        elif cntc == 1:
            y_train.append([False, True, False])
        elif cntc == 0:
            y_train.append([True, False, False])

        im0 = cv2.imread(namef, cv2.IMREAD_GRAYSCALE)/255
        im = cv2.resize(im0,(300, 300))
        im = np.array(im)

        imA = convIm(im).T
        x_train[:, numf] = imA
        numf = numf + 1

x_train = np.array(x_train)
y_train = np.array(y_train)
y_train = y_train.T
nnm = nnmulticlass()
nnm.fit(x_train, y_train, 1000)
