import cv2
import numpy as np

im = cv2.imread('lena_gray.bmp', cv2.IMREAD_GRAYSCALE)
im = np.array(im)
print(im.shape)
print(im.shape[0])
print(im.shape[1])
imarr = np.zeros((im.shape[0]*im.shape[1], 1))
print()

cnt = 0
for row in range(im.shape[0]):
    for col in range(im.shape[1]):
        imarr[cnt] = im[row, col]
        cnt = cnt + 1

imarrT = imarr.T
print()
