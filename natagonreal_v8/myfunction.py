import numpy as np

def convIm(im):
    imarr = np.zeros((im.shape[0] * im.shape[1]))

    cnt = 0
    for row in range(im.shape[0]):
        for col in range(im.shape[1]):
            imarr[cnt] = im[row, col]
            cnt = cnt + 1

    return imarr

