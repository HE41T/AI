import cv2

im = cv2.imread('lena_gray.bmp', cv2.IMREAD_GRAYSCALE)

a = im[0: 5, 507:512]
print(a)

# while True:
#     cv2.imshow('subimage',a)
#     cv2.waitKey(2)
# cv2.destroyAllWindows()
