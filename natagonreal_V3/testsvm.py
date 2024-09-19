from svm import SVM
import  numpy as np
svm = SVM()
datayy = [1,1,1,-1,-1,-1]
dataxx = [[1,5],[1,4],[2,4],[5,3],[4,2],[5,2]]
xx_train = np.array(dataxx)
yy_train = np.array(datayy)
w, b, losses = svm.fit(xx_train, yy_train)

xx_test = np.array([[1,3],[5,1]])
yy_test = [1,-1]
predict = svm.predict(xx_test)
print("YY_TEST : ", yy_test)
print("\nPREDICT : ", predict)
