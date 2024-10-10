from sklearn.datasets import make_classificationimport numpy as np
from svmmulticls import SVMimport matplotlib.pyplot as plt
np.random.seed(1)
X =np.array([[4,4],[4,5],[4,6],[4,7],[4,8],[4,9],[4,10],
             [5,4],[5,5],[5,6],[5,7],[5,8],[5,9],[5,10],             [6,4],[6,5],[6,6],[6,7],[6,8],[6,9],[6,10],
             [7,4],[7,5],[7,6],[7,7],[7,8],[7,9],[7,10],             [8,4],[8,5],[8,6],[8,7],[8,8],[8,9],[8,10],
             [9,4],[9,5],[9,6],[9,7],[9,8],[9,9],[9,10],             [10,4],[10,5],[10,6],[10,7],[10,8],[10,9],[10,10]])
y = np.array([2, 2, 2, 2, 2, 2, 2,
              2, 1, 1, 1, 1, 1, 2,              2, 1, 0, 0, 0, 1, 2,
              2, 1, 0, 0, 0, 1, 2,              2, 1, 0, 0, 0, 1, 2,
              2, 1, 1, 1, 1, 1, 2,              2, 2, 2, 2, 2, 2, 2])
# สร้างและฝึก SVM
svm = SVM(kernel='rbf', k=1)svm.fit(X, y, eval_train=True)
# สร้างกริดสำหรับการวาดเส้นแบ่ง
xx, yy = np.meshgrid(np.linspace(3, 11, 100), np.linspace(3, 11, 100))Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# วาดกราฟplt.contourf(xx, yy, Z, alpha=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')plt.title("SVM with RBF Kernel")
plt.xlabel("X1")plt.ylabel("X2")
plt.xlim(0, 11)plt.ylim(0, 11)
plt.legend()plt.show()
y_pred = svm.predict(np.array([[5,8]]))
print(y_pred)
print(f"Accuracy: {np.sum(y == y_pred)/y.shape[0]}")
from sklearn.svm import SVCclf = SVC(kernel='rbf', C=1, gamma=1)
clf.fit(X, y)y_pred = clf.predict(X)
print(f"Accuracy: {sum(y == y_pred)/y.shape[0]}")