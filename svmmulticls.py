import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import cvxopt
import copy


class SVM:
    linear = lambda x, xp, c=0: x @ xp.T
    polynomial = lambda x, xp, Q=5: (1 + x @ xp.T) ** Q
    rbf = lambda x, xp, y=10: np.exp(-y * distance.cdist(x, xp, 'sqeuclidean'))
    kernel_funs = {'linear': linear, 'polynomial': polynomial, 'rbf': rbf}

    def __init__(self, kernel='rbf', C=1, k=2):
        self.kernel_str = kernel
        self.kernel = SVM.kernel_funs[kernel]
        self.C = C
        self.k = k
        self.X, self.y, self.aps = None, None, None
        self.multiclass = False
        self.clfs = []

    def fit(self, X, y, eval_train=False):
        if len(np.unique(y)) > 2:
            self.multiclass = True
            return self.multi_fit(X, y, eval_train)

        if set(np.unique(y)) == {0, 1}:
            y[y == 0] = -1

        self.y = y.reshape(-1, 1).astype(np.double)
        self.X = X
        N = X.shape[0]
        self.K = self.kernel(X, X, self.k)

        # Formulate the optimization problem
        P = cvxopt.matrix(self.y @ self.y.T * self.K)
        q = cvxopt.matrix(-np.ones((N, 1)))
        A = cvxopt.matrix(self.y.T)
        b = cvxopt.matrix(np.zeros(1))
        G = cvxopt.matrix(np.vstack((-np.identity(N), np.identity(N))))
        h = cvxopt.matrix(np.vstack((np.zeros((N, 1)), np.ones((N, 1)) * self.C)))

        cvxopt.solvers.options['show_progress'] = False
        sol = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.aps = np.array(sol["x"])

        self.is_sv = ((self.aps > 1e-3) & (self.aps <= self.C)).squeeze()
        self.margin_sv = np.argmax((1e-3 < self.aps) & (self.aps < self.C - 1e-3))

        if eval_train:
            print(f"Finished training with accuracy {self.evaluate(X, y)}")

    def multi_fit(self, X, y, eval_train=False):
        self.k = len(np.unique(y))
        for i in range(self.k):
            Xs, Ys = X, copy.copy(y)
            Ys[Ys != i], Ys[Ys == i] = -1, +1
            clf = SVM(kernel=self.kernel_str, C=self.C, k=self.k)
            clf.fit(Xs, Ys)
            self.clfs.append(clf)
        if eval_train:
            print(f"Finished training with accuracy {self.multi_evaluate(X, y)}")

    def predict(self, X_t):
        if self.multiclass:
            return self.multi_predict(X_t)

        xs, ys = self.X[self.margin_sv, np.newaxis], self.y[self.margin_sv]
        aps, y, X = self.aps[self.is_sv], self.y[self.is_sv], self.X[self.is_sv]
        b = ys - np.sum(aps * y * self.kernel(X, xs, self.k), axis=0)
        score = np.sum(aps * y * self.kernel(X, X_t, self.k), axis=0) + b
        return np.sign(score).astype(int), score

    def multi_predict(self, X):
        preds = np.zeros((X.shape[0], self.k))
        for i, clf in enumerate(self.clfs):
            _, preds[:, i] = clf.predict(X)
        return np.argmax(preds, axis=1)

    def evaluate(self, X, y):
        outputs, _ = self.predict(X)
        accuracy = np.sum(outputs == y) / len(y)
        return round(accuracy, 2)

    def multi_evaluate(self, X, y):
        outputs = self.multi_predict(X)
        accuracy = np.sum(outputs == y) / len(y)
        return round(accuracy, 2)


# Data preparation
X = np.array([[1, 7], [2, 7], [7, 7], [2, 6], [4, 6], [6, 6], [7, 6],
              [2, 5], [3, 5], [4, 5], [5, 5], [6, 5], [7, 5],
              [3, 4], [5, 4], [6, 4], [2, 3], [3, 3], [4, 3], [5, 3],
              [1, 2], [2, 2], [4, 2], [5, 2], [6, 2], [7, 2],
              [1, 1], [2, 1], [6, 1], [7, 1]])

y = np.array([2, 2, 2, 1, 1, 1, 2,
              1, 0, 0, 0, 1, 2,
                       0, 0, 1,
                 1, 0, 0, 0,
              2, 1, 1, 1, 1, 2,
              2, 2, 2, 2])

# Test SVM
svm = SVM(kernel='rbf', k=3)
svm.fit(X, y, eval_train=True)

# Prediction
y_pred = svm.predict(np.array([[2, 4]]))
print(f"Predicted class: {y_pred[0]}")
print(f"Accuracy: {np.sum(y == y_pred[0]) / y.shape[0]}")

# Visualization
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title("SVM Classification with 3 Square Layers")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
