import numpy as np

class nnmulticlass:
    def Relu(self, x):
        return np.maximum(x,0)

    def softMax(self, x):
        expx = np.exp(x)
        return expx / np.sum(expx, axis=0)

    def derivative_Relu(self, u):
        return np.array(u > 0, dtype=np.float16)

    def initialize_paranetera(self, n_x, n_h, n_y):
        self.w1 = np.random.randn(n_h, n_x) * 0.01
        self.b1 = np.zeros((n_h, 1))

        self.w2 = np.random.randn(n_y, n_h) * 0.01
        self.b2 = np.zeros((n_y, 1))

        self.parameters = {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2
        }
        return self.parameters

    def forward_propagation(self, x, parameters):
        w1 = parameters['w1']
        b1 = parameters['b1']
        w2 = parameters['w2']
        b2 = parameters['b2']

        z1 = np.dot(w1, x) + b1
        a1 = self.Relu(z1)

        z2 = np.dot(w2, a1) + b2
        a2 = self.softMax(z2)

        self.forward_cache = {
            "z1": z1,
            "a1": a1,
            "z2": z2,
            "a2": a2
        }
        return self.forward_cache

    def cost_function(self, a2, y):
        m = y.shape[1]
        self.cost = -(1 / m) * np.sum(y * np.log(a2))

        return self.cost

    def backward_prop(self, x, y, parameters, forward_cache):
        w1 = parameters['w1']
        b1 = parameters['b1']
        w2 = parameters['w2']
        b2 = parameters['b2']

        a1 = forward_cache['a1']
        a2 = forward_cache['a2']

        m = x.shape[1]

        dz2 = (a2 - y)
        dw2 = (1 / m) * np.dot(dz2, a1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        dz1 = (1 / m) * np.dot(w2.T, dz2) * self.derivative_Relu(a1)
        dw1 = (1 / m) * np.dot(dz1, x.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        self.gradients = {
            "dw1": dw1,
            "db1": db1,
            "dw2": dw2,
            "db2": db2
        }
        return self.gradients

    def update_parameters(self, parameters, gradients, Learning_rate):
        w1 = parameters['w1']
        b1 = parameters['b1']
        w2 = parameters['w2']
        b2 = parameters['b2']

        dw1 = gradients['dw1']
        db1 = gradients['db1']
        dw2 = gradients['dw2']
        db2 = gradients['db2']

        self.w1 = w1 - Learning_rate * dw1
        self.b1 = b1 - Learning_rate * db1
        self.w2 = w2 - Learning_rate * dw2
        self.b2 = b2 - Learning_rate * db2

        self.parameters = {
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2
        }
        return self.parameters

    def fit(self, x_train, y_train, epoch=5000):
        n_x = x_train.shape[0]
        n_y = y_train.shape[0]
        n_h = 11
        self.Learning_rate = 0.01
        self.parameters = self.initialize_paranetera(n_x, n_h, n_y)
        for i in range(epoch):
            self.forward_cache = self.forward_propagation(np.array(x_train), self.parameters)
            self.cost = self.cost_function(self.forward_cache['a2'], y_train)
            self.gradients = self.backward_prop(np.array(x_train), np.array(y_train), self.parameters, self.forward_cache)
            self.parameters = self.update_parameters(self.parameters, self.gradients, self.Learning_rate)

    def predict(self, x_test):
        forward_ch = self.forward_propagation(x_test, self.parameters)
        y_pre = np.argmax(forward_ch['a2'], 0)

        return y_pre

