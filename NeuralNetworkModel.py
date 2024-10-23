import numpy as np
import math
from Layers import Layer

class DeepNeuralNetwork:
    
    def __init__(self, *layers):
        self.layers = layers
        self.layers_len = len(layers)
        self.weights = self.initialize()
        self.reshape_bias()
        self.dw, self.db = self.init_derivative()
        
    def initialize(self):
        weight = []
        for i in range(self.layers_len - 1):
            w = (np.random.rand(self.layers[i].get_dim(), self.layers[i + 1].get_dim())) * (math.sqrt(2 / (self.layers[i].get_dim() + self.layers[i + 1].get_dim())))
            weight.append(w)
        return weight
    
    def reshape_bias(self):
        for i in range(self.layers_len - 1):
            if self.layers[i].train_bias:
                self.layers[i].biases = np.zeros((self.weights[i].shape[1], 1))
    
    def init_derivative(self):
        db = []
        dw = []
        for i in range(self.layers_len):
            if self.layers[i].train_bias:
                db.append(np.zeros(self.layers[i].biases.shape))
            if i != self.layers_len - 1:
                dw.append(np.zeros((self.layers[i].get_dim(), self.layers[i + 1].get_dim())))
        return dw, db
    
    def __call__(self, x):
        return self.forward(x)[0][-1]
    
    def forward(self, x):
        activation_cache = [x]
        linear_cache = []
        for l in range(1, self.layers_len):
            z = np.matmul(self.weights[l-1].T, activation_cache[-1]) + self.layers[l-1].biases
            # print(f'z{l-1} shape :', z.shape)
            linear_cache.append(z)
            a = self.layers[l].activate(linear_cache[-1])
            # print(f'a{l} shape :', a.shape)
            activation_cache.append(a)
            
        return activation_cache, linear_cache
    
    def backpropagation(self, linear_cache, activation_cache, y_true):
        dZ = activation_cache[-1] - y_true
        for l in range(self.layers_len - 2, -1, -1):
            self.dw[l] = np.matmul(activation_cache[l], dZ.T)
            self.db[l] = dZ
            if l > 0:
                dA = np.matmul(self.weights[l], dZ)
                dZ = np.multiply(dA, self.layers[l].derivative(linear_cache[l-1]))
    
    def update(self, lr = 0.01):
        for i in range(self.layers_len - 1):
            self.weights[i] = self.weights[i] - lr * self.dw[i]
            self.layers[i].biases = self.layers[i].biases - lr * self.db[i]
    
    def predict(self, x):
        y_pred = []
        for i in x.values:
            i = np.array([i]).T
            output = self.forward(i)[0][-1]
            digit = np.argmax(output)
            y_pred.append(digit)
        return np.array(y_pred)
    
    def predict_single_point(self, x):
        output = self.forward(x)[0][-1]
        digit = np.argmax(output)
        return digit
    
    def accuracy(self, y_true, y_pred):
        y_true = y_true.to_numpy()
        c = 0
        for i in range(len(y_pred)):
            if y_pred[i] != y_true[i]:
                c += 1
        return 1-(c/len(y_pred))