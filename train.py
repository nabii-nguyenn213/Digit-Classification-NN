import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from NeuralNetworkModel import *
from tqdm import tqdm

def read_data():
    df = pd.read_csv('train.csv')

    x_train = df.drop(columns='label')
    y_train = df['label']
    return x_train, y_train

def convert(img):
    for i in range(len(img)):
        for j in range(len(img[i])):
            if img[i][j] > 0:
                img[i][j] = 1
    return img

def convert_data(x_train):
    for i in range(len(x_train)):
        xi = x_train.iloc[i, :]
        img = np.array(xi).reshape(28, 28)
        con = convert(img)
        x_train.iloc[i, :] = np.array(con).reshape(784, )

def one_hot_coding(y_train):
    m = len(y_train.unique())
    y_new = []
    for i in y_train.values:
        y_n = [0] * m
        y_n[i] = 1
        y_new.append(y_n)
    y_new = pd.DataFrame(y_new)
    return y_new

model = DeepNeuralNetwork(
        Layer(28*28),
        Layer(64, activation_func='relu'),
        Layer(64, activation_func='relu'),
        Layer(10, activation_func='softmax', train_bias=False)
    )

def fit(lr = 0.01, batch_size = 64, epochs = 1000):
    x_train, y_train = read_data()
    convert_data(x_train)
    y_train_one_hot = one_hot_coding(y_train)
    y_train_one_hot.index = y_train
    
    N, d = x_train.shape
    for it in tqdm(range(epochs)):
        rand_id = np.random.choice(N, size=batch_size, replace=False)
        for i in rand_id:
            xi = np.array([x_train.iloc[i, :]]).T
            yi = np.array([y_train_one_hot.iloc[i, :]]).T
            activation_cache, linear_cache = model.forward(xi)
            model.backpropagation(linear_cache, activation_cache, yi)
            model.update(lr = lr)
    
    y_train.to_numpy()
    y_pred = model.predict(x_train)
    acc = model.accuracy(y_train, y_pred)
    print("accuracy :", acc)

def pred(x):
    y = model.predict_single_point(x)
    return y