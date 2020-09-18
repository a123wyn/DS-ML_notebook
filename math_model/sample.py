import numpy as np
import pandas as pd
from softmax import Softmax
from sklearn.datasets import load_iris

xtrain = pd.read_excel("xtrain_v2.xlsx",index_col=0)
ytrain = pd.read_excel("ytrain_v2.xlsx", index_col=0)
xtest = pd.read_excel("xtest_v2.xlsx", index_col=0)
reg_strength = 1e-4
batch_size = 50
epochs = 100
learning_rate = 5e-2
weight_update = 'sgd'
sm = Softmax(batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, reg_strength=reg_strength, weight_update=weight_update)
sm.train(xtrain, ytrain.values)
pred = sm.predict(xtest)
print(np.mean(np.equal(ytrain, pred)))
