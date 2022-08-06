#%%
# 載入相關定義
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from scipy.stats import norm
# %%
# 取得mnist訓練資料
(x_tr, y_tr), (x_te, y_te) = mnist.load_data()
x_tr, x_te = x_tr.astype('float32') / 255., x_te.astype('float32')/255. 
x_tr, x_te = x_tr.reshape(x_tr.shape[0], -1), x_te.reshape(x_te.shape[0], -1)
print(x_tr.shape, x_te.shape)
# %%
# hyper parameters 設定
batch_size, n_epoch = 100, 100
n_hidden, z_dim = 256, 2 #encoder的hidden神經元個數、output layer的神經元個數
# %%
# encoder
x = Input(shape=(x_tr.shape[1:]))
x_encoded = Dense(n_hidden, activation="relu")(x)
x_encoded = Dense(n_hidden//2, activation="relu")(x_encoded)

# encoder後接dense，估算平均數mu
mu = Dense(z_dim)(x_encoded)

# encoder後接dense，估算log變異數log_var
log_var = Dense(z_dim)(x_encoded)

# %%
# 定義抽樣函數
def sampling(args):
    # 根據mu, log_var取隨機變數
    mu, log_var = args
    eps = K.random_normal(shape=(batch_size, z_dim), mean=0., stddev=1.0)
    return mu + K.exp(log_var) * eps

# 定義匿名函數
z = Lambda(sampling, output_shape=(z_dim,))([mu, log_var])

# %%
