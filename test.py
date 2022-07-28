# 載入相關套件
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# 超參數設定
batch_size = 128     # 訓練批量
max_epochs = 50      # 訓練執行週期
filters = [32,32,16] # 三層卷積層的輸出個數
# 只取 X ，不需 Y
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# 常態化
x_train = x_train / 255.
x_test = x_test / 255.

# 加一維：色彩
x_train = np.reshape(x_train, (len(x_train),28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
print(x_train, x_test,"finish")