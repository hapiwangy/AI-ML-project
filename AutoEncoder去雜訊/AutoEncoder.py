#%%
#載入套件
from pickletools import optimize
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D 
# %%
# hyper parameters設定
batch_size = 128 # 訓練批量
max_epochs = 50 # 執行週期
filters = [32,32,16] # 三層捲基層的輸出
# %%
# 取得minst的訓練資料
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
# 因為不用辨識所以可以不用用到label(不用載入Y的資料)

# 標準化
x_train = x_train / 255.
x_test = x_test / 255.

# 加一維:色彩
x_train = np.reshape(x_train, (len(x_train),28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))


# %%
# 加入雜訊
# 在既有的圖像加入雜訊
noise = 0.5

# 固定隨機變數
np.random.seed(11)
tf.random.set_seed(11)

# 隨機加雜訊
x_train_noisy = x_train + noise * np.random.normal(loc=0.0, scale = 1.0, size=x_train.shape)
x_test_noisy = x_test + noise * np.random.normal(loc=0.0, scale = 1.0, size=x_test.shape)

# 裁切數值避免大於1
x_train_noisy = np.clip(x_train_noisy, 0, 1)
x_test_noisy = np.clip(x_test_noisy, 0, 1)

# 轉為浮點數
x_train_noisy = x_train_noisy.astype("float32")
x_test_noisy = x_test_noisy.astype("float32")
# %%
# 建立encoder
class Encoder(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(Encoder, self).__init__()
        self.conv1 = Conv2D(filters=filters[0], kernel_size=3, strides=1, activation="relu", padding="same")
        self.conv2 = Conv2D(filters=filters[1], kernel_size=3, strides=1, activation="relu", padding="same")
        self.conv3 = Conv2D(filters=filters[2], kernel_size=3, strides=1, activation="relu", padding="same")
        self.pool = MaxPooling2D((2, 2), padding="same")

    def call(self, input_features):
        x = self.conv1(input_features)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        return x
# %%
# 建立Decoder
class Decoder(tf.keras.layers.Layer):
    def __init__(self, filters) -> None:
        super(Decoder, self).__init__()
        self.conv1 = Conv2D(filters=filters[2], kernel_size=3,strides=1,activation="relu",padding="same")
        self.conv2 = Conv2D(filters=filters[1], kernel_size=3,strides=1,activation="relu",padding="same")
        self.conv3 = Conv2D(filters=filters[0], kernel_size=3,strides=1,activation="relu",padding="valid")
        self.conv4 = Conv2D(1, 3, 1, activation="sigmoid",padding="same" )
        self.upsample = UpSampling2D((2,2))
    
    def call(self, encoded):
        x = self.conv1(encoded)
        x = self.upsample(x)
        x = self.conv2(x)
        x = self.upsample(x)
        x = self.conv3(x)
        x = self.upsample(x)
        return self.conv4(x)
# %%
# 結合encoder/decoder來建立AutoEncoder模型
class AutoEncoder(keras.Model):
    def __init__(self, filters):
        super(AutoEncoder, self).__init__()
        self.loss = []
        self.encoder = Encoder(filters)
        self.decoder = Decoder(filters)
    
    def call(self, input_features):
        encoded = self.encoder(input_features)
        reconstructed = self.decoder(encoded)
        return reconstructed
# %%
# 訓練模型
model = AutoEncoder(filters)

model.compile(loss = "binary_crossentropy", optimizer = "adam")

loss = model.fit(x_train_noisy, x_train, validation_data=(x_test_noisy, x_test),epochs = max_epochs, batch_size = batch_size)

plt.plot(range(max_epochs), loss.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
# %%
# 比較去雜訊前後的圖像
number = 10
plt.figure(figsize=(20, 4))
for index in range(number):
    # 有雜訊
    ax = plt.subplot(2, number, index + 1)
    plt.imshow(x_test_noisy[index].reshape(28, 28), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # 沒雜訊
    ax = plt.subplot(2, number, index + 1 + number)
    plt.imshow(tf.reshape(model(x_test_noisy)[index], (28, 28)), cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
# %%
