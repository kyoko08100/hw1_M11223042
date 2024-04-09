import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist #資料集
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Activation #神經網路層
from tensorflow.keras.models import Model #類神經網路模型
from tensorflow.keras.optimizers import Adam #優化器
from tensorflow.keras.utils import to_categorical #one-hot轉換
from sklearn.metrics import precision_score, recall_score, f1_score
#清空model
keras.backend.clear_session()
np.random.seed(1)
tf.random.set_seed(1)

# 載入 MNIST 數據集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 處理訓練集
x_train = x_train / 255.0
x_train = x_train.reshape((60000, 28, 28, 1))
y_train = to_categorical(y_train, num_classes=10)

# 處理測試集
x_test = x_test / 255.0
x_test = x_test.reshape((10000, 28, 28, 1))
y_test = to_categorical(y_test, num_classes=10)

# 使用 Functional API 建立模型
input_ = Input(shape=(28, 28, 1))
x = Conv2D(64, kernel_size=4, strides=2, padding='same')(input_)
x = Activation('relu')(x)
x = Conv2D(32, kernel_size=4, strides=2, padding='same')(x)
x = Activation('relu')(x)
x = Flatten()(x)
x = Dense(10)(x)
output = Activation('softmax')(x)

optimizer = Adam(learning_rate=0.003)
model = Model(inputs=input_, outputs=output)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer)
model.summary() #可以檢視類神經網路模型的資訊：各層的輸出shape與參數量

# 訓練模型
history = model.fit(x_train, y_train, epochs=4, batch_size=128, validation_data=(x_test, y_test))

# 計算模型在測試集上的預測結果
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# 計算 Precision、Recall 和 F1 Score
precision = precision_score(y_true, y_pred_classes, average='macro')
recall = recall_score(y_true, y_pred_classes, average='macro')
f1 = f1_score(y_true, y_pred_classes, average='macro')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# 繪製訓練過程中的準確度曲線
plt.title('Training Process (Accuracy)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

plt.title('Training Process (Loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()