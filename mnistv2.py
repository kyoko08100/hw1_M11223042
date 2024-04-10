import tensorflow as tf
import keras
import numpy as np 
import pandas as pd 
from keras import utils
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense
np.random.seed(10)
from keras.datasets import mnist
from sklearn.metrics import precision_score, recall_score, f1_score

(x_train_image,y_train_label),(x_test_image,y_test_label)=mnist.load_data()

print('train data= ',len(x_train_image))
print('test data=', len(x_test_image))

import matplotlib.pyplot as plt

def plot_image(image):
  fig = plt.gcf()
  fig.set_size_inches(2,2)
  plt.imshow(image,cmap='binary')
  plt.show()

#plot_image(x_train_image[0]) 

# 建立函數要來畫多圖的
def plot_images_labels_prediction(images,labels,prediction,idx,num=10): 
  
  # 設定顯示圖形的大小
  fig= plt.gcf()
  fig.set_size_inches(12,14)

  # 最多25張
  if num>25:num=25

  # 一張一張畫
  for i in range(0,num):

    # 建立子圖形5*5(五行五列)
    ax=plt.subplot(5,5,i+1)

    # 畫出子圖形
    ax.imshow(images[idx],cmap='binary')

    # 標題和label
    title="label=" +str(labels[idx])

    # 如果有傳入預測結果也顯示
    if len(prediction)>0:
      title+=",predict="+str(prediction[idx])

    # 設定子圖形的標題大小
    ax.set_title(title,fontsize=10)

    # 設定不顯示刻度
    ax.set_xticks([]);ax.set_yticks([])  
    idx+=1
  plt.show() 

plot_images_labels_prediction(x_train_image,y_train_label,[],0,10)

# 代表 train image 總共有6萬張，每一張是28*28的圖片
# label 也有6萬個
# 所以要把二維的圖片矩陣先轉換成一維
# 這裡的784是因為 28*28
x_Train=x_train_image.reshape(60000,784).astype('float32')
x_Test=x_test_image.reshape(10000,784).astype('float32')

# 由於是圖片最大的是255，所以全部除以255
x_Train_normalize=x_Train/255
x_Test_normalize=x_Test/255

#label 前處理 使用one-hot encoding
y_TrainOneHot=utils.to_categorical(y_train_label)
y_TestOneHot=utils.to_categorical(y_test_label)

# 建立模型
model = Sequential()

# 建立輸入層和隱藏層
model.add(Dense(units=256,input_dim=784,kernel_initializer='normal',activation='relu'))
# 定義隱藏層神經元個數256
# 輸入為28*28=784 個float 數字
# 使用 normal distribution 常態分布的亂數，初始化 weight權重 bias 偏差
# 定義激活函數為 relu


# 建立輸出層
model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))
# 定義輸出層為10個 (數字0~9)
# 也是使用常態分佈初始化
# 定義激活函數是 softmax
# 這裡建立的Dense 層，不用設定 input dim ，因為keras 會自動照上一層的256設定

print(model.summary())


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

#開始訓練
train_history=model.fit(x=x_Train_normalize,y=y_TrainOneHot,
            validation_split=0.2,epochs=10,batch_size=200,verbose=2)

#訓練過程畫出來
plt.title('Training Process (Accuracy)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.plot(train_history.history['accuracy'], label='accuracy')
plt.plot(train_history.history['val_accuracy'], label='val_accuracy')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

plt.title('Training Process (Loss)')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.plot(train_history.history['loss'], label='loss')
plt.plot(train_history.history['val_loss'], label='val_loss')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

#評估測試資料準確率
#scores=model.evaluate(x_Test_normalize,y_TestOneHot)
#print('accuracy',scores[1])

y_pred = model.predict(x_Test_normalize)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_TestOneHot, axis=1)

#計算 Precision、Recall 和 F1 Score
precision = precision_score(y_true, y_pred_classes, average='macro')
recall = recall_score(y_true, y_pred_classes, average='macro')
f1 = f1_score(y_true, y_pred_classes, average='macro')

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

#執行預測
prediction=model.predict(x_Test)
predicted_classes = np.argmax(prediction, axis=1)
plot_images_labels_prediction(x_test_image,y_test_label,predicted_classes,idx=340)

"""隱藏層增加為1000個神經元
model= Sequential()

model.add(Dense(units=1000,input_dim=784,kernel_initializer='normal',activation='softmax'))
model.add(Dense(units=10,kernel_initializer='normal',activation='softmax'))

print(model.summary())

# 開始訓練
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history=model.fit(x=x_Train_normalize,y=y_TrainOneHot,validation_split=0.2,epochs=10,batch_size=200,verbose=2)

show_train_history(train_history,'accuracy','val_accuracy')"""