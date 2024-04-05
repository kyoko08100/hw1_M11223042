import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
(train_data, train_targets),(test_data,test_targets)=boston_housing.load_data()
#check data
# print(train_data,train_targets,test_data,test_targets)

from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(train_data,train_targets, random_state=1)
#75%train/25%valid
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_valid = sc.transform(x_valid)
test_data = sc.transform(test_data)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

keras.backend.clear_session()
np.random.seed(1)
tf.random.set_seed(1)

model = Sequential()
model.add(Dense(units=100, activation='relu', input_shape=x_train.shape[1:]))
model.add(Dense(units=50,activation='relu'))
model.add(Dense(units=1))

model.summary()

weights, biases = model.layers[1].get_weights()

model.compile(loss='mse',
              optimizer=SGD(learning_rate=1e-3))
result= model.fit(x_train, y_train,
                  batch_size=32,
                  epochs=20,
                  validation_data=(x_valid, y_valid))
            
pd.DataFrame(result.history).plot()
plt.grid(True)
plt.show()

model.evaluate(test_data,test_targets)





# mean=train_data.mean(axis=0)
# train_data -= mean
# std=train_data.std(axis=0)
# train_data /= std # z-score data normalization
# test_data -= mean
# test_data /= std 

# from keras import models
# from keras import layers


# model = models.Sequential()
# model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(1))
# model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    

# model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
# test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
