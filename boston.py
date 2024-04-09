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

#清空model
keras.backend.clear_session()
np.random.seed(1)
tf.random.set_seed(1)

model = Sequential()
model.add(Dense(units=100, activation='relu', input_shape=x_train.shape[1:]))
model.add(Dense(units=50,activation='relu'))
model.add(Dense(units=1))

model.summary()

weights, biases = model.layers[1].get_weights()



# 優化ADAM、LOSS MSE、batch_size=32、epochs=20

print("優化ADAM、LOSS MSE、batch_size=32、epochs=20")

model.compile(optimizer='adam',
              loss='mse',
              metrics=[
              'MeanAbsoluteError',
              'MeanAbsolutePercentageError',
              'RootMeanSquaredError',])
            #   optimizer=SGD(learning_rate=1e-3))

result= model.fit(x_train, y_train,
                  batch_size=32,
                  epochs=20,
                  validation_data=(x_valid, y_valid))
            
pd.DataFrame(result.history).plot()
plt.grid(True)
plt.show()

model.evaluate(test_data,test_targets)

#   優化ADAM、LOSS MSE、batch_size=64、epochs=20
print("優化ADAM、LOSS MSE、batch_size=64、epochs=20")

model.compile(optimizer='adam',
              loss='mse',
              metrics=[
              'MeanAbsoluteError',
              'MeanAbsolutePercentageError',
              'RootMeanSquaredError',])
            #   optimizer=SGD(learning_rate=1e-3))

result= model.fit(x_train, y_train,
                  batch_size=64,
                  epochs=20,
                  validation_data=(x_valid, y_valid))
            
pd.DataFrame(result.history).plot()
plt.grid(True)
plt.show()

model.evaluate(test_data,test_targets)





#   優化ADAM、LOSS MSE、batch_size=64、epochs=40
print("優化ADAM、LOSS MSE、batch_size=64、epochs=40")

model.compile(optimizer='adam',
              loss='mse',
              metrics=[
              'MeanAbsoluteError',
              'MeanAbsolutePercentageError',
              'RootMeanSquaredError',])
            #   optimizer=SGD(learning_rate=1e-3))

result= model.fit(x_train, y_train,
                  batch_size=64,
                  epochs=40,
                  validation_data=(x_valid, y_valid))
            
pd.DataFrame(result.history).plot()
plt.grid(True)
plt.show()

model.evaluate(test_data,test_targets)