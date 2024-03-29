from keras.datasets import boston_housing
(train_data, train_targets),(test_data,test_targets)=boston_housing.load_data()

mean=train_data.mean(axis=0)
train_data -= mean
std=train_data.std(axis=0)
train_data /= std # z-score data normalization
test_data -= mean
test_data /= std 

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

models.fit(train_data, train_targets, epoch=80, batch_size=16, verbose=0)
test_mse_score, teset_mae_score = model.evaluate(test_data, test_tragets)
