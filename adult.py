import numpy as np
import pandas as pd
import keras
import tensorflow
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import f1_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, precision_score, recall_score, accuracy_score

keras.backend.clear_session()
np.random.seed(1)
tensorflow.random.set_seed(1)

title = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','income']
df_Train = pd.read_csv(r'adult\adult.data', names = title, header=None,encoding='unicode_escape')
df_Test = pd.read_csv(r'adult\adult.test', names = title, header=None)
raw_df_train = df_Train.copy()

# 將final weight 去除(因為那是人口普查ID，ID不能被列為Feature)以及education-num
df_Train = df_Train.drop(columns=['fnlwgt','education-num'])
df_Test = df_Test.drop(columns=['fnlwgt','education-num'])

# 删除含有缺失值的資料
df_Train.replace(" ?", pd.NA, inplace=True)
df_Train.dropna(inplace=True)
df_Test.replace(" ?", pd.NA, inplace=True)
df_Test.dropna(inplace=True)

# 將類別的相關欄位做Label Encoding
label_encoder = LabelEncoder()
df_Train['workclass'] = label_encoder.fit_transform(df_Train['workclass'])
df_Train['education'] = label_encoder.fit_transform(df_Train['education'])
df_Train['marital-status'] = label_encoder.fit_transform(df_Train['marital-status'])
df_Train['occupation'] = label_encoder.fit_transform(df_Train['occupation'])
df_Train['relationship'] = label_encoder.fit_transform(df_Train['relationship'])
df_Train['race'] = label_encoder.fit_transform(df_Train['race'])
df_Train['sex'] = label_encoder.fit_transform(df_Train['sex'])
df_Train['native-country'] = label_encoder.fit_transform(df_Train['native-country'])
df_Train['income'] = label_encoder.fit_transform(df_Train['income'])

df_Test['workclass'] = label_encoder.fit_transform(df_Test['workclass'])
df_Test['education'] = label_encoder.fit_transform(df_Test['education'])
df_Test['marital-status'] = label_encoder.fit_transform(df_Test['marital-status'])
df_Test['occupation'] = label_encoder.fit_transform(df_Test['occupation'])
df_Test['relationship'] = label_encoder.fit_transform(df_Test['relationship'])
df_Test['race'] = label_encoder.fit_transform(df_Test['race'])
df_Test['sex'] = label_encoder.fit_transform(df_Test['sex'])
df_Test['native-country'] = label_encoder.fit_transform(df_Test['native-country'])
df_Test['income'] = label_encoder.fit_transform(df_Test['income'])

"""--------------------Regression start----------------------
# 將Dataframe分為X(feature)跟Y(class)
train_set_x = df_Train.drop('hours-per-week', axis=1)
train_set_y = df_Train['hours-per-week']
test_set_x = df_Test.drop('hours-per-week', axis=1)
test_set_y = df_Test['hours-per-week']

# 標準化
standar_scaler = StandardScaler()
train_set_x = standar_scaler.fit_transform(train_set_x)
test_set_x = standar_scaler.transform(test_set_x)

# 將訓練集分20%作為驗證集
train_x, valid_x, train_y, valid_y = train_test_split(train_set_x, train_set_y, test_size=0.2, random_state=1)

# model = Sequential()
# model.add(Dense(units=128, activation='relu', input_dim=train_x.shape[1]))
# model.add(Dense(units=64, activation='relu'))
# model.add(Dense(units=32, activation='relu'))
# model.add(Dense(units=16, activation='relu'))
# model.add(Dense(units=1))

# model.compile(loss='mse', optimizer='adam')
# train = model.fit(train_x, train_y, epochs=20, batch_size=32, validation_data=(valid_x, valid_y))
# model.evaluate(test_set_x, test_set_y)
# predict_y = model.predict(test_set_x)
# # model.summary()

# # 計算MAE、MAPE、RMSE
# mae = mean_absolute_error(test_set_y, predict_y)
# print("MAE(平均絕對誤差)：", mae)

# mape = mean_absolute_percentage_error(test_set_y, predict_y)
# print("MAPE(平均絕對百分比誤差)：", mape)

# rmse = np.sqrt(mean_squared_error(test_set_y, predict_y))
# print("RMSE(均方誤差)：", rmse)
--------------------Regression end----------------------"""


"""--------------------Classification start----------------------
# 將Dataframe分為X(feature)跟Y(class) 0是<=50K,1是>50K
train_set_x = df_Train.drop('income', axis=1)
train_set_y = df_Train['income']
test_set_x = df_Test.drop('income', axis=1)
test_set_y = df_Test['income']

# 標準化
standar_scaler = StandardScaler()
train_set_x = standar_scaler.fit_transform(train_set_x)
test_set_x = standar_scaler.transform(test_set_x)

# 將訓練集分20%作為驗證集
train_x, valid_x, train_y, valid_y = train_test_split(train_set_x, train_set_y, test_size=0.2, random_state=1)

model = Sequential()
model.add(Dense(units=128, activation='relu', input_dim=train_x.shape[1]))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=32, activation='relu'))
# model.add(Dense(units=16, activation='relu'))
# model.add(Dense(units=8, activation='relu'))
# model.add(Dense(units=4, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam')
train = model.fit(train_x, train_y, epochs=25, batch_size=32, validation_data=(valid_x, valid_y))
model.evaluate(test_set_x, test_set_y)
predict_y = model.predict(test_set_x)
# model.summary()

# 將預測結果轉為二進制
for index, val in enumerate(predict_y):
    for i, v in enumerate(val):
        if v < 0.5:
            predict_y[index][i] = 0
        else:
            predict_y[index][i] = 1

# 計算績效指標
acc = accuracy_score(test_set_y, predict_y)
precision = precision_score(test_set_y, predict_y, average='binary')
recall = recall_score(test_set_y, predict_y, average='binary')
f1 = f1_score(test_set_y, predict_y, average='binary')

print("acc:", acc)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
--------------------Classification end----------------------"""

# 輸出圖表
# pd.DataFrame(train.history).plot()
# plt.grid(True)
# plt.show()
