import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

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

# 將Dataframe分為X(feature)跟Y(class) 0是<=50K,1是>50K
train_set_x = df_Train.drop('hours-per-week', axis=1)
train_set_y = df_Train['hours-per-week']
test_set_x = df_Test.drop('hours-per-week', axis=1)
test_set_y = df_Test['hours-per-week']

# 標準化
standar_scaler = StandardScaler()
train_set_x = standar_scaler.fit_transform(train_set_x)
test_set_x = standar_scaler.transform(test_set_x)

# 將訓練集分一部份作為驗證集
train_x, valid_x, train_y, valid_y = train_test_split(train_set_x, train_set_y, random_state=1)

# print(df_Train)
# print(df_Test)
print(len(train_x))
print(len(valid_x))