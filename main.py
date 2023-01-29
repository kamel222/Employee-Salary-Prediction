import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import lightgbm as ltb
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from sklearn import linear_model
from sklearn import metrics
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
# from xgboost import XGBClassifier as xgb
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# ================================================================================== #

# Load data
data_train = pd.read_csv('train.csv')
dataframe_train = pd.DataFrame(data_train)

data_test = pd.read_csv('test.csv')
dataframe_test = pd.DataFrame(data_test)

global final_train
global final_test

module_name = ['LR','SVC','NB','KNN','DT','RF','XGBOOST']
train_time = []
test_time = []

# ================================================================================== #

def preprocessing_train(d):
    d.replace(" ?", np.nan, inplace=True)

    # drop na values
    # d = d.dropna(axis=0, how='any')

    # drop work-fnl column
    d = d.drop('work-fnl', axis=1)


    # fill na values
    d.fillna(method='ffill', inplace=True)  # ffill backfill bfill pad

    # Convert "education" to string
    d = d.astype({'education': 'string'})

    global final_train
    final_train = d


# ================================================================================== #

def preprocessing_test(d):

    d.replace(" ?", np.nan, inplace=True)

    d.rename(columns={'workclass': 'work-class'}, inplace=True)
    d.rename(columns={'fnlwgt': 'work-fnl'}, inplace=True)
    d.rename(columns={'occupation': 'position'}, inplace=True)

    # drop work-fnl column
    d = d.drop('work-fnl', axis=1)

    # fill na values
    d.fillna(method='ffill', inplace=True)  # ffill backfill bfill pad

    d = d.astype({'education': 'string'})

    global final_test
    final_test = d

# ================================================================================== #

preprocessing_train(dataframe_train)
preprocessing_test(dataframe_test)
# lens = len(final_train['work-fnl'].value_counts())
# print(lens)
# ================================================================================== #

FE_train_columns = ['work-class','education', 'marital-status','position', 'relationship','race','sex','native-country','salary']
FE_test_columns = ['work-class','education', 'marital-status', 'position','relationship','race','sex','native-country']

# ================================================================================== #

label_encoder = preprocessing.LabelEncoder()
for i in FE_train_columns:
    final_train[i] = label_encoder.fit_transform(final_train[i])


for c in FE_test_columns:
    final_test[c] = label_encoder.fit_transform(final_test[c])

# ================================================================================== #

X = final_train
Y = final_train['salary']

# ================================================================================== #

# apply Standard Scaler techniques
# K = X.columns
# ss = StandardScaler()
# X = ss.fit_transform(X)
# X = pd.DataFrame(X, columns=K)


# apply normalization techniques
# df_max_scaled = X.copy()
# for column in df_max_scaled.columns:
#     df_max_scaled[column] = df_max_scaled[column] / df_max_scaled[column].abs().max()
# X = df_max_scaled

# ================================================================================== #

X=X.drop('salary', axis=1)
mutual_info = mutual_info_classif(X, Y)
mutual_info = pd.Series(mutual_info)
mutual_info.index = X.columns
mutual_info.sort_values(ascending=False)

# mutual_info_classif plot
listtt = list(mutual_info)
names = ['A','WC' , 'E' , 'EN','MS', 'P','R','RA','S','CG' ,'CL' ,'HPW','NC']
c = ['olive', 'teal', 'darkolivegreen', 'midnightblue', 'green']
plt.bar(names, listtt,color = c)
plt.show()

sel_five_cols = SelectKBest(mutual_info_classif , k = 8)#8 9 10 13 XGB
sel_five_cols.fit(X, Y)
train = X.columns[sel_five_cols.get_support()]

X = final_train[train]
Z = final_test[train]

# ================================================================================== #

X_train, X_valid, y_train, y_valid = train_test_split(X, Y, random_state = 10 , test_size = 0.2, shuffle=True)

# ====================================      (LR)       ============================================== #

# K = 6 , 81.2%
# 80% with fill na and scaling
# Specify the norm of the penalty (penalty)
# Maximum number of iterations taken for the solvers to converge (max_iterint)
# If the option chosen is ‘ovr’, then a binary problem is fit for each label (multi_class)

logr = linear_model.LogisticRegression()

S_train_time = time.time()
logr.fit(X_train,y_train)
E_train_time = time.time()
train_time.append(round(E_train_time - S_train_time,2))

S_test_time = time.time()
prediction = logr.predict(Z)
E_test_time = time.time()
test_time.append(round((E_test_time - S_test_time),2))

df = pd.DataFrame(prediction, columns=['salary'])

print(df['salary'].value_counts())
sns.countplot(x ='salary', data = df)
plt.show()

for i in range(len(df)):
    if df.at[i,'salary'] == 1:
        df.at[i, 'salary'] = ' <=50K'
    else:
        df.at[i, 'salary'] = ' >50K'

accuracy = metrics.accuracy_score(y_valid, prediction)
accuracy_percentage = 100 * accuracy
print("Accuracy Percentage",accuracy_percentage,"%" )
# df.to_csv('LR(ALL = 0).csv')

# ================================      (SVC)       ================================================== #

# 80.56% with fill na and scaling (rbf)
# Apply Support Vector classification on the selected features
# rbf sigmoid poly linear
classifier = SVC(kernel='rbf')
# fitting x samples and y classes

S_train_time = time.time()
classifier.fit(X_train, y_train)
E_train_time = time.time()
train_time.append(round(E_train_time - S_train_time,2))

S_test_time = time.time()
prediction = classifier.predict(Z)
E_test_time = time.time()
test_time.append(round((E_test_time - S_test_time),2))

# df = pd.DataFrame(prediction, columns=['salary'])
# print(df['salary'].value_counts())
# sns.countplot(x ='salary', data = df)
# plt.show()
#
# for i in range(len(df)):
#     if df.at[i,'salary'] == 0:
#         df.at[i, 'salary'] = ' <=50K'
#     else:
#         df.at[i, 'salary'] = ' >50K'
#
#
# accuracy = metrics.accuracy_score(y_valid, prediction)
# accuracy_percentage = 100 * accuracy
# print("Accuracy Percentage",accuracy_percentage,"%" )
# # df.to_csv('SVC(ALL=0).csv')

# ==================================      (NB)       ================================================ #

# 79% with fill na and scaling
# k = 8 , 80.1
# Apply Naive Bayes on the selected features
classifier = GaussianNB(var_smoothing=1e-9 , priors=None)

S_train_time = time.time()
classifier.fit(X_train, y_train)
E_train_time = time.time()
train_time.append(round(E_train_time - S_train_time,2))

S_test_time = time.time()
prediction = classifier.predict(Z)
E_test_time = time.time()
test_time.append(round((E_test_time - S_test_time),2))

# df = pd.DataFrame(prediction, columns=['salary'])
#
# print(df['salary'].value_counts())
# sns.countplot(x ='salary', data = df)
# plt.show()
#
# for i in range(len(df)):
#     if df.at[i,'salary'] == 0:
#         df.at[i, 'salary'] = ' <=50K'
#     else:
#         df.at[i, 'salary'] = ' >50K'
#
# accuracy = metrics.accuracy_score(y_valid, prediction)
# accuracy_percentage = 100 * accuracy
# print("Accuracy Percentage",accuracy_percentage,"%" )
# # df.to_csv('NB(ALL = 0).csv')

# ===================================      (KNN)       =============================================== #

# K = 6 , 84,0%
# 82% with fill na and no scaling
classifier = KNeighborsClassifier(n_neighbors=10)

S_train_time = time.time()
classifier.fit(X_train, y_train)
E_train_time = time.time()
train_time.append(round(E_train_time - S_train_time,2))

S_test_time = time.time()
prediction = classifier.predict(Z)
E_test_time = time.time()
test_time.append(round((E_test_time - S_test_time),2))

# df = pd.DataFrame(prediction, columns=['salary'])
# print(df['salary'].value_counts())
# sns.countplot(x ='salary', data = df)
# plt.show()
# for i in range(len(df)):
#     if df.at[i,'salary'] == 0:
#         df.at[i, 'salary'] = ' <=50K'
#     else:
#         df.at[i, 'salary'] = ' >50K'
#
# accuracy = metrics.accuracy_score(y_valid, prediction)
# accuracy_percentage = 100 * accuracy
# print("Accuracy Percentage of ", accuracy_percentage, "%")
# # df.to_csv('KNN(ALL=0).csv')

# ===============================      (DT)       =================================================== #

# k = 6 , 83.3%
# 81.5% with fill na and scaling
classifier = DecisionTreeClassifier(max_depth=None)

S_train_time = time.time()
classifier.fit(X_train, y_train)
E_train_time = time.time()
train_time.append(round(E_train_time - S_train_time,2))

S_test_time = time.time()
prediction = classifier.predict(Z)
E_test_time = time.time()
test_time.append(round((E_test_time - S_test_time),2))

# df = pd.DataFrame(prediction, columns=['salary'])
# print(df['salary'].value_counts())
# sns.countplot(x ='salary', data = df)
# plt.show()
#
# for i in range(len(df)):
#     if df.at[i,'salary'] == 0:
#         df.at[i, 'salary'] = ' <=50K'
#     else:
#         df.at[i, 'salary'] = ' >50K'
#
# accuracy = metrics.accuracy_score(y_valid, prediction)
# accuracy_percentage = 100 * accuracy
# print("Accuracy Percentage = ",accuracy_percentage,"%" )
#
# # df.to_csv('DT(None).csv')

# ===============================      (RF)       =================================================== #

# K=14 , 84.7%
# 85% with fill na and scaling and no FS
clf_4 = RandomForestClassifier(n_estimators=100,max_depth=None)

S_train_time = time.time()
clf_4.fit(X_train, y_train)
E_train_time = time.time()
train_time.append(round(E_train_time - S_train_time,2))

S_test_time = time.time()
prediction = clf_4.predict(Z)
E_test_time = time.time()
test_time.append(round((E_test_time - S_test_time),2))

# df = pd.DataFrame(prediction, columns=['salary'])
# print(df['salary'].value_counts())
# sns.countplot(x ='salary', data = df)
# plt.show()
#
# for i in range(len(df)):
#     if df.at[i,'salary'] == 0:
#         df.at[i, 'salary'] = ' <=50K'
#     else:
#         df.at[i, 'salary'] = ' >50K'
#
# accuracy = metrics.accuracy_score(prediction,y_valid)
# accuracy_percentage = 100 * accuracy
# print("Accuracy Percentage = ",accuracy_percentage,"%" )
# # df.to_csv('RF(Yasser,K=7,RS=11).csv')

# ===============================      (XGBOOST)       =================================================== #

# approx, hist and gpu_hist
# k = 14 , 87.0%
# The tree construction algorithm used in XGBoost (tree_method)
# Minimum loss reduction required to make a further partition on a leaf node of the tree

clf= xgb.XGBClassifier(tree_method="hist",enable_categorical=True,gamma =0)

S_train_time = time.time()
clf.fit(X_train, y_train)
E_train_time = time.time()
train_time.append(round(E_train_time - S_train_time,2))

S_test_time = time.time()
prediction = clf.predict(Z)
E_test_time = time.time()
test_time.append(round((E_test_time - S_test_time),2))

# df = pd.DataFrame(prediction, columns=['salary'])
# print(df['salary'].value_counts())
# sns.countplot(x ='salary', data = df)
# plt.show()
#
# for i in range(len(df)):
#     if df.at[i,'salary'] == 0:
#         df.at[i, 'salary'] = ' <=50K'
#     else:
#         df.at[i, 'salary'] = ' >50K'
#
# accuracy = metrics.accuracy_score(y_valid, prediction)
# accuracy_percentage = 100 * accuracy
# print("Accuracy Percentage = ",accuracy_percentage,"%" )
# # df.to_csv('XGB(kamel,dropna,ffil(test),K=13,RS=8).csv')

# ===============================      (CATBoost)       =================================================== #

# # K = 14 , 87.2
# model = CatBoostClassifier(iterations=1000,task_type="GPU",devices='0:1')
# model.fit(X_train, y_train,verbose=False)
# prediction = model.predict(Z)
#
#
# df = pd.DataFrame(prediction, columns=['salary'])
# print(df['salary'].value_counts())
# sns.countplot(x ='salary', data = df)
# plt.show()
#
# for i in range(len(df)):
#     if df.at[i,'salary'] == 0:
#         df.at[i, 'salary'] = ' <=50K'
#     else:
#         df.at[i, 'salary'] = ' >50K'
#
# # accuracy = metrics.accuracy_score(y_valid, prediction)
# # accuracy_percentage = 100 * accuracy
# # print("Accuracy Percentage = ",accuracy_percentage,"%" )
# # df.to_csv('new catboost(RS=10,dropNA).csv')

# ===============================      (LGBMClassifier)       =================================================== #

# # LGBMClassifier #
#
# # 89.2
# # Z['year_production']= Z['year_production'].astype(str).astype(int)
# # Z['category']= Z['category'].astype(str).astype(int)
# # X_train['year_production']= X_train['year_production'].astype(str).astype(int)
# # X_train['category']= X_train['category'].astype(str).astype(int)
#
# model = ltb.LGBMClassifier()
#
# model.fit(X_train, y_train)
#
# prediction = model.predict(Z)
#
#
# df = pd.DataFrame(prediction, columns=['salary'])
# print(df['salary'].value_counts())
# sns.countplot(x ='salary', data = df)
# plt.show()
#
# for i in range(len(df)):
#     if df.at[i,'salary'] == 0:
#         df.at[i, 'salary'] = ' <=50K'
#     else:
#         df.at[i, 'salary'] = ' >50K'
#
# # accuracy = metrics.accuracy_score(y_valid, prediction)
# # accuracy_percentage = 100 * accuracy
# # print("Accuracy Percentage = ",accuracy_percentage,"%" )
#
# # df.to_csv('LGBMClassifier.csv')

# ===============================      bar plot    =================================================== #

x = np.arange(len(module_name))
width = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, train_time, width, label='train_time')
rects2 = ax.bar(x + width/2, test_time, width, label='test_time')

ax.set_ylabel('Rating')
ax.set_title('Scores of Train and Test')
ax.set_xticks(x, module_name)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
fig.tight_layout()
plt.show()
