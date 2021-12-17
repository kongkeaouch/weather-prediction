import pylab
import scipy.stats as stats
from pycaret.classification import *
import pickle
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from google.colab import drive
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics

drive.mount('/content/drive')
df = pd.read_csv('/content/drive/kongkea/weather.csv')
pd.set_option('display.max_columns', None)
df
df.nunique()
num_var = [feature for feature in df.columns if df[feature].dtypes != 'O']
discrete_var = [feature for feature in num_var if len(
    df[feature].unique()) <= 25]
cont_var = [feature for feature in num_var if feature not in discrete_var]
categ_var = [feature for feature in df.columns if feature not in num_var]
df[categ_var]
df.isnull().sum()
df.isnull().sum()*100/len(df)


def find_var_type(var):
    if var in discrete_var:
        print('{} discrete numberical variable'.format(var))
    elif var in cont_var:
        print('{} continuous numerical variable'.format(var))
    else:
        print('{} categorical Variable'.format(var))


find_var_type('cloud_3pm')
def RandomSampleImputation(df, feature): df[feature] = df[feature]; random_sample = df[feature].dropna().sample(df[feature].isnull(
).sum(), random_state=0); random_sample.index = df[df[feature].isnull()].index; df.loc[(df[feature].isnull(), feature)] = random_sample


RandomSampleImputation(df, 'cloud_9am')
RandomSampleImputation(df, 'cloud_3pm')
RandomSampleImputation(df, 'evaporate')
RandomSampleImputation(df, 'sun')
df.isnull().sum()*100/len(df)
find_var_type('rain_now')


def MeanImputation(df, feature): df[feature] = df[feature]; mean = df[feature].mean(
); df[feature] = df[feature].fillna(mean)


MeanImputation(df, 'pressure_3pm')
MeanImputation(df, 'pressure_9am')
MeanImputation(df, 'min_tmp')
MeanImputation(df, 'max_tmp')
MeanImputation(df, 'rainfall')
MeanImputation(df, 'wind_gustspeed')
MeanImputation(df, 'wind_speed9am')
MeanImputation(df, 'wind_speed3pm')
MeanImputation(df, 'pressure_9am')
MeanImputation(df, 'humid_9am')
MeanImputation(df, 'humid_3pm')
MeanImputation(df, 'tmp_3pm')
MeanImputation(df, 'tmp_9am')
df.isnull().sum()*100/len(df)
corrmat = df.corr(method='spearman')
plt.figure(figsize=(20, 20))
g = sns.heatmap(corrmat, annot=True)
for feature in cont_var:
    data = df.copy()
    sns.distplot(df[feature])
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.title(feature)
    plt.figure(figsize=(15, 15))
    plt.show()
for feature in cont_var:
    data = df.copy()
    sns.boxplot(data[feature])
    plt.title(feature)
    plt.figure(figsize=(15, 15))
df['rain_now'] = pd.get_dummies(df['rain_now'], drop_first=True)
df['rain_tmr'] = pd.get_dummies(df['rain_tmr'], drop_first=True)
df
for feature in categ_var:
    print(feature, df.groupby([feature])[
          'rain_tmr'].mean().sort_values(ascending=False).index)
wind_gustdir = {'NNW': 0, 'NW': 1, 'WNW': 2, 'N': 3, 'W': 4, 'WSW': 5, 'NNE': 6, 'S': 7,
                'SSW': 8, 'SW': 9, 'SSE': 10, 'NE': 11, 'SE': 12, 'ESE': 13, 'ENE': 14, 'E': 15}
wind_dir9am = {'NNW': 0, 'N': 1, 'NW': 2, 'NNE': 3, 'WNW': 4, 'W': 5, 'WSW': 6, 'SW': 7,
               'SSW': 8, 'NE': 9, 'S': 10, 'SSE': 11, 'ENE': 12, 'SE': 13, 'ESE': 14, 'E': 15}
wind_dir3pm = {'NW': 0, 'NNW': 1, 'N': 2, 'WNW': 3, 'W': 4, 'NNE': 5, 'WSW': 6, 'SSW': 7,
               'S': 8, 'SW': 9, 'SE': 10, 'NE': 11, 'SSE': 12, 'ENE': 13, 'E': 14, 'ESE': 15}
df['wind_gustdir'] = df['wind_gustdir'].map(wind_gustdir)
df['wind_dir9am'] = df['wind_dir9am'].map(wind_dir9am)
df['wind_dir3pm'] = df['wind_dir3pm'].map(wind_dir3pm)
df['wind_gustdir'] = df['wind_gustdir'].fillna(
    df['wind_gustdir'].value_counts().index[0])
df['wind_dir9am'] = df['wind_dir9am'].fillna(
    df['wind_dir9am'].value_counts().index[0])
df['wind_dir3pm'] = df['wind_dir3pm'].fillna(
    df['wind_dir3pm'].value_counts().index[0])
df.isnull().sum()*100/len(df)
df.head()
df_loc = df.groupby(['Location'])[
    'rain_tmr'].value_counts().sort_values().unstack()
df_loc.head()
df_loc[1].sort_values(ascending=False)
df_loc[1].sort_values(ascending=False).index
len(df_loc[1].sort_values(ascending=False).index)
mapped_location = {'Portland': 1, 'Cairns': 2, 'Walpole': 3, 'Dartmoor': 4, 'MountGambier': 5, 'NorfolkIsland': 6, 'Albany': 7, 'Witchcliffe': 8, 'CoffsHarbour': 9, 'Sydney': 10, 'Darwin': 11, 'MountGinini': 12, 'NorahHead': 13, 'Ballarat': 14, 'GoldCoast': 15, 'SydneyAirport': 16, 'Hobart': 17, 'Watsonia': 18, 'Newcastle': 19, 'Wollongong': 20, 'Brisbane': 21, 'Williamtown': 22, 'Launceston': 23, 'Adelaide': 24,
                   'MelbourneAirport': 25, 'Perth': 26, 'Sale': 27, 'Melbourne': 28, 'Canberra': 29, 'Albury': 30, 'Penrith': 31, 'Nuriootpa': 32, 'BadgerysCreek': 33, 'Tuggeranong': 34, 'PerthAirport': 35, 'Bendigo': 36, 'Richmond': 37, 'WaggaWagga': 38, 'Townsville': 39, 'PearceRAAF': 40, 'SalmonGums': 41, 'Moree': 42, 'Cobar': 43, 'Mildura': 44, 'Katherine': 45, 'AliceSprings': 46, 'Nhil': 47, 'Woomera': 48, 'Uluru': 49}
df['Location'] = df['Location'].map(mapped_location)
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%dT', errors='coerce')
df['Date_month'] = df['Date'].dt.month
df['Date_day'] = df['Date'].dt.day
df.head()
sns.countplot(df['rain_tmr'])
df = df.drop(['Date'], axis=1)
df.head()


def plot_curve(df, feature): plt.figure(figsize=(10, 6)); plt.subplot(1, 2, 1); df[feature].hist(); plt.subplot(
    1, 2, 2); stats.probplot(df[feature], dist='norm', plot=pylab); plt.title(feature); plt.show()


for i in cont_var:
    plot_curve(df, i)
x = df.drop(['rain_tmr'], axis=1)
y = df['rain_tmr']
scale = StandardScaler()
scale.fit(x)
X = scale.transform(x)
x.columns
X = pd.DataFrame(X, columns=x.columns)
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)
ranfor = RandomForestClassifier()
ranfor.fit(X_train, y_train)
ypred = ranfor.predict(X_test)
print(confusion_matrix(y_test, ypred))
print(accuracy_score(y_test, ypred))
print(classification_report(y_test, ypred))
metrics.plot_roc_curve(ranfor, X_test, y_test)
metrics.roc_auc_score(y_test, ypred, average=None)
gnb = GaussianNB()
gnb.fit(X_train, y_train)
ypred2 = gnb.predict(X_test)
print(confusion_matrix(y_test, ypred2))
print(accuracy_score(y_test, ypred2))
print(classification_report(y_test, ypred2))
metrics.plot_roc_curve(gnb, X_test, y_test)
metrics.roc_auc_score(y_test, ypred2, average=None)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
ypred3 = knn.predict(X_test)
print(confusion_matrix(y_test, ypred3))
print(accuracy_score(y_test, ypred3))
print(classification_report(y_test, ypred3))
metrics.plot_roc_curve(knn, X_test, y_test)
metrics.roc_auc_score(y_test, ypred3, average=None)
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
ypred4 = xgb.predict(X_test)
print(confusion_matrix(y_test, ypred4))
print(accuracy_score(y_test, ypred4))
print(classification_report(y_test, ypred4))
metrics.plot_roc_curve(xgb, X_test, y_test)
metrics.roc_auc_score(y_test, ypred4, average=None)
file = open('rain_XGBnew_model.pkl', 'wb')
pickle.dump(xgb, file)
model = pickle.load(open('rain_XGBnew_model.pkl', 'rb'))
model = setup(data=df, target='rain_tmr')
compare_models()
lightgbm = create_model('lightgbm')
pred_holdout = predict_model(lightgbm, data=X_test)
pred_holdout
