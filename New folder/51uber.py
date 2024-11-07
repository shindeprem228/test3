
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


data = pd.read_csv('uber.csv')


print("Dataset Head:\n", data.head())
print("\nDataset Tail:\n", data.tail())
print("\nDataset Info:\n")
data.info()
print("\nDataset Description:\n", data.describe())
print("\nDataset Shape:", data.shape)
print("\nDataset Columns:", data.columns)


data = data.drop(['Unnamed: 0', 'key'], axis=1)
print("\nShape after dropping unnecessary columns:", data.shape)


data['month'] = data['pickup_datetime'].str.slice(start=5, stop=7)
data['hour'] = data['pickup_datetime'].str.slice(start=11, stop=13)
data = data.drop(['pickup_datetime'], axis=1)
print("\nDataset after extracting month and hour:\n", data.head())


data["dropoff_longitude"] = data["dropoff_longitude"].fillna(data['dropoff_longitude'].mean())
data["dropoff_latitude"] = data["dropoff_latitude"].fillna(data['dropoff_latitude'].mean())


def haversine(lon1, lon2, lat1, lat2):
    lon1, lon2, lat1, lat2 = map(np.radians, [lon1, lon2, lat1, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    return 2 * 6371 * np.arcsin(np.sqrt(a))

data['distance'] = haversine(data['pickup_longitude'], data['dropoff_longitude'],
                             data['pickup_latitude'], data['dropoff_latitude'])


data.replace(to_replace=0, value=data['passenger_count'].mean(), inplace=True)
data.replace(to_replace=0, value=data['distance'].mean(), inplace=True)
data.loc[data['fare_amount'] <= 0, 'fare_amount'] = data['fare_amount'].mean()


data.plot(kind="box", subplots=True, layout=(6, 2), figsize=(15, 20), title="Boxplots for Outlier Detection")
plt.show()


def remove_outlier(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower_bound, upper_bound)
    return df

for column in ['fare_amount', 'pickup_longitude', 'pickup_latitude', 
               'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'distance']:
    data = remove_outlier(data, column)


data.plot(kind="box", subplots=True, layout=(6, 2), figsize=(15, 20), title="Boxplots After Outlier Treatment")
plt.show()


print("\nMissing Values:\n", data.isnull().sum())


corr_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


X = data.drop(columns=['fare_amount'])
y = data['fare_amount']
print("\nFeature and Target Shapes:", X.shape, y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
print("Train and Test Split Shapes:", X_train.shape, y_train.shape, X_test.shape, y_test.shape)


linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
y_pred_lr = linear_regression.predict(X_test)


print("\nLinear Regression Performance:")
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred_lr))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred_lr))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred_lr)))
print("R Squared (R²):", metrics.r2_score(y_test, y_pred_lr))


random_forest = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)
y_pred_rf = random_forest.predict(X_test)


print("\nRandom Forest Regression Performance:")
print("Mean Absolute Error:", metrics.mean_absolute_error(y_test, y_pred_rf))
print("Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred_rf))
print("Root Mean Squared Error:", np.sqrt(metrics.mean_squared_error(y_test, y_pred_rf)))
print("R Squared (R²):", metrics.r2_score(y_test, y_pred_rf))


results = pd.DataFrame({'Actual': y_test, 'Linear Predicted': y_pred_lr, 'RF Predicted': y_pred_rf})
print("\nSample Predictions:\n", results.sample(10))
