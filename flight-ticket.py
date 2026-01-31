import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline

# --- STEP 1: DATA READING ---
df = pd.read_csv("archive/Clean_Dataset.csv")
df = df.drop('Unnamed: 0', axis=1) # Remove redundant column

# --- STEP 2: DATA PROCESSING ---
# Check popularity of airlines
df1 = df.groupby(['flight', 'airline'], as_index=False).count()
print(df1.airline.value_counts())

# Print unique values for each categorical column
for column in df.columns:
    if column not in ['duration', 'price', 'days_left', 'flight']:
        print(f"{column}: \n{df[column].unique()}\n")

# --- STEP 3: DATA ANALYSIS & VISUALIZATION ---
sns.set_theme(style="white")

# 3.1 Price difference between airlines
plt.figure(figsize=(10, 8))
ax = sns.barplot(data=df, x='airline', y='price', hue='class', palette='hls')
plt.title('PRICE DIFFERENCE BETWEEN AIRLINES')
plt.show()

# 3.2 Number of flights per airline
plt.figure(figsize=(10, 8))
ax = sns.countplot(x=df['airline'], palette='hls', hue=df["class"])
plt.title('Flight Statistics by Airline')
plt.show()

# 3.3 Price dependency on days left before departure
df_temp = df.groupby(['days_left'])['price'].mean().reset_index()
plt.figure(figsize=(15, 6))
ax = plt.axes()
sns.regplot(x=df_temp.loc[df_temp["days_left"]==1].days_left, y=df_temp.loc[df_temp["days_left"]==1].price, fit_reg=False, ax=ax)
sns.regplot(x=df_temp.loc[(df_temp["days_left"]>1)&(df_temp["days_left"]<20)].days_left, y=df_temp.loc[(df_temp["days_left"]>1)&(df_temp["days_left"]<20)].price, fit_reg=True, ax=ax)
sns.regplot(x=df_temp.loc[df_temp["days_left"]>=20].days_left, y=df_temp.loc[df_temp["days_left"]>=20].price, fit_reg=True, ax=ax)
plt.show()

# 3.4 Price based on Departure and Arrival time
departure_time_price = df.groupby('departure_time')['price'].mean().round(0).sort_values(ascending=True)
arrival_time_price = df.groupby('arrival_time')['price'].mean().round(0).sort_values(ascending=True)

plt.figure(figsize=(16, 13))
plt.subplot(1, 2, 1)
plt.bar(departure_time_price.index, departure_time_price.values)
plt.subplot(1, 2, 2)
plt.bar(arrival_time_price.index, arrival_time_price.values)
plt.show()

# --- STEP 4: MODEL BUILDING ---
# 4.1 Pre-processing
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

X = df.drop('price', axis=1).values
y = df['price'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.7, random_state=42)

mmscaler = MinMaxScaler()
X_train = mmscaler.fit_transform(X_train)
X_test = mmscaler.transform(X_test)

# 4.2 Decision Tree
modeldcr = DecisionTreeRegressor()
modeldcr.fit(X_train, Y_train)
y_pred_dt = modeldcr.predict(X_test)
print(f"Decision Tree R2: {r2_score(Y_test, y_pred_dt)}")

# 4.3 Random Forest
modelrfr = RandomForestRegressor()
modelrfr.fit(X_train, Y_train)
y_pred_rf = modelrfr.predict(X_test)
print(f"Random Forest R2: {r2_score(Y_test, y_pred_rf)}")

# 4.4 Linear Regression
modellr = LinearRegression()
modellr.fit(X_train, Y_train)
y_pred_lr = modellr.predict(X_test)
print(f"Linear Regression R2: {r2_score(Y_test, y_pred_lr)}")
