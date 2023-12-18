# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import  MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

df=pd.read_csv("abalone.csv");
# df=df.sample(n=3000)
print(df.head())
for column in df.columns:
    value=df[column]
    if(value.isnull().sum()>0):
        column_mean=value.mean()
        value.fillna(column_mean,inplace=True)
encod=LabelEncoder()
df["Sex"]=encod.fit_transform(df["Sex"]).astype("float")
# df["Sex"]=encod.fit_transform(df["Sex"])
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


X=df.drop(columns= ["Rings"])
y=df["Rings"]
scaler=MinMaxScaler()
X=scaler.fit_transform(X)
model=LinearRegression()
#best .2 test size and random_state =43
x_train,x_test , y_train,y_test= train_test_split(X,y,test_size=.2,random_state=43)

model.fit(x_train,y_train)
y_pred_train= model.predict(x_train)
y_pred_test = model.predict(x_test)
r2_score1=r2_score(y_train, y_pred_train)
r2_score2=r2_score(y_test, y_pred_test)
mse_train=mean_squared_error(y_train, y_pred_train, squared=False)
mse_test=mean_squared_error(y_test, y_pred_test, squared=False)
mae_test=mean_absolute_error(y_test, y_pred_test)
print(f"Regression model accuarcy for train= {r2_score1}" )
print(f"Regression model accuarcy for test = {r2_score2}" )
print(f"Regression model mean squared error for train  = {mse_train}" )
print(f"Regression model mean squared error for test  = {mse_test}" )
print(f"Regression model mean absolute error for test  = {mae_test}" )
mse_values = [mse_train, mse_test]
labels = ['Training MSE', 'Testing MSE']

plt.bar(labels, mse_values, color=['blue', 'orange'])
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE Comparison for Training and Testing Sets')
plt.show()

r2_values = [r2_score1, r2_score2]
labels = ['Training R2 Score', 'Testing R2 Score']

plt.bar(labels, r2_values, color=['green', 'red'])
plt.ylabel('R2 Score')
plt.title('R2 Score Comparison for Training and Testing Sets')
plt.ylim(0, 1)  # R2 score ranges from 0 to 1
plt.show()
