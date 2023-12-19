# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import  MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from sklearn.neighbors import KNeighborsRegressor

#read the file 
df=pd.read_csv("numerical_project/abalone.csv");

# print top 10 rows of the data 
print(df.head())

#fill the null values if there is 
print("\nthe null values :\n",df.isnull().sum())
for column in df.columns:
    value=df[column]
    if(value.isnull().sum()>0):
        column_mean=value.mean()
        value.fillna(column_mean,inplace=True)

#Encode the Sex column 
encod=LabelEncoder()
df["Sex"]=encod.fit_transform(df["Sex"]).astype("float")

#correlation matrix to display the relationship between columns in the datframe 
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

#decaler featuers and target 
X=df.drop(columns= ["Rings"])
y=df["Rings"]

#scale the data
scaler=MinMaxScaler()
X=scaler.fit_transform(X)

#Create LinearRegression model
model=LinearRegression()

#split the data into train and test 
x_train,x_test , y_train,y_test= train_test_split(X,y,test_size=.2,random_state=43)

#fit the model to the training data
model.fit(x_train,y_train)

#predict train and test 
y_pred_train= model.predict(x_train)
y_pred_test = model.predict(x_test)

#scores and mean erorrs
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

#plot the mean sqaured error 
mse_values = [mse_train, mse_test]
labels = ['Training MSE', 'Testing MSE']
plt.bar(labels, mse_values, color=['blue', 'orange'])
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE Comparison for Training and Testing Sets')
plt.show()

#plot the r2 scores 
r2_values = [r2_score1, r2_score2]
labels = ['Training R2 Score', 'Testing R2 Score']
plt.bar(labels, r2_values, color=['green', 'red'])
plt.ylabel('R2 Score')
plt.title('R2 Score Comparison for Training and Testing Sets')
plt.ylim(0, 1)  # R2 score ranges from 0 to 1
plt.show()


#knn analysis
knn_model= KNeighborsRegressor()

param_grid = {'n_neighbors': range(1, 21), 'weights': ['uniform', 'distance']}  

# Use GridSearchCV to search for the best parameter (number of neighbors)
grid_search = GridSearchCV(knn_model, param_grid, cv=10, scoring='r2')

# Fit the grid search to the data
grid_search.fit(x_train, y_train)

# Get the best parameter
best_k = grid_search.best_params_['n_neighbors']
best_w = grid_search.best_params_['weights']

# Print the best parameters
print("Best parameters: ", grid_search.best_params_)

#fit the best parameters to the model
knn_model= KNeighborsRegressor(n_neighbors=best_k, weights= best_w)
knn_model.fit(x_train, y_train)
# # Make predictions on the testing set
y_pred_train = knn_model.predict(x_train)
y_pred_test = knn_model.predict(x_test)

# Calculate accuracy
r2_score_Knn_train = r2_score(y_train, y_pred_train)
r2_score_Knn_test = r2_score(y_test, y_pred_test)
# Calculate Mean squared Error (MSE)
mse_Knn_train = mean_squared_error(y_train, y_pred_train)
mse_Knn_test = mean_squared_error(y_test, y_pred_test)

# Calculate Mean Absolute Error (MAE)
mae_Knn_train = mean_absolute_error(y_train, y_pred_train)
mae_Knn_test = mean_absolute_error(y_test, y_pred_test)

print(f"Accuracy for the TRAIN data : {r2_score_Knn_train}")
print(f"Accuracy for the TEST data : {r2_score_Knn_test}")

print(f"Mean Squared Error (MSE) FOR TRAIN DATA : {mse_Knn_train}")
print(f"Mean Squared Error (MSE) FOR TEST DATA : {mse_Knn_test}")

print(f"Mean Absolute Error (MAE) FOR TRAIN DATA: {mae_Knn_train}")
print(f"Mean Absolute Error (MAE) FOR TEST DATA: {mae_Knn_test}")
