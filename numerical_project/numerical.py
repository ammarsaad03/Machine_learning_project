import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import  MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV,learning_curve

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,confusion_matrix
from sklearn.neighbors import KNeighborsRegressor

def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='neg_mean_squared_error', train_sizes=np.linspace(0.1, 1.0, 10))

    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, label='Training Error')
    plt.plot(train_sizes, test_scores_mean, label='Testing Error')

    plt.title(title)
    plt.xlabel('Training Set Size')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.show()

#read the file 
df=pd.read_csv(r"D:\ammar college\Level 3\semester1\Machine learning\archive\abalone.csv");

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

print("Featuers names:\n",X.columns)

#scale the data
scaler=MinMaxScaler()
X=scaler.fit_transform(X)

#Create LinearRegression model
model=LinearRegression()

#split the data into train and test 
x_train,x_test , y_train,y_test= train_test_split(X,y,test_size=.2,random_state=43)

#fit the model to the training data
model.fit(x_train,y_train)

plot_learning_curve(model,X,y,'Linear Regression Learning Curve')
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

# Define threshold values for classification
threshold_low = 6
threshold_medium = 12
threshold_high = 18

# Convert regression predictions to binary classes based on the thresholds
y_pred_class = np.where(y_pred_test < threshold_low, '0-6', np.where(y_pred_test < threshold_medium, '7-11', np.where(y_pred_test < threshold_high, '12-16', '+16')))

# Convert actual values to binary classes based on the thresholds
y_test_class = np.where(y_test < threshold_low, '0-6', np.where(y_test < threshold_medium, '7-11', np.where(y_test < threshold_high, '12-16', '+16')))

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test_class, y_pred_class)

#Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0-6', '7-11', '12-16','+16'], yticklabels=['0-6', '7-11', '12-16','+16'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for linear Regression')
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
plot_learning_curve(knn_model,X,y,'KNN Regression Learning Curve')
# # Make predictions on the testing set
y_pred_train = knn_model.predict(x_train)
y_knn_pred_test = knn_model.predict(x_test)

# Calculate accuracy
r2_score_Knn_train = r2_score(y_train, y_pred_train)
r2_score_Knn_test = r2_score(y_test, y_knn_pred_test)
# Calculate Mean squared Error (MSE)
mse_Knn_train = mean_squared_error(y_train, y_pred_train)
mse_Knn_test = mean_squared_error(y_test, y_knn_pred_test)

# Calculate Mean Absolute Error (MAE)
mae_Knn_train = mean_absolute_error(y_train, y_pred_train)
mae_Knn_test = mean_absolute_error(y_test, y_knn_pred_test)

print(f"Accuracy for the TRAIN data : {r2_score_Knn_train}")
print(f"Accuracy for the TEST data : {r2_score_Knn_test}")

print(f"Mean Squared Error (MSE) FOR TRAIN DATA : {mse_Knn_train}")
print(f"Mean Squared Error (MSE) FOR TEST DATA : {mse_Knn_test}")

print(f"Mean Absolute Error (MAE) FOR TRAIN DATA: {mae_Knn_train}")
print(f"Mean Absolute Error (MAE) FOR TEST DATA: {mae_Knn_test}")

plt.scatter(y_test, y_knn_pred_test, color='blue', label='Actual vs Predicted')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle='--', color='red', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values - KNN Regression')
plt.legend()
plt.show()

# Define threshold values for classification
threshold_low = 6
threshold_medium = 12
threshold_high = 18

# Convert regression predictions to binary classes based on the thresholds
y_pred_class = np.where(y_knn_pred_test < threshold_low, '0-6', np.where(y_knn_pred_test < threshold_medium, '7-11', np.where(y_knn_pred_test < threshold_high, '12-16', '+16')))

# Convert actual values to binary classes based on the thresholds
y_test_class = np.where(y_test < threshold_low, '0-6', np.where(y_test < threshold_medium, '7-11', np.where(y_test < threshold_high, '12-16', '+16')))

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test_class, y_pred_class)

#Display the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['0-6', '7-11', '12-16','+16'], yticklabels=['0-6', '7-11', '12-16','+16'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix For KNN')
plt.show()
