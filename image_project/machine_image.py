import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, silhouette_score
from sklearn.decomposition import PCA
from skimage.feature import hog
from skimage import exposure
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.preprocessing import image
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

# Set the path to your dataset
data_path = "cell_images"

# Function to load and preprocess image data
def load_and_preprocess_images(dataset_dir, img_size=(64, 64)):
    images = []
    labels = []
    
    classes = os.listdir(dataset_dir)
    class_encoder = LabelEncoder()
    
    for class_name in classes:
        class_path = os.path.join(dataset_dir, class_name)
        class_images = os.listdir(class_path)
        
        for image_name in class_images:
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            
            # Resize image
            image = cv2.resize(image, img_size)
            
            # Convert to grayscale
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Normalize pixel values
            image_normalized = image_gray / 255.0
            
            # Extract HOG features
            hog_features = extract_hog_features(image_gray)
            
            images.append(hog_features)
            labels.append(class_name)
    
    # Encode labels
    labels_encoded = class_encoder.fit_transform(labels)
    
    return np.array(images), labels_encoded, class_encoder


# Function to extract HOG features
def extract_hog_features(image):
    features, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
    return features

# Function for data preprocessing
def preprocess_data(images, labels, random_state=42):
    # Shuffle the data
    images, labels = shuffle(images, labels, random_state=random_state)


# Load and preprocess a subset of images
X, y, class_encoder = load_and_preprocess_images(data_path)


# Reshape HOG features to 1D array
X = X.reshape(X.shape[0], -1)

# Use a subset of your data for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression Model
logreg = LogisticRegression()
losses = []

# Training loop
for i in range(1, 100):  # Adjust the number of iterations as needed
    logreg.fit(X_train_scaled, y_train)
    # Track the negative log-likelihood
    losses.append(-logreg.score(X_train_scaled, y_train))


# Predictions on the test set
y_pred_logreg = logreg.predict(X_test_scaled)


# Accuracy
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
print(f"Logistic Regression Accuracy: {accuracy_logreg * 100:.2f}%")

# Confusion Matrix
conf_matrix_logreg = confusion_matrix(y_test, y_pred_logreg)
print("Confusion Matrix for Logistic Regression:")
print(conf_matrix_logreg)


# Plot ROC Curve for Logistic Regression
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, logreg.predict_proba(X_test)[:, 1])
roc_auc_logreg = auc(fpr_logreg, tpr_logreg)

plt.figure(figsize=(8, 8))
plt.plot(fpr_logreg, tpr_logreg, color='darkorange', lw=2, label=f'Logistic Regression (AUC = {roc_auc_logreg:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Logistic Regression')
plt.legend(loc="lower right")
plt.show() 


# Plotting the loss curve
plt.plot(np.arange(1, len(losses) + 1), losses, label='Logistic Regression Loss')
plt.xlabel('Iteration')
plt.ylabel('Negative Log-Likelihood')
plt.title('Loss Curve for Logistic Regression')
plt.legend()
plt.show()


# Apply PCA for dimensionality reduction
num_components = 128
pca = PCA(n_components=num_components)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)


# Apply KMeans clustering with silhouette analysis
silhouette_scores = []
num_clusters_range = range(2, 11)  # Choose a range of clusters to try

for num_clusters in num_clusters_range:
    kmeans = KMeans(n_clusters=num_clusters, random_state=1111)
    cluster_labels = kmeans.fit_predict(X_train_pca)
    silhouette_avg = silhouette_score(X_train_pca, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Find the optimal number of clusters based on silhouette score
optimal_num_clusters = num_clusters_range[np.argmax(silhouette_scores)]
print(f"Optimal Number of Clusters: {optimal_num_clusters}")

# Train KMeans with the optimal number of clusters
kmeans_optimal = KMeans(n_clusters=optimal_num_clusters, random_state=1111)
train_cluster_labels = kmeans_optimal.fit_predict(X_train_pca)


# Train logistic regression on cluster assignments
logreg_on_clusters = LogisticRegression()
logreg_on_clusters.fit(X_train_pca, train_cluster_labels)


# Predict using the trained model on the test set
test_cluster_labels = kmeans_optimal.predict(X_test_pca)
kmeans_logreg_predictions = logreg_on_clusters.predict(X_test_pca)


# Evaluate accuracy
kmeans_logreg_accuracy = accuracy_score(y_test, kmeans_logreg_predictions)
print(f"K-Means with Logistic Regression Accuracy: {kmeans_logreg_accuracy}")


# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, kmeans_logreg_predictions)
# Print confusion matrix
print("Confusion Matrix:")
print(conf_matrix)

# Assuming binary labels in y_test (0 or 1)
fpr, tpr, _ = roc_curve(y_test, kmeans_logreg_predictions)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# Plot the loss curve
inertia_values = []
for num_clusters in num_clusters_range:
    kmeans = KMeans(n_clusters=num_clusters, random_state=1111)
    kmeans.fit(X_train_pca)
    inertia_values.append(kmeans.inertia_)

plt.plot(num_clusters_range, inertia_values, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (Loss)')
plt.title('K-Means Loss Curve')
plt.show()

