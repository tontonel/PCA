import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Define the file path
file_path = './pd_speech_features.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path, skiprows=[0])

labels = df.iloc[:, -1]

# Drop the first column of the DataFrame
df = df.iloc[:, :-1]

# Standardize the features
features = (df - df.mean()) / df.std()

# Calculate the covariance matrix of the standardized features
covariance_matrix = np.cov(features.T)

# Calculate the eigenvectors and eigenvalues of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

eigenvalues = np.real(eigenvalues)

eigenvectors = np.real(eigenvectors)

normalized_eigenvectors = np.zeros_like(eigenvectors)
for i in range(len(eigenvectors)):
    normalized_eigenvectors[:,i] = eigenvectors[:,i] / np.linalg.norm(eigenvectors[:,i])

# Number of dimensions after dimension reduction
n_components = 200

# This is the random eigenvectors selection
selected_eigenvectors = normalized_eigenvectors[:, np.random.choice(eigenvectors.shape[1], n_components, replace=False)]

# Traditional PCA
# selected_eigenvectors = normalized_eigenvectors[:, np.argsort(-eigenvalues)]
# selected_eigenvectors = selected_eigenvectors[:, :n_components]

transformed_data = np.dot(features, selected_eigenvectors)

train_data, test_data, train_labels, test_labels = train_test_split(transformed_data, labels, test_size=0.2, random_state=42)

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(n_components,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))

loss, accuracy = model.evaluate(test_data, test_labels)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))



