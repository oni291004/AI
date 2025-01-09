import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Function to preprocess data and train model
def load_and_train_model():
    # Load the dataset
    data = pd.read_csv('C:\\Users\\otnie\\Latihan\\.vscode\\creditcard.csv\\creditcard.csv')

    # Data Preprocessing
    data['Amount'] = (data['Amount'] - data['Amount'].mean()) / data['Amount'].std()  # Standardizing 'Amount'
    data = data.drop(['Time'], axis=1)  # Dropping 'Time' column

    # Split dataset into features and target
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Handle data imbalance using undersampling
    fraud = data[data['Class'] == 1]
    non_fraud = data[data['Class'] == 0].sample(n=len(fraud), random_state=42)
    undersampled_data = pd.concat([fraud, non_fraud])
    X = undersampled_data.drop('Class', axis=1)
    y = undersampled_data['Class']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Helper functions for manual Random Forest
    class DecisionTree:
        def __init__(self, max_depth):
            self.max_depth = max_depth
            self.tree = None

        def fit(self, X, y, depth=0):
            if len(set(y)) == 1 or depth == self.max_depth or len(y) < 2:
                return np.mean(y)

            best_feature, best_threshold = self.best_split(X, y)
            if best_feature is None:
                return np.mean(y)

            left_idx = X[:, best_feature] <= best_threshold
            right_idx = X[:, best_feature] > best_threshold

            return {
                'feature': best_feature,
                'threshold': best_threshold,
                'left': self.fit(X[left_idx], y[left_idx], depth + 1),
                'right': self.fit(X[right_idx], y[right_idx], depth + 1),
            }

        def best_split(self, X, y):
            best_gain = -1
            best_feature, best_threshold = None, None

            for feature in range(X.shape[1]):
                thresholds = np.unique(X[:, feature])
                for threshold in thresholds:
                    gain = self.information_gain(X[:, feature], y, threshold)
                    if gain > best_gain:
                        best_gain, best_feature, best_threshold = gain, feature, threshold

            return best_feature, best_threshold

        def information_gain(self, feature, y, threshold):
            parent_entropy = self.entropy(y)

            left = y[feature <= threshold]
            right = y[feature > threshold]

            if len(left) == 0 or len(right) == 0:
                return 0

            n = len(y)
            child_entropy = (len(left) / n) * self.entropy(left) + (len(right) / n) * self.entropy(right)

            return parent_entropy - child_entropy

        def entropy(self, y):
            proportions = np.bincount(y) / len(y)
            return -np.sum([p * np.log2(p) for p in proportions if p > 0])

        def predict_one(self, tree, x):
            if not isinstance(tree, dict):
                return tree

            if x[tree['feature']] <= tree['threshold']:
                return self.predict_one(tree['left'], x)
            else:
                return self.predict_one(tree['right'], x)

        def predict(self, X):
            return np.array([self.predict_one(self.tree, x) for x in X])

    class RandomForest:
        def __init__(self, n_estimators, max_depth, max_features):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.max_features = max_features
            self.trees = []

        def fit(self, X, y):
            for _ in range(self.n_estimators):
                bootstrap_X, bootstrap_y = self.bootstrap_sample(X, y)
                tree = DecisionTree(max_depth=self.max_depth)
                tree.tree = tree.fit(bootstrap_X, bootstrap_y)
                self.trees.append(tree)

        def bootstrap_sample(self, X, y):
            n_samples = X.shape[0]
            indices = np.random.choice(n_samples, n_samples, replace=True)
            return X[indices], y[indices]

        def predict(self, X):
            tree_predictions = np.array([tree.predict(X) for tree in self.trees])
            return np.round(np.mean(tree_predictions, axis=0))

    # Train the Random Forest
    forest = RandomForest(n_estimators=10, max_depth=5, max_features=5)
    forest.fit(X_train.values, y_train.values)

    # Test the model
    predictions = forest.predict(X_test.values)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    return forest, accuracy, cm, precision, recall, f1

# Streamlit Interface
st.title("Credit Card Fraud Detection")

st.write("This app uses a Random Forest model to detect fraudulent credit card transactions based on the provided dataset.")

# Load and train model
forest, accuracy, cm, precision, recall, f1 = load_and_train_model()

# Input for new transaction data
st.subheader("Enter Transaction Data:")
amount = st.number_input("Amount", min_value=0.0, max_value=10000.0)
feature_data = []

# For simplicity, we assume there are 30 features in the dataset besides the 'Amount' column.
for i in range(1, 31):  # Adjust according to the actual number of features
    feature_data.append(st.number_input(f"Feature {i}", value=0.0))

# Convert inputs into a DataFrame for prediction
if st.button("Predict Fraud"):
    input_data = np.array([feature_data + [amount]])  # Adding 'Amount' to the feature list
    prediction = forest.predict(input_data)
    if prediction == 1:
        st.write("Fraudulent transaction detected!")
    else:
        st.write("Transaction is legitimate.")
