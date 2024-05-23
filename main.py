import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd


import pandas as pd
from sklearn.preprocessing import Normalizer, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer

# Load the data from the CSV file
data = pd.read_csv('t-shirts.csv')

# Separate features and target variable
X = data[['size', 'material', 'color', 'sleeves']]
y = data['demand']

# Encode categorical features
label_encoders = {}
for column in ['size', 'material', 'color', 'sleeves']:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

# Handle missing values
X_imputed = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Data preprocessing methods
def normalize_data(X):
    normalizer = Normalizer()
    return normalizer.fit_transform(X)

def standardize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

# Evaluate the impact of data preprocessing on classification performance
def evaluate_preprocessing(X_train, X_val, y_train, y_val, preprocessing_func, classifier):
    # Apply preprocessing to training and validation data
    X_train_preprocessed = preprocessing_func(X_train)
    X_val_preprocessed = preprocessing_func(X_val)

    # Train and evaluate the classifier on preprocessed data
    classifier.fit(X_train_preprocessed, y_train)
    y_pred = classifier.predict(X_val_preprocessed)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='micro')
    recall = recall_score(y_val, y_pred, average='micro')
    confusion_mat = confusion_matrix(y_val, y_pred)

    print(f"Preprocessing: {preprocessing_func.__name__}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(confusion_mat)

# Evaluate different preprocessing methods
preprocessing_methods = [normalize_data, standardize_data]

for preprocessing_method in preprocessing_methods:
    evaluate_preprocessing(X_train, X_val, y_train, y_val, preprocessing_method, DecisionTreeClassifier())
    evaluate_preprocessing(X_train, X_val, y_train, y_val, preprocessing_method, GaussianNB())

# Evaluate classification with different classifiers and hyperparameters
classifiers = [DecisionTreeClassifier(), GaussianNB()]

for classifier in classifiers:
    hyperparameter_sets = []
    if classifier is DecisionTreeClassifier:
        # Evaluate different hyperparameter combinations
        hyperparameter_sets = [
            {'criterion': 'entropy', 'max_depth': 10},
            {'criterion': 'gini', 'max_depth': 5},
            {'criterion': 'entropy', 'max_depth': 15, 'min_samples_split': 20}
        ]
    # else:
    #     # Evaluate different hyperparameter combinations
    #     hyperparameter_sets = [
    #         {'criterion': 'entropy', 'max_depth': 10},
    #         {'criterion': 'gini', 'max_depth': 5},
    #         {'criterion': 'entropy', 'max_depth': 15, 'min_samples_split': 20}
    #     ]

    for hyperparameter_set in hyperparameter_sets:
        classifier.set_params(**hyperparameter_set)

        # Train and evaluate the classifier on imputed data
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='micro')
        recall = recall_score(y_val, y_pred, average='micro')
        confusion_mat = confusion_matrix(y_val, y_pred)

        print(f"Classifier: {type(classifier)}")
        print(f"Hyperparameters: {hyperparameter_set}")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall: {recall:.3f}")
        print(confusion_mat)
        print("-" * 60)  # Print a separator between evaluations
