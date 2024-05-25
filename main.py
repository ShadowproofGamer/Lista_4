import numpy as np
import pandas as pd
from sklearn.preprocessing import Normalizer, StandardScaler, LabelEncoder, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer

pd.options.mode.copy_on_write = True

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
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
X = imputer.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Data preprocessing methods
def normalize_data(X):
    normalizer = Normalizer()
    return normalizer.fit_transform(X)

def standardize_data(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def discretize_data(X):
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    return discretizer.fit_transform(X)

def pca_data(X, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

def select_features(X, y, k=2):
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new, selector

# Evaluate the impact of data preprocessing on classification performance
def evaluate_preprocessing(X_train, X_val, y_train, y_val, preprocessing_func, classifier, **kwargs):
    if preprocessing_func == select_features:
        X_train_preprocessed, selector = preprocessing_func(X_train, y_train, **kwargs)
        X_val_preprocessed = selector.transform(X_val)
    else:
        X_train_preprocessed = preprocessing_func(X_train, **kwargs)
        X_val_preprocessed = preprocessing_func(X_val, **kwargs)

    # Train and evaluate the classifier on preprocessed data
    classifier.fit(X_train_preprocessed, y_train)
    y_pred = classifier.predict(X_val_preprocessed)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
    confusion_mat = confusion_matrix(y_val, y_pred)

    print(f"Preprocessing: {preprocessing_func.__name__}")
    print(f"Classifier: {classifier.__class__.__name__}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(confusion_mat)
    print("-" * 60)

# Evaluate different preprocessing methods
preprocessing_methods = [
    normalize_data,
    standardize_data,
    discretize_data,
    pca_data
]

for preprocessing_method in preprocessing_methods:
    evaluate_preprocessing(X_train, X_val, y_train, y_val, preprocessing_method, DecisionTreeClassifier())
    evaluate_preprocessing(X_train, X_val, y_train, y_val, preprocessing_method, GaussianNB())
