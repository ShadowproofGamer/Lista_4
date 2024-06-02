import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Data exploration - present basic statistical data and remarks on features and labels
print("Basic Statistical Data:")
print(data.describe(include='all'))

print("\nMissing Values:")
print(data.isnull().sum())

print("\nValue Counts for Categorical Features:")
for column in ['size', 'material', 'color', 'sleeves']:
    print(f"\n{column.capitalize()} value counts:")
    print(data[column].value_counts())

print("\nDemand value counts:")
print(data['demand'].value_counts())
print()

# Visualizing data exploration results
# Plotting count plots for categorical features

# Size distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='size', data=data, order=data['size'].value_counts().index)
plt.title('Size Distribution')
plt.xlabel('Size')
plt.ylabel('Count')
ax = plt.gca()
bars = ax.containers[0]
ax.bar_label(bars)
plt.show()

# Material distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='material', data=data, order=data['material'].value_counts().index)
plt.title('Material Distribution')
plt.xlabel('Material')
plt.ylabel('Count')
ax = plt.gca()
bars = ax.containers[0]
ax.bar_label(bars)
plt.show()

# Color distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='color', data=data, order=data['color'].value_counts().index)
plt.title('Color Distribution')
plt.xlabel('Color')
plt.ylabel('Count')
ax = plt.gca()
bars = ax.containers[0]
ax.bar_label(bars)
plt.show()

# Sleeves distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='sleeves', data=data, order=data['sleeves'].value_counts().index)
plt.title('Sleeves Distribution')
plt.xlabel('Sleeves')
plt.ylabel('Count')
ax = plt.gca()
bars = ax.containers[0]
ax.bar_label(bars)
plt.show()

# Demand distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='demand', data=data, order=data['demand'].value_counts().index)
plt.title('Demand Distribution')
plt.xlabel('Demand')
plt.ylabel('Count')
ax = plt.gca()
bars = ax.containers[0]
ax.bar_label(bars)
plt.show()

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
    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform', subsample=None)
    return discretizer.fit_transform(X)

def pca_data(X, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

def select_features(X, y, k=2):
    selector = SelectKBest(f_classif, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new, selector

def no_transform(X):
    return X

# Hyperparameters for classifiers
decision_tree_params = [
    {'criterion': 'gini', 'max_depth': None},
    {'criterion': 'entropy', 'max_depth': 10},
    {'criterion': 'gini', 'max_depth': 5}
]

naive_bayes_params = [
    {'var_smoothing': 1e-1},
    {'var_smoothing': 1e-2},
    {'var_smoothing': 1e-3}
]

# Function to plot confusion matrix as a heatmap
def plot_multiple_confusion_matrices_heatmap(confusion_matrices, titles):
    n = len(confusion_matrices)
    cols = 3
    rows = (n // cols) + (n % cols > 0)

    plt.figure(figsize=(5 * cols, 5 * rows))

    for i, (confusion_matrix, title) in enumerate(zip(confusion_matrices, titles)):
        plt.subplot(rows, cols, i + 1)
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {title}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

    plt.tight_layout()
    plt.show()

confusion_matrices = []
titles = []


# Evaluate the impact of data preprocessing on classification performance
def evaluate_preprocessing(X_train, X_val, y_train, y_val, preprocessing_func, classifier, params, **kwargs):
    if preprocessing_func == select_features:
        X_train_preprocessed, selector = preprocessing_func(X_train, y_train, **kwargs)
        X_val_preprocessed = selector.transform(X_val)
    else:
        X_train_preprocessed = preprocessing_func(X_train, **kwargs)
        X_val_preprocessed = preprocessing_func(X_val, **kwargs)

    # Train and evaluate the classifier on preprocessed data
    classifier.set_params(**params)
    classifier.fit(X_train_preprocessed, y_train)
    y_pred = classifier.predict(X_val_preprocessed)

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_val, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)
    confusion_mat = confusion_matrix(y_val, y_pred)
    if preprocessing_func == standardize_data:
        confusion_matrices.append(confusion_mat)
        titles.append(f"{classifier.__class__.__name__}\n{params}")

    print(f"Preprocessing: {preprocessing_func.__name__}")
    print(f"Classifier: {classifier.__class__.__name__}")
    print(f"Parameters: {params}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print(confusion_mat)
    print("-" * 60)

    return {
        'preprocessing': preprocessing_func.__name__,
        'classifier': classifier.__class__.__name__,
        'params': str(params),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Evaluate different preprocessing methods with different hyperparameters
preprocessing_methods = [
    normalize_data,
    standardize_data,
    discretize_data,
    no_transform,
    pca_data
]

results = []

for preprocessing_method in preprocessing_methods:
    for params in decision_tree_params:
        results.append(evaluate_preprocessing(X_train, X_val, y_train, y_val, preprocessing_method, DecisionTreeClassifier(), params))
    for params in naive_bayes_params:
        results.append(evaluate_preprocessing(X_train, X_val, y_train, y_val, preprocessing_method, GaussianNB(), params))

# Convert results to DataFrame for plotting
results_df = pd.DataFrame(results)

# Plotting classifier performance metrics
metrics = ['accuracy', 'precision', 'recall', 'f1']
classifiers = results_df['classifier'].unique()
preprocessing_methods_names = results_df['preprocessing'].unique()

for metric in metrics:
    plt.figure(figsize=(14, 8))
    for classifier in classifiers:
        data = results_df[results_df['classifier'] == classifier]
        for params in data['params'].unique():
            param_data = data[data['params'] == params]
            plt.plot(preprocessing_methods_names, param_data[metric], marker='o', label=f"{classifier}, params: {params}")

    plt.title(f'{metric.capitalize()} with Different Preprocessing Methods')
    plt.xlabel('Preprocessing Method')
    plt.ylabel(metric.capitalize())
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout(rect=[0.0, 0.0, 1, 1.0])
    # plt.legend()
    plt.show()









# Plotting specific classifier + param combo
for classifier in classifiers:
    for param in results_df[results_df['classifier'] == classifier]['params'].unique():
        # Filter results for the specific classifier and parameters
        specific_results_df = results_df[(results_df['classifier'] == classifier) & (results_df['params'] == param)]

        plt.figure(figsize=(14, 8))

        for preprocessing_method in specific_results_df['preprocessing'].unique():
            metric_values = specific_results_df[specific_results_df['preprocessing'] == preprocessing_method][metrics].values.flatten()
            plt.plot(metrics, metric_values, marker='o', label=f"{preprocessing_method}")

        plt.title(f'Performance Metrics for {classifier} with params: {param}')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.tight_layout(rect=[0.0, 0.0, 1, 1.0])
        plt.show()

plot_multiple_confusion_matrices_heatmap(confusion_matrices, titles)