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

# print("\nFeature Correlations:")
# print(data.corr())

print("\nValue Counts for Categorical Features:")
for column in ['size', 'material', 'color', 'sleeves']:
    print(f"\n{column.capitalize()} value counts:")
    print(data[column].value_counts())

print("\nDemand value counts:")
print(data['demand'].value_counts())
print()

# Observations:
# - Displaying summary statistics for numerical and categorical features
# - Checking for missing values in the dataset
# - Examining the correlation matrix to understand relationships between numerical features
# - Counting unique values for categorical features to understand their distributions

# # Visualizing data exploration results
# plt.figure(figsize=(18, 12))
#
# # Plotting count plots for categorical features
# plt.subplot(2, 3, 1)
# sns.countplot(x='size', data=data, order=data['size'].value_counts().index)
# plt.title('Size Distribution')
#
# plt.subplot(2, 3, 2)
# sns.countplot(x='material', data=data, order=data['material'].value_counts().index)
# plt.title('Material Distribution')
#
# plt.subplot(2, 3, 3)
# sns.countplot(x='color', data=data, order=data['color'].value_counts().index)
# plt.title('Color Distribution')
#
# plt.subplot(2, 3, 4)
# sns.countplot(x='sleeves', data=data, order=data['sleeves'].value_counts().index)
# plt.title('Sleeves Distribution')
#
# plt.subplot(2, 3, 5)
# sns.countplot(x='demand', data=data, order=data['demand'].value_counts().index)
# plt.title('Demand Distribution')
#
# plt.tight_layout()
# plt.show()


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

    return {
        'preprocessing': preprocessing_func.__name__,
        'classifier': classifier.__class__.__name__,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# Evaluate different preprocessing methods
preprocessing_methods = [
    normalize_data,
    standardize_data,
    discretize_data,
    no_transform,
    pca_data

]

results = []

for preprocessing_method in preprocessing_methods:
    results.append(evaluate_preprocessing(X_train, X_val, y_train, y_val, preprocessing_method, DecisionTreeClassifier()))
    results.append(evaluate_preprocessing(X_train, X_val, y_train, y_val, preprocessing_method, GaussianNB()))

# Convert results to DataFrame for plotting
results_df = pd.DataFrame(results)

# Plotting
metrics = ['accuracy', 'precision', 'recall', 'f1']
classifiers = results_df['classifier'].unique()
preprocessing_methods_names = results_df['preprocessing'].unique()

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Classifier Performance with Different Preprocessing Methods')

for i, metric in enumerate(metrics):
    ax = axs[i // 2, i % 2]
    for classifier in classifiers:
        data = results_df[results_df['classifier'] == classifier]
        ax.plot(preprocessing_methods_names, data[metric], label=classifier)

    ax.set_title(metric.capitalize())
    ax.set_xlabel('Preprocessing Method')
    ax.set_ylabel(metric.capitalize())
    ax.legend()

plt.tight_layout()
plt.show()
