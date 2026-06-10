# Step : 1
# Installing required Libraries

# Step : 2
# Importing required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

# Step : 3
# Loading data
df = pd.read_csv("Titanic-Dataset (1).csv")
df.head()

# Step : 4
# Understanding and cleaning of the data
df.shape
df.info()

# Data Cleaning
df.columns = df.columns.str.lower()

df.columns
df.isnull().sum()

df.isnull().sum() / len(df)
df.duplicated().sum()

# Filling null values with mean
imputer_1  = SimpleImputer(strategy="mean")
df['age'] = imputer_1.fit_transform(df[['age']])

df.shape
df.isnull().sum()

df.drop(columns="cabin" , inplace=True)
df.shape
df.dropna(inplace=True)
df.isnull().sum()

df.duplicated().sum()

# Step : 5
# Checking Spread and removing outlier(If any)

df.describe()
cols = df.select_dtypes("number").columns
cols

fig = plt.subplots(figsize=(12 , 12))
for i , col in enumerate(cols):
  plt.subplot(len(cols) ,1 , i+1)
  plt.boxplot(df[col] , vert=False)
  plt.title(col)

plt.tight_layout()

# too many outliers we need to try different approuches and models
df_removed = df

print(f"before removing outliers: {len(df_removed)}")
for col in cols:
  Q1 = df_removed[col].quantile(0.25)
  Q3 = df_removed[col].quantile(0.75)
  IQR = Q3-Q1
  df_removed = df_removed[~((df_removed[col] <(Q1-1.5 * IQR)) | (df_removed[col] > (Q3+1.5*IQR)))]

print(f"after removing outliers: {len(df_removed)}")

# Treating outliers
df.columns

cols = ['fare', 'sibsp', 'parch']

for col in cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Clip values
    if col in ['sibsp', 'parch']:
        # family cant be in negative
        lower = max(lower, 0)

    df[col] = df[col].clip(lower, upper)

    # Step : 5
# Feature Engineering
# Creating Age Group
bins = [0, 18, 30, 45, 60, float('inf')]
labels = ["under_18", "18-30", "31-45", "46-60", "60_above"]
df["age_group"] = pd.cut(df["age"], bins=bins, labels=labels)
df.head()

# Step : 6
# Feature Selection
# checking unique values of columns
df.nunique()

for col in df.select_dtypes("object").columns:
  print(f"{col}: {df[col].nunique()}")

drop_cols = []
for col in df.select_dtypes("object").columns:
  if df[col].nunique() > 3:
    drop_cols.append(col)

drop_cols

for col in drop_cols:
  print(col)

df.drop(columns=[*drop_cols ,"passengerid" , "age" ] , inplace=True)
df.head()

df['age_group'].dtype

# Only changing data type from object to category.
# Memory Efficient
#category uses less memory than object (stores values as integer codes internally).

# Better for ML preprocessing
# Encoders (like OneHotEncoder) and some models handle category dtype more cleanly.

# Faster operations
# Grouping, sorting, comparisons are faster with categorical data.

for col in df.select_dtypes("object").columns:
    df[col] = df[col].astype('category')

print(df["sex"].nunique())
print(df["embarked"].nunique())
print(df["age_group"].nunique())

encoder = ce.OneHotEncoder(cols=["embarked" ,"sex", "age_group" ] , use_cat_names=True)
df_encoded = encoder.fit_transform(df)
df_encoded.head()

df_encoded.drop(columns=["sex_male" , "embarked_S" , "age_group_under_18"] ,inplace=True)

df_encoded.head()

# Step : 7
# Feature Target Spliting

# Features
X = df_encoded.drop(columns='survived')
# Target
y = df_encoded['survived']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Step : 8
# Feature Scaling
numeric_cols = ['fare', 'sibsp', 'parch']  # numeric features only (no binary)
scaler = StandardScaler()

# fit scaler only on train
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

# transforming test using same scaler
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Step : 9
# Creating baseline model
dummy = DummyClassifier(strategy="most_frequent") #sklearn
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)

print("Baseline Accuracy:", accuracy_score(y_test, y_pred_dummy))

# Step : 10
# Building Logistic regression

# Initialize
lr = LogisticRegression(random_state=42)

# Train
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)
y_pred_lr

# Accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

# Accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_dt))

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

# Accuracy
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_rf))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "criterion":["gini", "entropy", "log_loss"]
}

grid_rf = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
grid_rf.fit(X_train, y_train)

print("Best Params:", grid_rf.best_params_)
print("Test Accuracy:", grid_rf.best_estimator_.score(X_test, y_test))

from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3
)

gb.fit(X_train, y_train)

print("Accuracy:", gb.score(X_test, y_test))

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

param_grid = {
    'n_estimators': [100, 300, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}

gb = GradientBoostingClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=gb,
    param_grid=param_grid,
    cv=5,        # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Best Params:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)
print("Test Accuracy:", grid_search.score(X_test, y_test))