# ML_Model_Train:
Titanic Survival Prediction (End-to-End ML Project)

A complete Machine Learning pipeline built on the Titanic dataset to predict passenger survival using multiple classification models, feature engineering, and hyperparameter tuning.

**Project Overview:**

**This project covers the full ML workflow:**

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Encoding & Scaling
- Model Training & Evaluation
- Hyperparameter Tuning

**Dataset:**
  
Titanic Dataset (Passenger survival data)
Features include:
Age, Fare, Sex, Embarked, SibSp, Parch, etc.

**Target:**
Survived (0 = No, 1 = Yes)

**Data Preprocessing:**

- Handled missing values using SimpleImputer
- Dropped unnecessary columns (Cabin, etc.)
- Removed duplicates
- Converted column names to lowercase
- Handled missing age values

**Outlier Handling:**

- Visualized outliers using boxplots
- Applied IQR method
- Clipped extreme values in:
- Fare
- SibSp
- Parch

**Feature Engineering:**

**Created new feature:**

Age Group (Under 18, 18–30, 31–45, etc.)
Converted categorical data into meaningful groups

**Feature Encoding:**

Used OneHotEncoder (Category Encoders)

**Encoded:**

- Sex
- Embarked
- Age Group
- Removed redundant dummy variables

**Feature Scaling:**

Applied StandardScaler

**Scaled numerical features:**

- Fare
- SibSp
- Parch

**Machine Learning Models**
The following models were trained and evaluated:

 - Dummy Classifier (Baseline Model)
 - Logistic Regression
 - Decision Tree Classifier
 - Random Forest Classifier
 - Gradient Boosting Classifier
 - Hyperparameter Tuning

**Random Forest:**

Used GridSearchCV

**Tuned:**

n_estimators
max_depth
min_samples_split
min_samples_leaf
criterion
Gradient Boosting:
Used GridSearchCV

**Evaluation:**

Accuracy Score used for evaluation
Model comparison performed
Best model selected using cross-validation

**Technologies Used:**

Python 
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Category Encoders

**Project Structure:**

Titanic-ML-Project/
│
├── Titanic-Dataset.csv
├── main.py / notebook.ipynb
├── README.md
└── requirements.txt

** How to Run:**

**Clone Repository:**

git clone https://github.com/your-username/Titanic-ML-Project.git

**Move to Project Folder:**

cd Titanic-ML-Project

**Create Virtual Environment**

python -m venv .venv
.venv\Scripts\activate   # Windows
Install Dependencies
pip install -r requirements.txt
5️⃣ Run Project
python main.py
📈 Key Learnings
End-to-end ML pipeline building
Handling missing data & outliers
Feature engineering techniques
Model selection strategy
Hyperparameter tuning using GridSearchCV
Practical classification problem solving
🎯 Interview Explanation Line

👉 “I built an end-to-end machine learning pipeline on the Titanic dataset including preprocessing, feature engineering, multiple model training, and hyperparameter tuning using GridSearchCV.”

⚡ Final Note

This project demonstrates a complete real-world ML workflow from raw data to optimized model, suitable for beginner to intermediate machine learning portfolios.
