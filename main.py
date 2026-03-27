import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Import the class you just built!
from preprocessing.optimized import OptimizedPreprocessor # Change 'optimized' to whatever you named your file

print("Fetching FULL UCI Adult Income Dataset from the web...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
columns = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
    'marital-status', 'occupation', 'relationship', 'race', 'sex', 
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]
df_real = pd.read_csv(url, names=columns, na_values=" ?", skipinitialspace=True).dropna()
print(f"Dataset loaded! Total real records: {len(df_real)}")

# ==========================================
# THE INDUSTRY FIX: DIMENSIONALITY REDUCTION
# ==========================================
print("\nCompressing state space to prevent Out-Of-Memory errors...")

# 1. Compress 16 Education levels into 3 Macro-levels
df_real['education'] = df_real['education'].replace(
    ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad'], 'Low')
df_real['education'] = df_real['education'].replace(
    ['Some-college', 'Assoc-voc', 'Assoc-acdm', 'Bachelors'], 'Mid')
df_real['education'] = df_real['education'].replace(
    ['Masters', 'Prof-school', 'Doctorate'], 'High')

# 2. Compress 14 Occupations into 3 Macro-categories
white_collar = ['Exec-managerial', 'Prof-specialty', 'Sales', 'Tech-support']
blue_collar = ['Craft-repair', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Transport-moving']
df_real['occupation'] = df_real['occupation'].apply(
    lambda x: 'White-Collar' if x in white_collar else ('Blue-Collar' if x in blue_collar else 'Service/Other')
)

# 3. Keep only the highest-impact predictive features
features_to_keep = ['age', 'education', 'occupation', 'race', 'sex', 'hours-per-week', 'income']
df_clean = df_real[features_to_keep].copy()

# 4. Encode text into integers for CVXPY
encoder = LabelEncoder()
df_clean['education'] = encoder.fit_transform(df_clean['education'])
df_clean['occupation'] = encoder.fit_transform(df_clean['occupation'])

# Map binaries (Income > 50K is 1)
df_clean['income'] = df_clean['income'].apply(lambda x: 1 if '>50K' in x else 0)
df_clean['sex'] = df_clean['sex'].apply(lambda x: 1 if 'Male' in x else 0)
df_clean['race'] = df_clean['race'].apply(lambda x: 1 if 'White' in x else 0)

print("\n--- ORIGINAL BIASED INCOME RATES (Real 1994 US Census Data) ---")
print(df_clean.groupby(['race', 'sex'])['income'].mean().round(3))
print("-" * 60)

# ==========================================
# RUN THE FAIRNESS PIPELINE
# ==========================================
# e=0.05 allows a 5% margin of error between groups.
# distortion=3.0 allows the algorithm to bump features around a bit more to achieve fairness.
engine = OptimizedPreprocessor(
    prot=['race', 'sex'], 
    target='income', 
    e=0.05, 
    distortion=3.0 
)

# Bin continuous variables into 4 buckets
engine.fit(df_clean, cols=['age', 'hours-per-week'], bins=4)
df_fair = engine.transform(df_clean, cols=['age', 'hours-per-week'])

print("\n--- NEW OPTIMIZED FAIR INCOME RATES ---")
print(df_fair.groupby(['race', 'sex'])['income'].mean().round(3))

# ==========================================
# THE ML "BEFORE & AFTER" AUDIT
# ==========================================
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from afs import FairnessAuditor # Your custom auditor!

print("\n==========================================")
print(" 🚀 THE ULTIMATE 'BEFORE & AFTER' AUDIT")
print("==========================================")

def train_and_audit(data, dataset_name):
    # 1. Prepare the data
    X = data.drop(columns=['income'])
    y = data['income']
    
    # We will audit based on 'sex' for this demonstration (1=Male/Privileged, 0=Female)
    protected = data['sex'] 
    
    X_train, X_test, y_train, y_test, attr_train, attr_test = train_test_split(
        X, y, protected, test_size=0.2, random_state=42
    )
    
    # 2. Train the Model
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    # 3. Run the Fairness Engine!
    # Using X_test.values to avoid pandas warnings in NearestNeighbors
    auditor = FairnessAuditor(
        y_true=y_test,
        y_pred=y_pred,
        protected_attr=attr_test,
        priv_class=1,
        unpriv_class=0,
        X_features=X_test.values 
    )
    report = auditor.calculate_afs()
    
    # 4. Print the Scorecard
    print(f"\n>> Model trained on {dataset_name}")
    print(f"   Accuracy: {accuracy*100:.1f}%")
    print(f"   Aggregated Fairness Score (AFS): {report['Aggregated_Fairness_Score']:.1f} / 100")
    print(f"   Verdict: {report['Verdict']}")
    print(f"   - Disparate Impact (80% Rule): {report['Metrics']['DI']:.2f}")
    print(f"   - Equal Opportunity Diff:      {report['Metrics']['EOD']:.2f}")
    print(f"   - Theil Index (Individual):    {report['Metrics']['Theil_Index']:.3f}")

# Run the audit on the original biased data
train_and_audit(df_clean, "ORIGINAL BIASED DATA")

# Run the audit on your newly optimized data
train_and_audit(df_fair, "OPTIMIZED FAIR DATA")