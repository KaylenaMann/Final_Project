#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegressionCV
from Clean_data import clean_data
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report


# In[3]:


#Creating paths so anyone can run the code
script_dir = Path(__file__).parent
model_dir = script_dir / "model"
data_dir = script_dir.parent / "Raw Data"
output_dir = script_dir.parent / "Output"

model_dir.mkdir(exist_ok=True)
output_dir.mkdir(exist_ok=True)

data = pd.read_csv(data_dir / "EducationData.csv")

#implementing walk forward style validation 
years = sorted(data["year"].unique())
train_years = years[:-1]  
test_year = years[-1]     

train_data = data[data["year"].isin(train_years)].copy()
test_data = data[data["year"] == test_year].copy()

#Cleaning Train data

X_train, y_train, cutoff, training_columns, continuous_features = clean_data(
    train_data, 
    cutoff=None,
    training_columns=None,
    is_training=True
)

print(f"\nTotal features: {len(training_columns)}")
print(f"Continuous features to scale: {len(continuous_features)}")
print(f"Categorical features (dummies): {len(training_columns) - len(continuous_features)}")

#Cleaning Test data
X_test_raw = clean_data(
    test_data,  
    cutoff=cutoff,
    training_columns=training_columns,
    is_training=False
)

test_data["math_test_pct_prof_midpt"] = pd.to_numeric(
    test_data["math_test_pct_prof_midpt"], errors="coerce"
)
valid_idx = test_data["math_test_pct_prof_midpt"].notna()
y_test = (test_data.loc[valid_idx, "math_test_pct_prof_midpt"] <= cutoff).astype(int)
X_test = X_test_raw.loc[valid_idx]

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"Cutoff: {cutoff:.2f}%")

# Scale features
categorical_features = [col for col in training_columns if col not in continuous_features]
scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[continuous_features] = scaler.fit_transform(X_train[continuous_features])
X_test_scaled[continuous_features] = scaler.transform(X_test[continuous_features])

#Run Final Model
model = LogisticRegressionCV(
    Cs=np.logspace(-4, 4, 20), 
    cv=5,
    penalty="l2",
    solver="lbfgs",
    class_weight="balanced",
    scoring="f1",
    max_iter=5000,
    random_state=33,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

y_train_pred = model.predict(X_train_scaled)
y_train_prob = model.predict_proba(X_train_scaled)[:, 1]

#printing training performance
print("\n" + "="*70)
print("TRAINING SET PERFORMANCE (2016-2017)")
print("="*70)
print(f"Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_train, y_train_prob):.4f}")
print(f"F1 Score: {f1_score(y_train, y_train_pred):.4f}")

y_test_pred = model.predict(X_test_scaled)
y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

#printing test performance
print("\n" + "="*70)
print("TEST SET PERFORMANCE (2018)")
print("="*70)
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_test_prob):.4f}")
print(f"F1 Score: {f1_score(y_test, y_test_pred):.4f}")

print(f"\nClassification Report (2018):")
print(classification_report(y_test, y_test_pred, 
                          target_names=['Not Bottom Quartile', 'Bottom Quartile']))

joblib.dump(model, model_dir / "final_model.pkl")
joblib.dump(scaler, model_dir / "scaler.pkl")
joblib.dump(training_columns, model_dir / "training_columns.pkl")
joblib.dump(cutoff, model_dir / "cutoff.pkl")
joblib.dump(continuous_features, model_dir / "continuous_features.pkl")

print("\n" + "="*70)
print("✓ MODEL SAVED")
print("="*70)
print(f"Trained on: {train_years} (years)")
print(f"Validated on: {test_year} (year)")
print("This simulates real early-warning: using 2016-2017 patterns to predict 2018 outcomes")
print("="*70)

