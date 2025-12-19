#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import joblib
from pathlib import Path
from Clean_data import clean_data


# In[12]:

#making sure anyone can run this script
script_dir = Path(__file__).parent
model_dir = script_dir / "model"
data_dir = script_dir.parent / "Raw Data"

new_data = pd.read_csv(data_dir / "Excel_template.csv")

# In[13]:

#loading in Model
model = joblib.load(model_dir / "final_model.pkl")
scaler = joblib.load(model_dir / "scaler.pkl")
training_columns = joblib.load(model_dir / "training_columns.pkl")
cutoff = joblib.load(model_dir / "cutoff.pkl")
continuous_features = joblib.load(model_dir / "continuous_features.pkl")

print(f"Loaded {len(new_data)} school(s) for prediction")
print(f"Cutoff threshold: {cutoff:.4f}")
print(f"\nInput data:")
print(new_data.T)


# In[14]:


#cleaning data
try:
    X_new = clean_data(
        new_data,
        cutoff=cutoff,
        training_columns=training_columns,
        is_training=False
    )
    print(f"\nFeatures after cleaning: {X_new.shape}")
except Exception as e:
    print(f"\n❌ ERROR cleaning data: {e}")
    print("\nMake sure your CSV has these required columns:")
    print("  - school_level")
    print("  - school_type")
    print("  - charter")
    print("  - title_i_status")
    print("  - enrollment")
    print("  - direct_certification")
    print("  - meps_poverty_pct")
    print("  - meps_mod_poverty_pct")
    print("\nColumn names are case-sensitive!")
    exit(1)

print(f"Continuous features to scale: {len(continuous_features)}")
print(f"Categorical features (not scaled): {len(training_columns) - len(continuous_features)}")

X_new_scaled = X_new.copy()
X_new_scaled[continuous_features] = scaler.transform(X_new[continuous_features])

print(f"\nKey features (after scaling):")
print(f"  enrollment: {X_new_scaled['enrollment'].values[0]:.2f}")
print(f"  meps_poverty_pct: {X_new_scaled['meps_poverty_pct'].values[0]:.2f}")
print(f"  direct_certification: {X_new_scaled['direct_certification'].values[0]:.2f}")

#Making Predictions
pred_class = model.predict(X_new_scaled)[0]
pred_prob = model.predict_proba(X_new_scaled)[0, 1]

print(f"\n{'='*70}")
print(f"PREDICTION RESULTS")
print(f"{'='*70}")
print(f"Prediction: {'BOTTOM QUARTILE (HIGH RISK)' if pred_class == 1 else '✓ NOT bottom quartile'}")
print(f"Probability of bottom quartile: {pred_prob:.1%}")
print(f"{'='*70}")

