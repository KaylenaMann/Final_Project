#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np  

def clean_data(df, cutoff=None, training_columns=None, is_training=True, verbose=False):
    
    df = df.copy()
    
    df["title_i_binary"] = df["title_i_status"].apply(
        lambda x: 1 if "swp" in str(x).lower() or "tas" in str(x).lower() else 0
    )
    
    df["school_level"] = (
        df["school_level"]
        .astype(str)
        .str.strip()
        .str.title()
    )
    
    school_level_map = {
        "Prekindergarten": "Elementary",
        "Primary": "Elementary",
        "Elementary": "Elementary",
        "Middle": "Middle",
        "Secondary": "High",
        "High": "High"
    }
    df["school_level_clean"] = df["school_level"].map(school_level_map).fillna("Other")
    
    df["school_type"] = df["school_type"].astype(str).str.strip()
    school_type_map = {
        "Special education school": "Specialized",
        "Vocational school": "Specialized",
        "Regular school": "Regular school"
    }
    df["school_type_clean"] = df["school_type"].map(school_type_map).fillna("Other")
    
    df["charter"] = df["charter"].astype(str).str.strip().str.title()
    df["charter"] = df["charter"].map({
        "Yes": "Yes",
        "No": "No",
        "Charter": "Yes",
        "Not Charter": "No",
        "1": "Yes",
        "0": "No",
        "True": "Yes",
        "False": "No"
    }).fillna("No")
    
    df["meps_poverty_pct"] = pd.to_numeric(df["meps_poverty_pct"], errors="coerce")
    
    if is_training:
        df["math_test_pct_prof_midpt"] = pd.to_numeric(
            df["math_test_pct_prof_midpt"], errors="coerce"
        )
        df = df.dropna(subset=["math_test_pct_prof_midpt"]) 
    
        if cutoff is None:
            cutoff = df["math_test_pct_prof_midpt"].quantile(0.25)
    
        df["y"] = (df["math_test_pct_prof_midpt"] <= cutoff).astype(int)
    
    predictors = [
        "enrollment",
        "direct_certification",  
        "meps_poverty_pct",
        "meps_mod_poverty_pct", 
        "school_level_clean",
        "school_type_clean",  
        "charter",
        "title_i_binary"
    ]
    
    X = df[predictors].copy()
    
    continuous_cols = [
        "enrollment",
        "direct_certification",
        "meps_poverty_pct",
        "meps_mod_poverty_pct"
    ]
    
    X[continuous_cols] = X[continuous_cols].fillna(X[continuous_cols].mean())
 
    X = pd.get_dummies(X, drop_first=True)
    
    if not is_training and training_columns is not None:
        X = X.reindex(columns=training_columns, fill_value=0)

    if is_training:
        y = df["y"]
        return X, y, cutoff, X.columns.tolist(), continuous_cols
    else:
        return X

