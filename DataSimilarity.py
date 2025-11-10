# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 01:51:57 2025

@author: nasir.ibrahim
"""
#%% CReate Dataset
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Generate small target dataset (local hospital)
np.random.seed(42)
n_target = 20
age_target = np.random.randint(40, 80, n_target)
bmi_target = np.random.uniform(18, 35, n_target)
bp_target = 100 + 0.5 * age_target + 1.2 * bmi_target + np.random.normal(0, 5, n_target)

target_data = pd.DataFrame({'age': age_target, 'bmi': bmi_target, 'bp': bp_target})
target_data['source'] = 'target'  # Mark the source

# Generate larger external dataset (another hospital)
n_external = 100
age_external = np.random.randint(30, 90, n_external)
bmi_external = np.random.uniform(16, 40, n_external)
bp_external = 95 + 0.4 * age_external + 1.5 * bmi_external + np.random.normal(0, 8, n_external)

external_data = pd.DataFrame({'age': age_external, 'bmi': bmi_external, 'bp': bp_external})
external_data['source'] = 'external'  # Mark the source

#%% Compute Propensity SCore
from sklearn.linear_model import LogisticRegression

# Combine target and external data
data = pd.concat([target_data, external_data], ignore_index=True)
data['label'] = (data['source'] == 'target').astype(int)  # 1 for target, 0 for external

# Standardize features for logistic regression
scaler = StandardScaler()
X = scaler.fit_transform(data[['age', 'bmi']])

# Fit logistic regression model to predict target vs. external
logreg = LogisticRegression()
logreg.fit(X, data['label'])

# Compute propensity scores (probability of being in target group)
data['propensity_score'] = logreg.predict_proba(X)[:, 1]
"""
Higher propensity scores (closer to 1) mean the data point is more similar to the target dataset.
Lower propensity scores (closer to 0) mean the data point is dissimilar.
"""
#%%Step 3: Adjust Weights Using Inverse Probability Weighting (IPW)
# Now, we adjust the weights based on how well the logistic model can distinguish between datasets.

# Compute inverse probability weighting (IPW)
data['weight'] = data['propensity_score'] / (1 - data['propensity_score'])

# Cap extreme weights to avoid instability (truncate at 95th percentile)
max_weight = data['weight'].quantile(0.95)
data['weight'] = np.clip(data['weight'], 0, max_weight)

# Assign full weight of 1 to the target dataset
data.loc[data['source'] == 'target', 'weight'] = 1

#%% Train Weighted Linear Regression Model
# We train a weighted linear regression model, where each observation contributes according to its assigned weight.

# Train weighted linear regression
X_train = data[['age', 'bmi']]
y_train = data['bp']
weights = data['weight']
weights_2 = weights.copy()
indices = np.where(weights < 0.1)
weights_2[indices] = 0

model = LinearRegression()
model.fit(X_train, y_train, sample_weight=weights)

predicts = model.predict(X_train)
predicts = pd.Series(predicts)

# Model coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Score:", model.score(X_train, y_train, sample_weight=weights))

#%% Evaluate the Model

# Train an unweighted model on only target data
model_target_only = LinearRegression()
model_target_only.fit(target_data[['age', 'bmi']], target_data['bp'])
targetpredicts = model.predict(X_train)
targetpredicts = pd.Series(targetpredicts)

# Train an unweighted model using all data equally
model_global = LinearRegression()
model_global.fit(data[['age', 'bmi']], data['bp'])
globalpredicts = model.predict(X_train)
globalpredicts = pd.Series(globalpredicts)

# Print comparison of models
print("\nTarget-Only Model Coefficients:", model_target_only.coef_)
print("Target Score:", model_target_only.score(target_data[['age', 'bmi']], target_data['bp']))

print("\nGlobal Model Coefficients:", model_global.coef_)
print("GlobalScore:", model_global.score(data[['age', 'bmi']], data['bp']))

print("\nWeighted Model Coefficients:", model.coef_)
print("Weighted Score:", model.score(X_train, y_train, sample_weight=weights))

