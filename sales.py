import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 1. LOAD THE DATA
# ==========================================
# Ensure these files are in your current working directory
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print("Data loaded successfully. Combining datasets for uniform preprocessing...")

# Add a marker so we can separate them later
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['Item_Outlet_Sales'] = 0  # Placeholder for target

combined_df = pd.concat([train_df, test_df], ignore_index=True)

# ==========================================
# 2. DATA PREPROCESSING & CLEANING
# ==========================================

# A. Fix inconsistent text labels in Item_Fat_Content
# (BigMart data usually contains 'LF', 'low fat', 'Regular' typos)
combined_df['Item_Fat_Content'] = combined_df['Item_Fat_Content'].replace(
    {'LF': 'Low Fat', 'low fat': 'Low Fat', 'reg': 'Regular'}
)

# B. Handle Missing Values
# Fill missing Item_Weight with the overall mean weight
combined_df['Item_Weight'] = combined_df['Item_Weight'].fillna(combined_df['Item_Weight'].mean())

# Fill missing Outlet_Size with the mode (most common value)
combined_df['Outlet_Size'] = combined_df['Outlet_Size'].fillna(combined_df['Outlet_Size'].mode()[0])

# C. Encode Categorical Features
# Drop high-cardinality unique IDs that won't help the regression algorithms directly
features_to_encode = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

le = LabelEncoder()
for col in features_to_encode:
    combined_df[col] = le.fit_transform(combined_df[col])

# ==========================================
# 3. SEPARATE BACK INTO TRAIN & TEST
# ==========================================
train_clean = combined_df[combined_df['is_train'] == 1].drop(columns=['is_train'])
test_clean = combined_df[combined_df['is_train'] == 0].drop(columns=['is_train', 'Item_Outlet_Sales'])

# Drop ID strings from features
X = train_clean.drop(columns=['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
y = train_clean['Item_Outlet_Sales']

# Split the training data to evaluate model performance locally
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 4. TRAIN & EVALUATE MODELS
# ==========================================
# Model 1: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_val)

# Evaluation metrics
mae = mean_absolute_error(y_val, rf_preds)
rmse = np.sqrt(mean_squared_error(y_val, rf_preds))
r2 = r2_score(y_val, rf_preds)

print("\n====================================")
print("     Random Forest Evaluation       ")
print("====================================")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score (Accuracy Metric): {r2:.2f}")

# ==========================================
# 5. GENERATE FINAL TEST PREDICTIONS
# ==========================================
# Clean up test features to match X
X_test_final = test_clean.drop(columns=['Item_Identifier', 'Outlet_Identifier'])
final_predictions = rf_model.predict(X_test_final)

# Create submission layout matching your 'sample_submission' structure
submission = pd.DataFrame({
    'Item_Identifier': test_df['Item_Identifier'],
    'Outlet_Identifier': test_df['Outlet_Identifier'],
    'Item_Outlet_Sales': final_predictions
})

submission.to_csv('final_predictions.csv', index=False)
print("\n[Success] Final test predictions saved to 'final_predictions.csv'!")

# ==========================================
# 6. VISUALIZE ACTUAL VS PREDICTED
# ==========================================
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_val, y=rf_preds, alpha=0.4, color='purple')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], color='red', lw=2, linestyle='--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Random Forest Regressor: Actual vs Predicted Sales')
plt.tight_layout()
plt.show()