# Project: Cleaning data for problem "predict powerful client"
# Date: 05/07/2025
# Last Modification: 05/07/2025
# Author: Daniel Martín Medina

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('dataset_banco.csv')
print(f"Initial shape: {data.shape}")

# Remove rows with missing values
data.dropna(inplace=True)
print(f"Shape after dropping NA: {data.shape}")

# Count different sublevels in categorical columns
cols_cat = ['job', 'marital', 'education', 'default', 'housing',
            'loan', 'contact', 'month', 'poutcome', 'y']

print("\nUnique sublevels in categorical columns:")
for col in cols_cat:
    print(f' - {col}: {data[col].nunique()}')

# Check diversity in numeric values using standard deviation
cols_num = ['age', 'balance', 'day', 'duration', 'campaign',
            'pdays', 'previous']

stds = data[cols_num].std()
print("\nNumeric columns with 0 standard deviation (no variability):")
print(stds[stds == 0])

# Remove duplicate rows
print(f"\nLength before removing duplicates: {data.shape[0]}")
data.drop_duplicates(inplace=True)
print(f"Length after removing duplicates: {data.shape[0]}")

# Plot boxenplots for numeric columns to detect outliers
fig, ax = plt.subplots(nrows=7, ncols=1, figsize=(8, 30))
fig.subplots_adjust(hspace=0.5)

for i, col in enumerate(cols_num):
    sns.boxenplot(x=col, data=data, ax=ax[i])
    ax[i].set_title(col)

plt.tight_layout()
plt.show()

# Remove outliers based on domain knowledge
before = data.shape[0]
data = data[data['age'] <= 100]
print(f"Removed {before - data.shape[0]} rows with age > 100")

before = data.shape[0]
data = data[data['duration'] > 0]
print(f"Removed {before - data.shape[0]} rows with duration <= 0")

before = data.shape[0]
data = data[data['previous'] < 100]
print(f"Removed {before - data.shape[0]} rows with previous >= 100")

print(f"\nLength after outlier removal: {data.shape[0]}")

# Plot counts for categorical variables
fig, ax = plt.subplots(nrows=10, ncols=1, figsize=(10, 30))
fig.subplots_adjust(hspace=1)

for i, col in enumerate(cols_cat):
    sns.countplot(x=col, data=data, ax=ax[i])
    ax[i].set_title(col)
    ax[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Correlation heatmap for numeric columns
plt.figure(figsize=(10, 8))
sns.heatmap(data[cols_num].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix for Numeric Features")
plt.show()

# Save the cleaned dataset
data.to_csv('cleaned_dataset_banco.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_dataset_banco.csv'")
