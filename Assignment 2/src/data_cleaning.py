import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("C:\\Users\\kudip\\Downloads\\dataset.csv")
print(df.head())
print(df.info())

# ----- Handling Missing Values [completed] -----
df.isnull().sum()


# ----- Remove Duplicates [completed] -----
df.drop_duplicates(inplace=True)

# ----- Fix Data Types [completed] -----
for col in df.select_dtypes(include='object'):
    df[col] = df[col].astype(str)

# ----- Outlier Treatment (IQR) [blocked] -----
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = np.clip(df[col], lower, upper)

# Save cleaned dataset
df.to_csv("data/cleaned_dataset.csv", index=False)

print("Data preprocessing completed successfully!")