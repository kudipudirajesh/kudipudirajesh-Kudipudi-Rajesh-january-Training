import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
df = pd.read_csv("data/cleaned_dataset.csv")
categorical_cols = df.select_dtypes(include='object').columns
print("Categorical Columns:", categorical_cols)
df = pd.get_dummies(df, columns=['Applicant_Gender'], drop_first=True)

ordinal_encoder = OrdinalEncoder(categories=[['UG', 'PG', 'PhD']])
df['Education_Ordinal'] = ordinal_encoder.fit_transform(df[['Education_Type']])
freq_map = df['City'].value_counts().to_dict()
df['City_Frequency'] = df['City'].map(freq_map)

df.to_csv("data/encoded_dataset.csv", index=False)
print("Categorical encoding completed successfully")