
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Data = pd.read_csv("c:\\Users\\kudip\\Downloads\\insurance.csv")
Data = pd.read_csv("C:\\Users\\kudip\\Documents\\GitHub\\https---github.com-kudipudirajesh-Kudipudi-Rajesh-jan2026-Training.git\\Assignment-03\\data\\insurance.csv")

print(Data.head())
print(Data.info()) 
print(Data.describe())

print(Data.isnull().sum())

Data.fillna(Data.mean(numeric_only=True), inplace=True)

Data.drop_duplicates(inplace=True)

Data = pd.get_dummies(Data, drop_first=True)

plt.figure(figsize=(10,6))
sns.heatmap(Data.corr(), annot=True, cmap="coolwarm")
plt.show()


sns.scatterplot(x=Data['age'], y=Data['charges'])
plt.show()


X = Data.drop('charges', axis=1)   
y = Data['charges']                

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("R2 Score:", r2)

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print(coefficients)







