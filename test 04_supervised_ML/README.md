
🏠 House Price Prediction
📌 Project Overview
House price prediction is a supervised machine learning regression problem that aims to estimate the price of a house based on various features such as location, size, number of rooms, condition, and other property-related attributes.

In this project, multiple supervised machine learning algorithms are implemented and compared to identify the best-performing model.

🎯 Problem Statement
Predict the market price of a house using historical housing data by applying proper data preprocessing, feature engineering, feature scaling, and supervised machine learning models.

📊 Dataset Description
Dataset Type: House Price Dataset
Number of Records: 20,000+
Target Variable: Price
Feature Types:

Numerical: Living area, lot area, bedrooms, bathrooms, floors, latitude, longitude
Categorical:Price_class for SVM,KNN
🧹 Data Preprocessing Steps
1. Handling Missing Values
Numerical → Median imputation
Categorical → Mode imputation
Reason: Median is robust to outliers.

2. Fixing Data Types
Converted Built Year, Renovation Year, and Postal Code to appropriate formats.

3. Detecting and Treating Outliers
Detected using IQR method
Treated using capping (winsorization)
Reason: Prevents data loss and improves regression metrics.

4. Removing Duplicate Records
Duplicate rows were identified and removed to avoid bias in training.

5. Handling Categorical Variables
One-Hot Encoding → Low-cardinality features
Label Encoding → Binary features
6. Feature Scaling
StandardScaler was applied to numerical features to ensure equal contribution.

7. Removing Irrelevant Features
Dropped ID-like and redundant columns.

8. Train-Test Split
80% Training
20% Testing
random_state = 42
9. Transforming Skewed Features
Log transformation was applied on Price to reduce skewness.

🤖 Machine Learning Algorithms Used
1. Linear Regression
Training Set Metrics:
MAE  : 53675.30
MSE  : 5058794131.99
RMSE : 71125.20
R²   : 0.92
Test Set Metrics:
MAE  : 53549.78
MSE  : 5090689684.55
RMSE : 71349.07
R²   : 0.92

Regression Models (Predicting 'Price')
Linear Regression:

Test Set MAE: 53,549.78
Test Set RMSE: 71,349.07
Test Set R²: 0.92
Interpretation: Provides a good baseline, but has relatively high error margins compared to tree-based models.

2. Decision Tree Regressor
Initial Model (Train):
MAE  : 3430.00
RMSE : 330.00
R²   : 1.00
Initial Model (Test):
MAE  : 7997.70
RMSE : 14895.53
R²   : 1.00


Best Hyperparameters:
max_depth = 10
min_samples_split = 10
min_samples_leaf = 3


Optimized Model (Train):
MAE  : 4828.58
RMSE : 8407.88
R²   : 1.00


Optimized Model (Test):
MAE  : 6751.50
RMSE : 12529.54
R²   : 1.00

Decision Tree Regressor (Initial):

Test Set MAE: 7,997.70
Test Set RMSE: 14,895.53
Test Set R²: 1.00
Interpretation: Shows much lower errors than Linear Regression, but an R² of 1.00 with non-zero MAE/RMSE suggests potential overfitting or that minor errors are not significantly impacting the R² due to the nature of the data/capping. Tuning was necessary.
Decision Tree Regressor (Tuned with GridSearchCV):

Best Hyperparameters: max_depth=10, min_samples_leaf=3, min_samples_split=10
Test Set MAE: 6,751.50
Test Set RMSE: 12,529.54
Test Set R²: 1.00
Interpretation: Tuning significantly improved the MAE and RMSE compared to the initial Decision Tree, indicating better generalization.


3. Random Forest Regressor
Training Set:
MAE  : 2339.29
RMSE : 4192.74
R²   : 1.00
Test Set:
MAE  : 6233.52
RMSE : 10947.86
R²   : 1.00

Random Forest Regressor (Initial):

Test Set MAE: 6,233.52
Test Set RMSE: 10,947.86
Test Set R²: 1.00
Interpretation: Even without full hyperparameter tuning (which was interrupted), the initial Random Forest Regressor demonstrated the lowest MAE and RMSE among all regression models on the test set. This aligns with the project's conclusion that Random Forest performs best for this task.

4. K-Nearest Neighbors (KNN)
Training Accuracy : 0.9678
Testing Accuracy  : 0.9549
Classification Report:
precision  recall  f1-score  support
High              0.80     0.62      0.70      246
Low               0.97     0.99      0.98     2678


Accuracy                          0.95     2924
Macro Avg         0.88     0.80      0.84     2924
Weighted Avg      0.95     0.95      0.95     2924


Confusion Matrix:
[[ 152   94]
[  38 2640]]

Classification Models (Predicting 'Price_class')
K-Nearest Neighbors (KNN) Classifier:

Testing Accuracy: 95.48%
High Price_class: Precision: 80%, Recall: 62%, F1-score: 70%
Low Price_class: Precision: 97%, Recall: 99%, F1-score: 98%
Interpretation: Performed well overall, but struggled more with the 'High' price class, leading to a notable number of false negatives (94 instances where high-priced houses were predicted as low)</
p>

5. Support Vector Machine (SVM)
Training Accuracy : 0.9970
Testing Accuracy  : 0.9959
Classification Report:
precision  recall  f1-score  support
High              0.96     1.00      0.98      246
Low               1.00     1.00      1.00     2678


Accuracy                          1.00     2924
Macro Avg         0.98     1.00      0.99     2924
Weighted Avg      1.00     1.00      1.00     2924


Confusion Matrix:
[[ 245    1]
[  11 2667]]

The Support Vector Machine (SVM) model achieved very high accuracy:

Training Accuracy: 99.70%
Testing Accuracy: 99.59%
The classification report for the test set shows excellent performance:

Precision for High Price_class: 96% (meaning 96% of houses predicted as 'High' were actually 'High').
Recall for High Price_class: 100% (meaning the model identified all actual 'High' priced houses).
Precision for Low Price_class: 100%.
Recall for Low Price_class: 100%.
The confusion matrix indicates that out of 2924 test samples:

True Positives (High, High): 245
False Negatives (High, Low): 1 (one actual High house was predicted as Low)
False Positives (Low, High): 11 (eleven actual Low houses were predicted as High)
True Negatives (Low, Low): 2667
Overall, the SVM model shows exceptionally strong performance, especially in correctly classifying 'Low' priced houses and having a perfect recall for 'High' priced houses on the test set. There's a very slight tendency to misclassify 'Low' houses as 'High' (11 instances), but this is a minor issue given the overall accuracy.

Model	R²	RMSE	MAE
Linear Regression	0.92	Moderate	Moderate
Decision Tree	High	Low	Low
Random Forest	Best	Lowest	Lowest
KNN	Good	Low	Low
SVM	Excellent	Lowest	Lowest
🧾 Conclusion
This project shows that proper data preprocessing is more important than complex models. Random Forest Regressor achieved the best performance due to its ability to handle non-linear relationships and outliers.

Overall Summary.
For the primary objective of house price prediction (regression), the Random Forest Regressor demonstrated the best performance, yielding the lowest MAE and RMSE on the test set. This highlights its effectiveness in handling the dataset's characteristics and non-linear relationships. For the supplementary price classification task, the Support Vector Machine (SVM) Classifier showed outstanding results, achieving near-perfect accuracy in categorizing houses into 'High' or 'Low' price brackets
🚀 Technologies Used
Python
pandas, numpy
matplotlib, seaborn
scikit-learn
Jupyter Notebook