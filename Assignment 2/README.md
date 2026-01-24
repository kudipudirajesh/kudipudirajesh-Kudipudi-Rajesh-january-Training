Missing Value Handling
Median imputation worked best for numerical features because it is robust to outliers, while mode was effective for categorical features as it preserved the most frequent category.
Categorical Encoding
One-Hot Encoding performed best for nominal features with few categories. Ordinal Encoding was effective for ordered features as it preserved ranking. Frequency and Target Encoding handled high-cardinality features efficiently without increasing dimensionality.
Feature Scaling
Z-score standardization was the most effective scaling method as it centered features around zero with unit variance, making it suitable for most machine learning algorithms. Min-Max scaling was useful when bounded ranges were required.
Outlier & Skewness Treatment
Outlier treatment using IQR reduced noise and improved feature stability. Log and power transformations helped normalize skewed distributions, improving model readiness.
Final Preprocessing Choice
The final preprocessing pipeline was selected to balance data integrity, model compatibility, and performance, ensuring the dataset was clean, normalized, and ready for machine learning.