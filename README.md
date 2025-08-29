# Loan_approval_prediction
Loan Approval Prediction 

This project builds a Machine Learning pipeline to predict whether a loan application will be approved using Kaggle’s Loan Prediction Dataset
 (or similar).

We handle missing values, encode categorical features, balance the dataset, and evaluate multiple classification models. The focus is on performance metrics such as precision, recall, and F1-score, which are crucial for imbalanced datasets.

Project Overview
Goal: Predict loan approval status (Loan_Status).
Dataset: Loan Approval Prediction dataset from Kaggle.
Problem Type: Binary Classification (Approved vs. Not Approved).
Challenges: Missing values, categorical encoding, and class imbalance.

Workflow
1. Import Libraries
We use pandas, numpy, matplotlib, seaborn for preprocessing & visualization and scikit-learn for ML models.
2. Load Dataset
Dataset is read from CSV and inspected.
3. Handle Missing Values
Categorical columns → filled with mode.
Numerical columns → filled with median.
4. Encode Categorical Features
Label Encoding applied to convert categorical values into numerical form.
5. Train-Test Split
Split into 80% training and 20% testing.
Stratified sampling ensures class proportions remain consistent.
6. Handle Class Imbalance
Manual Oversampling: Minority class is oversampled to match majority.
(🔹 Bonus: SMOTE or other methods can also be applied).
7. Train & Evaluate Models
We trained three models:
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
For each model, we report:
Accuracy
Precision, Recall, F1-score (from classification report)
Confusion Matrix
8. Model Comparison
Compare accuracy scores using a barplot.

Balanced dataset improved model fairness.
Random Forest generally performed best in terms of accuracy & F1-score.
Precision/Recall trade-offs highlight importance of handling imbalance.

Future Improvements
Apply SMOTE, ADASYN, or Tomek Links for advanced balancing.
Hyperparameter tuning with GridSearchCV / RandomizedSearchCV.
Try boosting methods (XGBoost, LightGBM, CatBoost).
Deploy with Flask/Streamlit for real-time predictions.

Tech Stack
Python (Pandas, NumPy, Matplotlib, Seaborn)
Scikit-learn (ML models, evaluation)
Imbalanced-learn (optional for SMOTE & resampling)

Key Learnings
Handling missing values systematically improves data quality.
Label Encoding prepares categorical data for ML.
Balancing datasets is critical in binary classification.
Comparing multiple models helps in selecting the best performing one.
