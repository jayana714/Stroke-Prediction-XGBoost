# Stroke-Prediction-XGBoost
Predicting stroke using XGBoost with comparisons to Random Forest and SVM
# Stroke Prediction Using XGBoost

## Overview
This project demonstrates stroke prediction using machine learning models. It compares the performance of three algorithms:
- **XGBoost (Extreme Gradient Boosting)**
- **Random Forest**
- **Support Vector Machines (SVM)**

The dataset used is the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset), and the objective is to classify individuals as being at risk of stroke or not based on various health and demographic attributes.

## Dataset
The dataset contains the following features:
- `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`
- Target variable: `stroke`

- ### Preprocessing Steps:
1. Handled missing values in the `bmi` column by replacing them with the mean.
2. Converted categorical variables into numerical ones using one-hot encoding.
3. Standardized numerical features for better model performance.

## Models Used
### 1. **XGBoost**
- **Boosting-based ensemble model** that sequentially improves weak learners.
- Hyperparameter tuning was performed using GridSearchCV with the following parameters:
  - `learning_rate`: [0.01, 0.1, 0.2]
  - `n_estimators`: [100, 200, 300]
  - `max_depth`: [3, 5, 7]
  - `subsample`: [0.8, 1.0]
  - `colsample_bytree`: [0.8, 1.0]
- Evaluation metric: **ROC-AUC**

### 2. **Random Forest**
- Ensemble model based on **bagging**, which builds multiple decision trees independently and averages their predictions.
- Used 100 trees with a maximum depth of 10.

### 3. **SVM**
- Classifies data by finding the best hyperplane separating the classes.
- Linear kernel was used for simplicity.

## Results
| Model            | AUC Score |
|-------------------|-----------|
| SVM              | 0.76      |
| Random Forest     | 0.81      |
| XGBoost          | 0.83      |

XGBoost outperformed the other models due to its ability to handle complex relationships and optimize residuals iteratively.

## How to Run
### Prerequisites:
- Python 3.8 or higher
- Libraries:
  - `xgboost`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
Key Files
Project.ipynb: Main notebook containing the code for data preprocessing, model training, and evaluation.
Dataset: Ensure the dataset is in the appropriate directory (healthcare-dataset-stroke-data.csv).
Future Improvements
Incorporate feature engineering techniques to improve model accuracy.
Experiment with additional hyperparameters for XGBoost.
Address class imbalance using SMOTE or weighted loss functions.
Acknowledgements
Dataset: Kaggle - Stroke Prediction Dataset


