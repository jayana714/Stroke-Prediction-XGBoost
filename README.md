# Stroke-Prediction-XGBoost
Predicting stroke using XGBoost with comparisons to Random Forest and SVM

## Overview
This project demonstrates stroke prediction using machine learning models. It compares the performance of three algorithms:
- **XGBoost (Extreme Gradient Boosting)**
- **Random Forest**
- **Support Vector Machines (SVM)**

The dataset used is the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset), and the objective is to classify individuals as being at risk of stroke or not based on various health and demographic attributes.

---

## Dataset
The dataset contains the following features:
- `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`
- **Target variable:** `stroke` (1 = Stroke occurred, 0 = No stroke)

---

### Preprocessing Steps:
1. **Handled Missing Values:**
   - Replaced missing values in the `bmi` column with the mean value.

2. **One-Hot Encoding:**
   - Converted categorical variables (e.g., `gender`, `work_type`) into numerical form using one-hot encoding.

3. **Feature Standardization:**
   - Standardized numerical features using `StandardScaler` to ensure optimal model performance.

**Code Snippet:**
```python
# Handle Missing Values
df['bmi'].fillna(df['bmi'].mean(), inplace=True)

# One-Hot Encoding
df = pd.get_dummies(df, columns=['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'], drop_first=True)

# Standardize Features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## Models Used

### 1. **XGBoost**
- A **boosting-based ensemble model** that builds decision trees sequentially, where each tree corrects the errors of the previous ones.
- **Hyperparameter Tuning:** GridSearchCV was used to find the best combination of parameters:
```python
xgb_params = {
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

xgb_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
grid_search = GridSearchCV(xgb_model, xgb_params, scoring="roc_auc", cv=5, verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)
```
- **Evaluation Metric:** ROC-AUC

---

### 2. **Random Forest**
- An **ensemble model** that builds multiple decision trees independently (bagging) and averages their predictions to improve accuracy.
- Key parameters:
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)
```

---

### 3. **SVM (Support Vector Machine)**
- A classification model that separates data points using a hyperplane.
- **Linear Kernel:** Used for simplicity and interpretability:
```python
from sklearn.svm import SVC

svm_model = SVC(probability=True, kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
```

---

## Results
The models were evaluated using the **AUC (Area Under the Curve)** score:

| Model            | AUC Score |
|-------------------|-----------|
| **SVM**          | 0.76      |
| **Random Forest** | 0.81      |
| **XGBoost**      | 0.83      |

**Conclusion:**  
XGBoost outperformed both SVM and Random Forest due to its ability to handle complex relationships, sequentially optimize residual errors, and incorporate hyperparameter tuning.

**Code Snippet: AUC Calculation**
```python
from sklearn.metrics import roc_auc_score

auc_svm = roc_auc_score(y_test, svm_model.predict_proba(X_test)[:, 1])
auc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
auc_xgb = roc_auc_score(y_test, grid_search.best_estimator_.predict_proba(X_test)[:, 1])

print(f"SVM AUC: {auc_svm:.4f}")
print(f"Random Forest AUC: {auc_rf:.4f}")
print(f"XGBoost AUC: {auc_xgb:.4f}")
```

---

## How to Run

### Prerequisites:
- Python 3.8 or higher
- Libraries:
  - `xgboost`
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`


---

## Future Improvements
1. **Feature Engineering:** Create new features to capture more relationships in the data.
2. **Advanced Hyperparameter Tuning:** Use `RandomizedSearchCV` or Bayesian optimization for faster and broader searches.
3. **Class Imbalance Handling:** Implement techniques like **SMOTE** (Synthetic Minority Oversampling) or class weighting to improve predictions for minority classes.
4. **Additional Models:** Explore deep learning models or other ensemble techniques like CatBoost or LightGBM.

---

## Acknowledgements
- **Dataset:** [Kaggle - Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
- **Libraries:**  
  - [Scikit-learn](https://scikit-learn.org/)  
  - [XGBoost](https://xgboost.readthedocs.io/)  
  - [Matplotlib](https://matplotlib.org/)

---


---





