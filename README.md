# Stroke Prediction with XGBoost

Predicting stroke risk using ensemble ML models on health and demographic data.

## Results

| Model | AUC Score |
|---|---|
| SVM | 0.76 |
| Random Forest | 0.81 |
| XGBoost | **0.83** |

XGBoost outperformed both baselines after hyperparameter tuning with GridSearchCV across learning rate, tree depth, and subsampling parameters.

## Key decisions

- Replaced missing BMI values with column mean; one-hot encoded categorical variables
- Standardized features with StandardScaler before training
- Evaluated with ROC-AUC instead of accuracy due to class imbalance in stroke outcomes
- Tuned XGBoost across 108 hyperparameter combinations using 5-fold cross-validation

## Run locally

```bash
pip install -r requirements.txt
jupyter notebook stroke_prediction.ipynb
```

## Stack
Python · XGBoost · Scikit-learn · Pandas · Matplotlib

