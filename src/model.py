from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# define XGBoost model
def get_xgboost_model():
  return XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss',
  )

# define random forest model
def get_random_forest_model():
    return RandomForestClassifier(
        random_state=42
    )