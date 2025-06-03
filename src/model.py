from xgboost import XGBClassifier

# define xgboost model
def get_xgboost_model():
  return XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss',
  )
