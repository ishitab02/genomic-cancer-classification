import pickle
from src.model import get_xgboost_model, get_random_forest_model
from src.evaluation import evaluate_model

#loading and assigning data
with open('processed/X_train_pca.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('processed/X_test_pca.pkl', 'rb') as f:
    X_test = pickle.load(f)

with open('processed/y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open('processed/y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# training model(XGBoost)
model_xgb = get_xgboost_model()
model_xgb.fit(X_train, y_train)

# training model(Random Forest)
model_rf = get_random_forest_model()
model_rf.fit(X_train, y_train)

# Predict and evaluate accuracy (XGBoost)
y_pred_xgb = model_xgb.predict(X_test)
accuracy_xgb = evaluate_model(y_test, y_pred_xgb, model_name="XGBoost")

# Predict and evaluate accuracy (Random Forest)
y_pred_rf = model_rf.predict(X_test)
accuracy_rf = evaluate_model(y_test, y_pred_rf, model_name="Random Forest")

print("\nCONCLUSION:")
if accuracy_xgb > accuracy_rf:
    print("XGBoost performed better!")
elif accuracy_rf > accuracy_xgb:
    print("Random Forest performed better!")
else:
    print("Both models performed equally well!")

# Save trained model
with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(model_xgb, f)
with open('models/random_forest_model.pkl', 'wb') as f:
    pickle.dump(model_rf, f)
