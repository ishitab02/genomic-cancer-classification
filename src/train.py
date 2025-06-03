import pickle
from src.model import get_xgboost_model
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

# training model
model = get_xgboost_model()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
evaluate_model(y_test, y_pred)

# Save trained model
with open('models/xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)
