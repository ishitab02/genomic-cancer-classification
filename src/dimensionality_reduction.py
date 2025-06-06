import pickle
from sklearn.decomposition import PCA

with open('processed/X_train_filtered.pkl', 'rb') as f:
    X_train_filtered = pickle.load(f)
with open('processed/X_test_filtered.pkl', 'rb') as f:
    X_test_filtered = pickle.load(f)

# Fitting PCA (Principal Component Analysis) to training data
pca = PCA()
pca.fit(X_train_filtered)

# Calculating cumulative explained variance (95% threshold)
cumulative_variance = pca.explained_variance_ratio_.cumsum()
n_components_95 = (cumulative_variance >= 0.95).argmax() + 1
print(f"Components to retain 95% variance: {n_components_95}")

# PCA transformation based on the number of components that retain 95% variance
pca_final = PCA(n_components=n_components_95, random_state=42)
X_train_pca = pca_final.fit_transform(X_train_filtered)
X_test_pca = pca_final.transform(X_test_filtered)

# Saving the data
with open('processed/X_train_pca.pkl', 'wb') as f:
    pickle.dump(X_train_pca, f)
with open('processed/X_test_pca.pkl', 'wb') as f:
    pickle.dump(X_test_pca, f)
with open('processed/pca_vars.pkl', 'wb') as f:
    pickle.dump((cumulative_variance, n_components_95), f)
