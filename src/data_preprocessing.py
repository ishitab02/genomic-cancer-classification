import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

gene_df = pd.read_csv('data/data.csv', index_col=0) 
labels_df = pd.read_csv('data/labels.csv')
labels_df.rename(columns={'Unnamed: 0': 'sample_id', 'Class': 'class'}, inplace=True)
 
merged_df = gene_df.merge(labels_df, left_index = True, right_on = "sample_id")
print(merged_df.head())

# Separating feature columns and target column
X = merged_df.drop(columns = ["sample_id", "class"])
y = merged_df["class"]

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalizing the feature columns (using z-score normalization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)

# print(X_scaled_df.shape)
# print(X_scaled_df.head())

# Calculating variance per gene to identify low-variance genes (only on training set)
variances = X_train_scaled_df.var(axis=0)

# Plotting variance distribution 
plt.figure(figsize=(10, 5))
plt.hist(variances, bins=100, color='red', edgecolor='black')
plt.title("Distribution of Gene Variances After Z-Score Normalization on Training Set")
plt.xlabel("Variance")
plt.ylabel("Number of Genes")
plt.grid(True)
plt.show()

# Dropping low-variance genes (variance < 0.01 based on the plot)
variance_threshold = 0.01
high_variance_genes = variances[variances >= variance_threshold].index

X_train_filtered_df = X_train_scaled_df[high_variance_genes]
X_test_filtered_df = X_test_scaled_df[high_variance_genes]

print(f"Original number of genes: {X_train_scaled_df.shape[1]}")
print(f"Number of genes after filtering: {X_train_filtered_df.shape[1]}")


with open('processed/X_train_filtered.pkl', 'wb') as f:
    pickle.dump(X_train_filtered_df, f)

with open('processed/X_test_filtered.pkl', 'wb') as f:
    pickle.dump(X_test_filtered_df, f)

with open('processed/y_train.pkl', 'wb') as f:
    pickle.dump(y_train, f)

with open('processed/y_test.pkl', 'wb') as f:
    pickle.dump(y_test, f)
