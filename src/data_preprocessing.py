import pandas as pd

X = pd.read_csv('data/data.csv', index_col=0) # (801, 16383)
y = pd.read_csv('data/labels.csv', index_col=0) #(801,1)

# assert (X.index == y.index).all(), "Sample IDs in X and y do not match!"
