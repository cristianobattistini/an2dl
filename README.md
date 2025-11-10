# an2dl
Repo for challenge1 and challenge2 of the Polimi Class,
containing the code for the two assignments of the Artificial Neural Networks and Deep Learning course at Politecnico di Milano.



# y_train_df must contain sample_index + label columns
y_train_df = pd.DataFrame({
    "sample_index": X_train["sample_index"].unique(),
    "label": y_train
})

X_train_seq, y_train_seq = build_sequences(X_train, y_train_df)

N, T, F = X_train_seq.shape
X_train_flat = X_train_seq.reshape(N, T * F).astype(np.float32)


y_val_df = pd.DataFrame({
    "sample_index": X_val["sample_index"].unique(),
    "label": y_val
})

X_val_seq, y_val_seq = build_sequences(X_val, y_val_df)
X_val_flat   = X_val_seq.reshape(X_val_seq.shape[0], -1).astype(np.float32)


X_test_seq, _ = build_sequences(X_test)  # no labels → returns (dataset, None)
N, T, F = X_test_seq.shape
X_test_flat = X_train_seq.reshape(N, T * F).astype(np.float32)

input_shape = X_train_flat.shape[1] # extract the shape of a single sequence
num_classes = len(np.unique(y_train)) # how many unique pain level exists
input_shape, num_classes