import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

# Features with one image per row
# Take the first 30 eigens
x_train = np.loadtxt("data/16_20/trPCA_01.txt")[:, :30]
x_val = np.loadtxt("data/16_20/valPCA_01.txt")[:, :30]
x_test = np.loadtxt("data/16_20/tsPCA_01.txt")[:, :30]

# Labels are all on one line where 1=male, 2=female
y_train = np.loadtxt("data/16_20/TtrPCA_01.txt", dtype=int).flatten()
y_val = np.loadtxt("data/16_20/TvalPCA_01.txt", dtype=int).flatten()
y_test = np.loadtxt("data/16_20/TtsPCA_01.txt", dtype=int).flatten()


scaler = MinMaxScaler(feature_range=(-1, 1))
# Learns MinMax from train then applies to train, val, test
x_train_s = scaler.fit_transform(x_train)  # fit and transform
x_val_s = scaler.transform(x_val)  # transform only
x_test_s = scaler.transform(x_test)

# C=0.1, 1, 10, 100

# Polynomial: gamma=1, coef0=0 as specified in the assignment
# d=1, 2, and 3
# γ=1 and c0=0
clf = SVC(kernel="poly", C=10, degree=2, gamma=1.0, coef0=0.0)

# RBF
# γ=0.1., 1, 10, and 100
clf = SVC(kernel="rbf", C=10, gamma=0.1)

# Sweep to find (γopt, Copt)
# Then use them to compute the misclassifciation rate on each fold and average
# Repeat for other image size

# Same API for both
clf.fit(x_train_s, y_train)
val_preds = clf.predict(x_val_s)
test_preds = clf.predict(x_test_s)

error_rate = (test_preds != y_test).mean() * 100
print(f"Test error: {error_rate:.2f}%")
