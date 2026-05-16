import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from bayes import est_sample_cov, est_sample_mean, mahalanobis_distance

# Note: sklearn.svm uses LibSVM in its backend so I can use python :)

# All of our params
resolutions = ["16_20", "48_60"]
folds = [1, 2, 3]
n_eigens = 30
c_values = [0.1, 1, 10, 100]
d_values = [1, 2, 3]
gamma_values = [0.1, 1, 10, 100]

for resolution in resolutions:
    print(f"Resolution: {resolution}")
    poly_test_errors = []
    rbf_test_errors = []
    bayes_test_errors = []

    for fold in folds:
        x_train = np.loadtxt(f"data/{resolution}/trPCA_{fold:02d}.txt")[:, :n_eigens]
        x_val = np.loadtxt(f"data/{resolution}/valPCA_{fold:02d}.txt")[:, :n_eigens]
        x_test = np.loadtxt(f"data/{resolution}/tsPCA_{fold:02d}.txt")[:, :n_eigens]
        y_train = np.loadtxt(f"data/{resolution}/TtrPCA_{fold:02d}.txt", dtype=int).flatten()
        y_val = np.loadtxt(f"data/{resolution}/TvalPCA_{fold:02d}.txt", dtype=int).flatten()
        y_test = np.loadtxt(f"data/{resolution}/TtsPCA_{fold:02d}.txt", dtype=int).flatten()

        scaler = MinMaxScaler(feature_range=(-1, 1))
        x_train_s = scaler.fit_transform(x_train)
        x_val_s = scaler.transform(x_val)
        x_test_s = scaler.transform(x_test)

        best_poly_err, best_poly_params = float("inf"), None
        best_rbf_err, best_rbf_params = float("inf"), None

        for c in c_values:
            for d in d_values:
                clf = SVC(kernel="poly", C=c, degree=d, gamma=1.0, coef0=0.0)
                clf.fit(x_train_s, y_train)
                err = (clf.predict(x_val_s) != y_val).mean() * 100
                if err < best_poly_err:
                    best_poly_err, best_poly_params = err, (c, d)

            for g in gamma_values:
                clf = SVC(kernel="rbf", C=c, gamma=g)
                clf.fit(x_train_s, y_train)
                err = (clf.predict(x_val_s) != y_val).mean() * 100
                if err < best_rbf_err:
                    best_rbf_err, best_rbf_params = err, (c, g)

        clf = SVC(
            kernel="poly",
            C=best_poly_params[0],
            degree=best_poly_params[1],
            gamma=1.0,
            coef0=0.0,
        )
        clf.fit(x_train_s, y_train)
        poly_test_err = (clf.predict(x_test_s) != y_test).mean() * 100
        poly_test_errors.append(poly_test_err)

        clf = SVC(kernel="rbf", C=best_rbf_params[0], gamma=best_rbf_params[1])
        clf.fit(x_train_s, y_train)
        rbf_test_err = (clf.predict(x_test_s) != y_test).mean() * 100
        rbf_test_errors.append(rbf_test_err)

        # ML estimation
        classes = np.unique(y_train)
        means, covs = {}, {}
        for c in classes:
            x_c = x_train[y_train == c]
            means[c] = est_sample_mean(x_c)
            covs[c] = np.diag(np.diag(est_sample_cov(x_c, means[c])))

        d = {c: mahalanobis_distance(x=x_test, mean=means[c], cov=covs[c]) for c in classes}
        bayes_preds = np.where(d[classes[0]] < d[classes[1]], classes[0], classes[1])
        bayes_test_err = (bayes_preds != y_test).mean() * 100
        bayes_test_errors.append(bayes_test_err)

        print(f"Fold: {fold}:")
        print(
            f"Poly best: C={best_poly_params[0]}, D={best_poly_params[1]} | val err={best_poly_err:.2f}% | test err={poly_test_err:.2f}%",
        )
        print(
            f"RBF best: C={best_rbf_params[0]}, gamma={best_rbf_params[1]} | val err={best_rbf_err:.2f}% | test err={rbf_test_err:.2f}%",
        )
        print(f"Bayes | test err={bayes_test_err:.2f}%")

    print(f"{resolution} Poly avg test error: {np.mean(poly_test_errors):.2f}%")
    print(f"{resolution} RBF avg test error: {np.mean(rbf_test_errors):.2f}%")
    print(f"{resolution} Bayes avg test error: {np.mean(bayes_test_errors):.2f}%")
