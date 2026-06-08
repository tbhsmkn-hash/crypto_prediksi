from sklearn.svm import SVR
from skopt import BayesSearchCV
from skopt.space import Real


def optimize_svr(X_train, y_train):

    search_space = {
        "C": Real(
            0.1,
            1000.0,
            prior="log-uniform"
        ),

        "gamma": Real(
            1e-5,
            1.0,
            prior="log-uniform"
        ),

        "epsilon": Real(
            0.001,
            1.0,
            prior="log-uniform"
        )
    }

    optimizer = BayesSearchCV(
        estimator=SVR(kernel="rbf"),

        search_spaces=search_space,

        n_iter=20,

        cv=3,

        n_jobs=-1,

        random_state=42,

        scoring="neg_mean_absolute_error"
    )

    optimizer.fit(
        X_train,
        y_train
    )

    return optimizer.best_estimator_