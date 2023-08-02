import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model
from sklearn import svm
from sklearn import tree, neighbors
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, BaggingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from src.ml import metric_cal
from . import plotting


def linear_regression(path: str, target_variable: str, independent_variables: list):
    df = pd.read_csv(path)
    regr = linear_model.LinearRegression()
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.90, random_state=1)
    regr.fit(train_x, train_y)
    test_y_ = regr.predict(test_x)

    return_dict = {"Coefficients": regr.coef_.tolist()[0], "Intercept": regr.intercept_.tolist()[0]}
    r2 = metric_cal.metric_r2(test_y, test_y_)
    mean_absolute = metric_cal.metric_mean_absolute(test_y, test_y_)
    mean_squared = metric_cal.metric_mean_squared(test_y, test_y_)
    root_mean_squared = metric_cal.metric_root_mean_squared(test_y, test_y_)
    mean_squared_log = metric_cal.metric_mean_squared_log(test_y, test_y_)
    mean_absolute_percentage = metric_cal.metric_mean_absolute_percentage(test_y, test_y_)
    media_absolute = metric_cal.metric_media_absolute(test_y, test_y_)
    max_error = metric_cal.metric_max_error(test_y, test_y_)

    # Feature importance chart
    importance_values = regr.coef_[0].tolist()
    fig = plotting.plot_importance_figure(importance_values, independent_variables)
    feature_imp_plot = plotting.figure_to_base64(fig)

    return_dict.update({"r2_score": r2})
    return_dict.update({"mae": mean_absolute})
    return_dict.update({"mse": mean_squared})
    return_dict.update({"rmse": root_mean_squared})
    return_dict.update({"mean_squared_log": mean_squared_log})
    return_dict.update({"mean_absolute_percentage": mean_absolute_percentage})
    return_dict.update({"media_absolute": media_absolute})
    return_dict.update({"max_error": max_error})
    return_dict.update({"feature_imp_plot": feature_imp_plot})
    return_dict.update({"importance_values": importance_values})
    return_dict.update({"independent_variables": independent_variables})
    return return_dict


def support_vector_machines(path: str, target_variable: str, independent_variables: list):
    df = pd.read_csv(path)   #change
    x = df[independent_variables]
    y = df[[target_variable]]
    regr = svm.SVR(kernel="linear", C=100, gamma="auto")
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.90, random_state=1)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    regr.fit(train_x, train_y)
    test_y_ = regr.predict(test_x)
    print('Coefficients: ', regr.coef_)
    print('Intercept: ', regr.intercept_)
    print("scikit metrics mean absolute error: %.6f" % mean_absolute_error(test_y_, test_y))
    print("scikit metrics mean squared error: %.4f" % mean_squared_error(test_y_, test_y))
    print("Residual sum of squares (MSE): %.4f" % np.mean((test_y_ - test_y) ** 2))
    print("R2-score: %.4f" % r2_score(test_y, test_y_))

    return_dict = {"Coefficients": regr.coef_.tolist()[0], "Intercept": regr.intercept_.tolist()[0]}
    r2 = metric_cal.metric_r2(test_y, test_y_)
    mean_absolute = metric_cal.metric_mean_absolute(test_y, test_y_)
    mean_squared = metric_cal.metric_mean_squared(test_y, test_y_)
    root_mean_squared = metric_cal.metric_root_mean_squared(test_y, test_y_)
    mean_squared_log = metric_cal.metric_mean_squared_log(test_y, test_y_)
    mean_absolute_percentage = metric_cal.metric_mean_absolute_percentage(test_y, test_y_)
    media_absolute = metric_cal.metric_media_absolute(test_y, test_y_)
    max_error = metric_cal.metric_max_error(test_y, test_y_)

    return_dict.update({"r2_score": r2})
    return_dict.update({"mae": mean_absolute})
    return_dict.update({"mse": mean_squared})
    return_dict.update({"rmse": root_mean_squared})
    return_dict.update({"mean_squared_log": mean_squared_log})
    return_dict.update({"mean_absolute_percentage": mean_absolute_percentage})
    return_dict.update({"media_absolute": media_absolute})
    return_dict.update({"max_error": max_error})

    return return_dict


def decision_trees(path: str, target_variable: str, independent_variables: list, algo_params: dict):
    df = pd.read_csv(path)

    regr = tree.DecisionTreeRegressor(**algo_params)
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.90, random_state=1)
    regr.fit(train_x, train_y)
    test_y_ = regr.predict(test_x)

    print('Coefficients: ', regr.score(test_x, test_y))
    print("scikit metrics mean absolute error: %.6f" % mean_absolute_error(test_y_, test_y))
    print("scikit metrics mean squared error: %.4f" % mean_squared_error(test_y_, test_y))
    print("R2-score: %.4f" % r2_score(test_y, test_y_))

    r2 = metric_cal.metric_r2(test_y, test_y_)
    mean_absolute = metric_cal.metric_mean_absolute(test_y, test_y_)
    mean_squared = metric_cal.metric_mean_squared(test_y, test_y_)
    root_mean_squared = metric_cal.metric_root_mean_squared(test_y, test_y_)
    mean_squared_log = metric_cal.metric_mean_squared_log(test_y, test_y_)
    mean_absolute_percentage = metric_cal.metric_mean_absolute_percentage(test_y, test_y_)
    media_absolute = metric_cal.metric_media_absolute(test_y, test_y_)
    max_error = metric_cal.metric_max_error(test_y, test_y_)

    # Feature importance chart
    importance_values = regr.feature_importances_
    fig = plotting.plot_importance_figure(importance_values, independent_variables)
    feature_imp_plot = plotting.figure_to_base64(fig)

    return_dict = {"r2_score": r2}
    return_dict.update({"mae": mean_absolute})
    return_dict.update({"mse": mean_squared})
    return_dict.update({"rmse": root_mean_squared})
    return_dict.update({"mean_squared_log": mean_squared_log})
    return_dict.update({"mean_absolute_percentage": mean_absolute_percentage})
    return_dict.update({"media_absolute": media_absolute})
    return_dict.update({"max_error": max_error})
    return_dict.update({"feature_imp_plot": feature_imp_plot})
    return_dict.update({"importance_values": importance_values})
    return_dict.update({"independent_variables": independent_variables})

    return return_dict


def kth_nearest_neighbors(path: str, target_variable: str, independent_variables: list, algo_params: dict):
    df = pd.read_csv(path)

    regr = neighbors.KNeighborsRegressor(**algo_params)
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.90, random_state=1)
    regr.fit(train_x, train_y)
    test_y_ = regr.predict(test_x)

    print('Coefficients: ', regr.score(test_x, test_y))
    print("scikit metrics mean absolute error: %.6f" % mean_absolute_error(test_y_, test_y))
    print("scikit metrics mean squared error: %.4f" % mean_squared_error(test_y_, test_y))
    print("R2-score: %.4f" % r2_score(test_y, test_y_))

    r2 = metric_cal.metric_r2(test_y, test_y_)
    mean_absolute = metric_cal.metric_mean_absolute(test_y, test_y_)
    mean_squared = metric_cal.metric_mean_squared(test_y, test_y_)
    root_mean_squared = metric_cal.metric_root_mean_squared(test_y, test_y_)
    mean_squared_log = metric_cal.metric_mean_squared_log(test_y, test_y_)
    mean_absolute_percentage = metric_cal.metric_mean_absolute_percentage(test_y, test_y_)
    media_absolute = metric_cal.metric_media_absolute(test_y, test_y_)
    max_error = metric_cal.metric_max_error(test_y, test_y_)

    return_dict = {"r2_score": r2}
    return_dict.update({"mae": mean_absolute})
    return_dict.update({"mse": mean_squared})
    return_dict.update({"rmse": root_mean_squared})
    return_dict.update({"mean_squared_log": mean_squared_log})
    return_dict.update({"mean_absolute_percentage": mean_absolute_percentage})
    return_dict.update({"media_absolute": media_absolute})
    return_dict.update({"max_error": max_error})

    return return_dict


def voting_regressor(path: str, target_variable: str, independent_variables: list, algo_params: dict):
    df = pd.read_csv(path)

    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.90, random_state=1)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    estimators = []
    estimator_counts = {'tree': 0, 'knn': 0,}

    for i, algo_i_params in algo_params.items():
        estimator_params = {key: value for key, value in algo_i_params["algo_params"].items() if
                            value is not None and value != ''}
        if algo_i_params['algo_id'] == 'decision_trees_regr':
            estimator_counts['tree'] += 1
            estimator = ('tree_' + str(estimator_counts['tree']), tree.DecisionTreeRegressor(**estimator_params))
        elif algo_i_params['algo_id'] == 'knn_regr':
            estimator_counts['knn'] += 1
            estimator = ('knn_' + str(estimator_counts['knn']), neighbors.KNeighborsRegressor(**estimator_params))
        else:
            raise Exception('invalid estimator algo selected for voting regression')
        estimators.append(estimator)

    reg = VotingRegressor(estimators)
    reg.fit(train_x, train_y)
    test_y_ = reg.predict(test_x)

    print('Coefficients: ', reg.score(test_x, test_y))
    print("scikit metrics mean absolute error: %.6f" % mean_absolute_error(test_y_, test_y))
    print("scikit metrics mean squared error: %.4f" % mean_squared_error(test_y_, test_y))
    print("R2-score: %.4f" % r2_score(test_y, test_y_))

    r2 = metric_cal.metric_r2(test_y, test_y_)
    mean_absolute = metric_cal.metric_mean_absolute(test_y, test_y_)
    mean_squared = metric_cal.metric_mean_squared(test_y, test_y_)
    root_mean_squared = metric_cal.metric_root_mean_squared(test_y, test_y_)
    mean_squared_log = metric_cal.metric_mean_squared_log(test_y, test_y_)
    mean_absolute_percentage = metric_cal.metric_mean_absolute_percentage(test_y, test_y_)
    media_absolute = metric_cal.metric_media_absolute(test_y, test_y_)
    max_error = metric_cal.metric_max_error(test_y, test_y_)

    return_dict = {"r2_score": r2}
    return_dict.update({"mae": mean_absolute})
    return_dict.update({"mse": mean_squared})
    return_dict.update({"rmse": root_mean_squared})
    return_dict.update({"mean_squared_log": mean_squared_log})
    return_dict.update({"mean_absolute_percentage": mean_absolute_percentage})
    return_dict.update({"media_absolute": media_absolute})
    return_dict.update({"max_error": max_error})

    return return_dict


def random_forest(path: str, target_variable: str, independent_variables: list, algo_params: dict):
    df = pd.read_csv(path)

    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.90, random_state=1)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    regr = RandomForestRegressor(**algo_params)
    regr.fit(train_x, train_y)
    test_y_ = regr.predict(test_x)

    print('Coefficients: ', regr.score(test_x, test_y))
    print("scikit metrics mean absolute error: %.6f" % mean_absolute_error(test_y_, test_y))
    print("scikit metrics mean squared error: %.4f" % mean_squared_error(test_y_, test_y))
    print("R2-score: %.4f" % r2_score(test_y, test_y_))

    r2 = metric_cal.metric_r2(test_y, test_y_)
    mean_absolute = metric_cal.metric_mean_absolute(test_y, test_y_)
    mean_squared = metric_cal.metric_mean_squared(test_y, test_y_)
    root_mean_squared = metric_cal.metric_root_mean_squared(test_y, test_y_)
    mean_squared_log = metric_cal.metric_mean_squared_log(test_y, test_y_)
    mean_absolute_percentage = metric_cal.metric_mean_absolute_percentage(test_y, test_y_)
    media_absolute = metric_cal.metric_media_absolute(test_y, test_y_)
    max_error = metric_cal.metric_max_error(test_y, test_y_)

    # Feature importance chart
    importance_values = regr.feature_importances_
    fig = plotting.plot_importance_figure(importance_values, independent_variables)
    feature_imp_plot = plotting.figure_to_base64(fig)

    return_dict = {"r2_score": r2}
    return_dict.update({"mae": mean_absolute})
    return_dict.update({"mse": mean_squared})
    return_dict.update({"rmse": root_mean_squared})
    return_dict.update({"mean_squared_log": mean_squared_log})
    return_dict.update({"mean_absolute_percentage": mean_absolute_percentage})
    return_dict.update({"media_absolute": media_absolute})
    return_dict.update({"max_error": max_error})
    return_dict.update({"feature_imp_plot": feature_imp_plot})
    return_dict.update({"importance_values": importance_values})
    return_dict.update({"independent_variables": independent_variables})

    return return_dict


def bagging_regr(path: str, target_variable: str, independent_variables: list, algo_params: dict):
    df = pd.read_csv(path)

    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.90, random_state=1)

    estimator_str = algo_params.pop('estimator')
    estimator_params = algo_params.pop('estimator_params')

    algo_params.update({'max_samples': float(algo_params['max_samples'])})
    algo_params.update({'max_features': float(algo_params['max_features'])})

    model_mapping = {
        "knn_regr": neighbors.KNeighborsRegressor,
        "decision_trees_regr": tree.DecisionTreeRegressor,
    }

    estimator = model_mapping.get(estimator_str)(**estimator_params) if estimator_str in model_mapping else None
    regr = BaggingRegressor(estimator, **algo_params)
    regr.fit(train_x, train_y)
    test_y_ = regr.predict(test_x)

    print('Coefficients: ', regr.score(test_x, test_y))
    print("scikit metrics mean absolute error: %.6f" % mean_absolute_error(test_y_, test_y))
    print("scikit metrics mean squared error: %.4f" % mean_squared_error(test_y_, test_y))
    print("R2-score: %.4f" % r2_score(test_y, test_y_))

    r2 = metric_cal.metric_r2(test_y, test_y_)
    mean_absolute = metric_cal.metric_mean_absolute(test_y, test_y_)
    mean_squared = metric_cal.metric_mean_squared(test_y, test_y_)
    root_mean_squared = metric_cal.metric_root_mean_squared(test_y, test_y_)
    mean_squared_log = metric_cal.metric_mean_squared_log(test_y, test_y_)
    mean_absolute_percentage = metric_cal.metric_mean_absolute_percentage(test_y, test_y_)
    media_absolute = metric_cal.metric_media_absolute(test_y, test_y_)
    max_error = metric_cal.metric_max_error(test_y, test_y_)

    return_dict = {"r2_score": r2}
    return_dict.update({"mae": mean_absolute})
    return_dict.update({"mse": mean_squared})
    return_dict.update({"rmse": root_mean_squared})
    return_dict.update({"mean_squared_log": mean_squared_log})
    return_dict.update({"mean_absolute_percentage": mean_absolute_percentage})
    return_dict.update({"media_absolute": media_absolute})
    return_dict.update({"max_error": max_error})

    return return_dict
