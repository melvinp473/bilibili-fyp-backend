import pandas as pd
import sklearn.linear_model as linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import tree, neighbors
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
import numpy as np
from flask import Flask, Response, request, Blueprint, make_response, jsonify
from sklearn import svm
from src.ml import metric_cal

def linear_regression(path: str, target_variable: str, independent_variables: list):
    df = pd.read_csv(path)
    regr = linear_model.LinearRegression()
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, random_state=0)
    regr.fit(train_x, train_y)
    test_y_ = regr.predict(test_x)

    return_dict = {"Coefficients": regr.coef_.tolist()[0], "Intercept": regr.intercept_.tolist()[0]}
    r2 = metric_cal.metric_r2(test_y,test_y_)
    mean_absolute = metric_cal.metric_mean_absolute(test_y,test_y_)
    mean_squared = metric_cal.metric_mean_squared(test_y,test_y_)
    mean_squared_log = metric_cal.metric_mean_squared_log(test_y,test_y_)
    mean_absolute_percentage = metric_cal.metric_mean_absolute_percentage(test_y,test_y_)
    media_absolute = metric_cal.metric_media_absolute(test_y,test_y_)
    max_error = metric_cal.metric_max_error(test_y,test_y_)

    return_dict.update({"r2_score": r2})
    return_dict.update({"mae": mean_absolute})
    return_dict.update({"mse": mean_squared})
    return_dict.update({"mean_squared_log": mean_squared_log})
    return_dict.update({"mean_absolute_percentage": mean_absolute_percentage})
    return_dict.update({"media_absolute": media_absolute})
    return_dict.update({"max_error": max_error})
    return return_dict

def support_vector_machines(path: str, target_variable: str, independent_variables: list):
    df = pd.read_csv(path)
    x = df[independent_variables]
    y = df[[target_variable]]
    regr = svm.SVR(kernel="linear", C=100, gamma="auto")
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, random_state=0)
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
    r2 = metric_cal.metric_r2(test_y,test_y_)
    mean_absolute = metric_cal.metric_mean_absolute(test_y,test_y_)
    mean_squared = metric_cal.metric_mean_squared(test_y,test_y_)
    mean_squared_log = metric_cal.metric_mean_squared_log(test_y,test_y_)
    mean_absolute_percentage = metric_cal.metric_mean_absolute_percentage(test_y,test_y_)
    media_absolute = metric_cal.metric_media_absolute(test_y,test_y_)
    max_error = metric_cal.metric_max_error(test_y,test_y_)

    return_dict.update({"r2_score": r2})
    return_dict.update({"mae": mean_absolute})
    return_dict.update({"mse": mean_squared})
    return_dict.update({"mean_squared_log": mean_squared_log})
    return_dict.update({"mean_absolute_percentage": mean_absolute_percentage})
    return_dict.update({"media_absolute": media_absolute})
    return_dict.update({"max_error": max_error})

    return return_dict


def decision_trees(path: str, target_variable: str, independent_variables: list, additional_params: dict):
    df = pd.read_csv(path)

    max_depth = additional_params['max_depth']
    regr = tree.DecisionTreeRegressor(max_depth=max_depth)
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, random_state=0)
    regr.fit(train_x, train_y)
    test_y_ = regr.predict(test_x)

    print('Coefficients: ', regr.score(test_x, test_y))
    print("scikit metrics mean absolute error: %.6f" % mean_absolute_error(test_y_, test_y))
    print("scikit metrics mean squared error: %.4f" % mean_squared_error(test_y_, test_y))
    print("R2-score: %.4f" % r2_score(test_y, test_y_))


    r2 = metric_cal.metric_r2(test_y,test_y_)
    mean_absolute = metric_cal.metric_mean_absolute(test_y,test_y_)
    mean_squared = metric_cal.metric_mean_squared(test_y,test_y_)
    mean_squared_log = metric_cal.metric_mean_squared_log(test_y,test_y_)
    mean_absolute_percentage = metric_cal.metric_mean_absolute_percentage(test_y,test_y_)
    media_absolute = metric_cal.metric_media_absolute(test_y,test_y_)
    max_error = metric_cal.metric_max_error(test_y,test_y_)

    return_dict = {"r2_score": r2}
    return_dict.update({"mae": mean_absolute})
    return_dict.update({"mse": mean_squared})
    return_dict.update({"mean_squared_log": mean_squared_log})
    return_dict.update({"mean_absolute_percentage": mean_absolute_percentage})
    return_dict.update({"media_absolute": media_absolute})
    return_dict.update({"max_error": max_error})



    return return_dict


def kth_nearest_neighbors(path: str, target_variable: str, independent_variables: list, additional_params: dict):
    df = pd.read_csv(path)

    n_neighbours = additional_params['neighbours_count']
    regr = neighbors.KNeighborsRegressor(n_neighbours)
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, random_state=0)
    regr.fit(train_x, train_y)
    test_y_ = regr.predict(test_x)

    print('Coefficients: ', regr.score(test_x, test_y))
    print("scikit metrics mean absolute error: %.6f" % mean_absolute_error(test_y_, test_y))
    print("scikit metrics mean squared error: %.4f" % mean_squared_error(test_y_, test_y))
    print("R2-score: %.4f" % r2_score(test_y, test_y_))

    r2 = metric_cal.metric_r2(test_y,test_y_)
    mean_absolute = metric_cal.metric_mean_absolute(test_y,test_y_)
    mean_squared = metric_cal.metric_mean_squared(test_y,test_y_)
    mean_squared_log = metric_cal.metric_mean_squared_log(test_y,test_y_)
    mean_absolute_percentage = metric_cal.metric_mean_absolute_percentage(test_y,test_y_)
    media_absolute = metric_cal.metric_media_absolute(test_y,test_y_)
    max_error = metric_cal.metric_max_error(test_y,test_y_)

    return_dict = {"r2_score": r2}
    return_dict.update({"mae": mean_absolute})
    return_dict.update({"mse": mean_squared})
    return_dict.update({"mean_squared_log": mean_squared_log})
    return_dict.update({"mean_absolute_percentage": mean_absolute_percentage})
    return_dict.update({"media_absolute": media_absolute})
    return_dict.update({"max_error": max_error})

    return return_dict


"the regression method can add more, this is only the test type"
def voting_regressor(path: str, target_variable: str, independent_variables: list):
    df = pd.read_csv(path)


    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, random_state=0)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    reg1 = GradientBoostingRegressor(random_state=1)
    reg2 = RandomForestRegressor(random_state=1)
    reg3 = linear_model.LinearRegression()
    ereg = VotingRegressor(estimators=[('gb', reg1), ('rf', reg2), ('lr', reg3)])
    ereg = ereg.fit(train_x, train_y)
    test_y_ = ereg.predict(test_x)

    print('Coefficients: ', ereg.score(test_x, test_y))
    print("scikit metrics mean absolute error: %.6f" % mean_absolute_error(test_y_, test_y))
    print("scikit metrics mean squared error: %.4f" % mean_squared_error(test_y_, test_y))
    print("R2-score: %.4f" % r2_score(test_y, test_y_))

    r2 = metric_cal.metric_r2(test_y,test_y_)
    mean_absolute = metric_cal.metric_mean_absolute(test_y,test_y_)
    mean_squared = metric_cal.metric_mean_squared(test_y,test_y_)
    mean_squared_log = metric_cal.metric_mean_squared_log(test_y,test_y_)
    mean_absolute_percentage = metric_cal.metric_mean_absolute_percentage(test_y,test_y_)
    media_absolute = metric_cal.metric_media_absolute(test_y,test_y_)
    max_error = metric_cal.metric_max_error(test_y,test_y_)

    return_dict = {"r2_score": r2}
    return_dict.update({"mae": mean_absolute})
    return_dict.update({"mse": mean_squared})
    return_dict.update({"mean_squared_log": mean_squared_log})
    return_dict.update({"mean_absolute_percentage": mean_absolute_percentage})
    return_dict.update({"media_absolute": media_absolute})
    return_dict.update({"max_error": max_error})

    return return_dict

