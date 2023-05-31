import pandas as pd
import sklearn.linear_model as linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn import tree, neighbors
import numpy as np
from flask import Flask, Response, request, Blueprint, make_response, jsonify
from sklearn import svm


def linear_regression(path: str, selected_attributes: list):
    df = pd.read_csv(path)
    regr = linear_model.LinearRegression()
    x = df[selected_attributes]
    y = df[["STROKE"]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, random_state=0)
    regr.fit(train_x, train_y)
    test_y_ = regr.predict(test_x)

    # print('Coefficients: ', regr.coef_)
    # print('Intercept: ', regr.intercept_)
    # print("scikit metrics mean absolute error: %.6f" % mean_absolute_error(test_y_, test_y))
    # print("scikit metrics mean squared error: %.4f" % mean_squared_error(test_y_, test_y))
    # print("Residual sum of squares (MSE): %.4f" % np.mean((test_y_ - test_y) ** 2))
    # print("R2-score: %.4f" % r2_score(test_y, test_y_))
    return_dict = {"Coefficients": regr.coef_.tolist()[0], "Intercept": regr.intercept_.tolist()[0]}
    return_dict.update({"mae": mean_absolute_error(test_y_, test_y)})
    return_dict.update({"mse": mean_squared_error(test_y_, test_y)})
    return_dict.update({"r2_score": r2_score(test_y, test_y_)})
    return_dict = {'data': return_dict}
    json_data = jsonify(return_dict)
    return json_data

def support_vector_machines(path: str, selected_attributes: list):
    df = pd.read_csv(path)
    x = df[selected_attributes]
    y = df[["STROKE"]]
    regr = svm.SVR(kernel="linear", C=100, gamma="auto")
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, random_state=0)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()
    print(train_x)
    print(train_y)

    regr.fit(train_x, train_y)
    test_y_ = regr.predict(test_x)
    print('Coefficients: ', regr.coef_)
    print('Intercept: ', regr.intercept_)
    print("scikit metrics mean absolute error: %.6f" % mean_absolute_error(test_y_, test_y))
    print("scikit metrics mean squared error: %.4f" % mean_squared_error(test_y_, test_y))
    print("Residual sum of squares (MSE): %.4f" % np.mean((test_y_ - test_y) ** 2))
    print("R2-score: %.4f" % r2_score(test_y, test_y_))

    return_dict = {"Coefficients": regr.coef_.tolist()[0], "Intercept": regr.intercept_.tolist()[0]}
    return_dict.update({"mae": mean_absolute_error(test_y_, test_y)})
    return_dict.update({"mse": mean_squared_error(test_y_, test_y)})
    return_dict.update({"r2_score": r2_score(test_y, test_y_)})
    return_dict = {'data': return_dict}
    json_data = jsonify(return_dict)

    return json_data

def decision_trees(path: str, selected_attributes: list):
    df = pd.read_csv(path)
    regr = tree.DecisionTreeRegressor()
    x = df[selected_attributes]
    y = df[["STROKE"]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, random_state=0)
    # test_x = test_x.to_numpy().ravel()
    # train_x = train_x.to_numpy().ravel()
    # test_y = test_y.to_numpy().ravel()
    # train_y = train_y.to_numpy().ravel()
    regr.fit(train_x, train_y)
    test_y_ = regr.predict(test_x)

    print('Coefficients: ', regr.score(test_x, test_y))
    # print('Intercept: ', regr.intercept_)
    print("scikit metrics mean absolute error: %.6f" % mean_absolute_error(test_y_, test_y))
    print("scikit metrics mean squared error: %.4f" % mean_squared_error(test_y_, test_y))
    # print("Residual sum of squares (MSE): %.4f" % np.mean((test_y_ - test_y) ** 2))
    print("R2-score: %.4f" % r2_score(test_y, test_y_))
    return_dict = {"Coefficients": regr.score(test_x, test_y)}
    return_dict.update({"mae": mean_absolute_error(test_y_, test_y)})
    return_dict.update({"mse": mean_squared_error(test_y_, test_y)})
    return_dict.update({"r2_score": r2_score(test_y, test_y_)})
    return_dict = {'data': return_dict}
    json_data = jsonify(return_dict)
    return json_data


def kth_nearest_neighbors(path: str, selected_attributes: list):
    df = pd.read_csv(path)
    regr = neighbors.KNeighborsRegressor(11)
    x = df[selected_attributes]
    y = df[["STROKE"]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, random_state=0)
    regr.fit(train_x, train_y)
    test_y_ = regr.predict(test_x)

    print('Coefficients: ', regr.score(test_x, test_y))
    # print('Intercept: ', regr.intercept_)
    print("scikit metrics mean absolute error: %.6f" % mean_absolute_error(test_y_, test_y))
    print("scikit metrics mean squared error: %.4f" % mean_squared_error(test_y_, test_y))
    # print("Residual sum of squares (MSE): %.4f" % np.mean((test_y_ - test_y) ** 2))
    print("R2-score: %.4f" % r2_score(test_y, test_y_))
    return_dict = {"Coefficients": regr.score(test_x, test_y)}
    return_dict.update({"mae": mean_absolute_error(test_y_, test_y)})
    return_dict.update({"mse": mean_squared_error(test_y_, test_y)})
    return_dict.update({"r2_score": r2_score(test_y, test_y_)})
    return_dict = {'data': return_dict}
    json_data = jsonify(return_dict)
    return json_data

