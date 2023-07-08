from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.naive_bayes import GaussianNB


def decision_trees_classification(path: str, target_variable: str, independent_variables: list, algo_params: dict):
    df = pd.read_csv(path)
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, random_state=0)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    clf = tree.DecisionTreeClassifier(**algo_params)
    clf.fit(train_x, train_y)
    test_y_ = clf.predict(test_x)
    test_y_auc = clf.predict_proba(test_x)[:, 1]

    rfc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)
    plt.show()

    print("---------------------------------------------------------------------")
    print(test_y_)
    auc = roc_auc_score(test_y, test_y_auc)
    print("AUC-ROC:", auc)
    precision = precision_score(test_y, test_y_)
    print("Precision:", precision)
    accuracy = accuracy_score(test_y, test_y_.round())
    print("Accuracy:", accuracy)
    return_dict = {"auc": auc}
    return_dict.update({"precision": precision})
    return_dict.update({"accuracy": accuracy})
    return return_dict


def random_forest_classification(path: str, target_variable: str, independent_variables: list, algo_params: dict):
    df = pd.read_csv(path)
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, random_state=0)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    clf = RandomForestClassifier(**algo_params)
    clf.fit(train_x, train_y)
    test_y_ = clf.predict(test_x)
    test_y_auc = clf.predict_proba(test_x)[:, 1]

    rfc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)
    plt.show()

    print("---------------------------------------------------------------------")
    print(test_y_)
    auc = roc_auc_score(test_y, test_y_auc)
    print("AUC-ROC:", auc)
    precision = precision_score(test_y, test_y_)
    print("Precision:", precision)
    accuracy = accuracy_score(test_y, test_y_.round())
    print("Accuracy:", accuracy)
    return_dict = {"auc": auc}
    return_dict.update({"precision": precision})
    return_dict.update({"accuracy": accuracy})
    return return_dict


def k_nearest_neighbor_classification(path: str, target_variable: str, independent_variables: list, algo_params: dict):
    df = pd.read_csv(path)
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, random_state=0)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    clf = RadiusNeighborsClassifier(**algo_params)
    clf.fit(train_x, train_y)

    test_y_ = clf.predict(test_x)
    test_y_auc = clf.predict_proba(test_x)[:, 1]

    rfc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)
    plt.show()

    print(test_y)
    print("---------------------------------------------------------------------")
    print(test_y_)
    auc = roc_auc_score(test_y, test_y_auc)
    print("AUC-ROC:", auc)

    precision = precision_score(test_y, test_y_)
    print("Precision:", precision)
    accuracy = accuracy_score(test_y, test_y_.round())
    print("Accuracy:", accuracy)
    return_dict = {"auc": auc}
    return_dict.update({"precision": precision})
    return_dict.update({"accuracy": accuracy})
    return return_dict


def gaussian_naive_bayes(path: str, target_variable: str, independent_variables: list):
    df = pd.read_csv(path)
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, random_state=0)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    clf = GaussianNB()
    clf.fit(train_x, train_y)
    test_y_ = clf.predict(test_x)
    test_y_auc = clf.predict_proba(test_x)[:, 1]

    rfc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)
    plt.show()

    print("---------------------------------------------------------------------")
    print(test_y_)
    auc = roc_auc_score(test_y, test_y_auc)
    print("AUC-ROC:", auc)
    precision = precision_score(test_y, test_y_)
    print("Precision:", precision)
    accuracy = accuracy_score(test_y, test_y_.round())
    print("Accuracy:", accuracy)
    return_dict = {"auc": auc}
    return_dict.update({"precision": precision})
    return_dict.update({"accuracy": accuracy})
    return return_dict


def voting_cls(path: str, target_variable: str, independent_variables: list, algo_params: dict):
    df = pd.read_csv(path)

    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, random_state=0)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    estimators = []

    voting_params = {key: value for key, value in algo_params["voting_params"].items() if
                     value is not None and value != ''}

    estimator_counts = {'tree': 0, 'knn': 0, 'random_forest': 0, 'gaussian_nb': 0}

    for estimator_i_params in algo_params["estimators_list"]:
        estimator_params = {key: value for key, value in estimator_i_params["algo_params"].items() if
                            value is not None and value != ''}
        if estimator_i_params['algo_id'] == 'decision_trees_cls':
            estimator_counts['tree'] += 1
            estimator = ('tree_' + str(estimator_counts['tree']), tree.DecisionTreeClassifier(**estimator_params))
        elif estimator_i_params['algo_id'] == 'knn_cls':
            estimator_counts['knn'] += 1
            estimator = ('knn_' + str(estimator_counts['knn']), RadiusNeighborsClassifier(**estimator_params))
        elif estimator_i_params['algo_id'] == 'random_forest_cls':
            estimator_counts['random_forest'] += 1
            estimator = ('random_forest_' + str(estimator_counts['random_forest']), RandomForestClassifier(**estimator_params))
        elif estimator_i_params['algo_id'] == 'gauss_naive_bayes_cls':
            estimator_counts['gaussian_nb'] += 1
            estimator = ('gaussian_nb_' + str(estimator_counts['gaussian_nb']), GaussianNB(**estimator_params))
        else:
            raise Exception('invalid estimator algo selected for voting classification')
        estimators.append(estimator)

    clf = VotingClassifier(estimators, **voting_params)
    clf.fit(train_x, train_y)
    test_y_ = clf.predict(test_x)
    test_y_auc = clf.predict_proba(test_x)[:, 1]

    rfc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)
    plt.show()

    print("---------------------------------------------------------------------")
    print(test_y_)
    auc = roc_auc_score(test_y, test_y_auc)
    print("AUC-ROC:", auc)
    precision = precision_score(test_y, test_y_)
    print("Precision:", precision)
    accuracy = accuracy_score(test_y, test_y_.round())
    print("Accuracy:", accuracy)
    return_dict = {
        "auc": auc,
        "precision": precision,
        "accuracy": accuracy,
    }
    return return_dict

# if __name__ == '__main__':
#     results = gaussian_naive_bayes(
#         'C:\\Users\kohji\PycharmProjects\\bilibili-fyp-backend\\test\Default MLDATA CLS (normalized)(1).csv',
#         'risk_levels',
#         ['SMOKING', 'OBESITY', 'DRINKING', 'UNEMPLOYMENT', 'DIABETES', 'MENTAL_DISEASE', 'PSYCHOLOGICAL_DISTRESS',
#          'HYPERTENSION'])
#
#     print(results)


# path = "C:\Default MLDATA CLS (normalized).csv"
# target_variable = 'risk_levels'
# independent_variables = ['SMOKING','OBESITY']
# random_forest_classification(path,target_variable,independent_variables)
