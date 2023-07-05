from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import roc_auc_score, precision_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay


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
    test_y_= clf.predict(test_x)
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
    test_y_= clf.predict(test_x)
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

    test_y_= clf.predict(test_x)
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

# path = "C:\Default MLDATA CLS (normalized).csv"
# target_variable = 'risk_levels'
# independent_variables = ['SMOKING','OBESITY']
# random_forest_classification(path,target_variable,independent_variables)
