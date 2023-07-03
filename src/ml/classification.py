from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import RadiusNeighborsClassifier

def decision_trees_classification(path: str, target_variable: str, independent_variables: list):
    df = pd.read_csv(path)
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, random_state=0)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    clf = tree.DecisionTreeClassifier()
    clf.fit(train_x, train_y)
    test_y_= clf.predict(test_x)
    print("---------------------------------------------------------------------")
    print(test_y_)
    return

def random_forest_classification(path: str, target_variable: str, independent_variables: list):
    df = pd.read_csv(path)
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, random_state=0)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(train_x, train_y)
    test_y_= clf.predict(test_x)
    print("---------------------------------------------------------------------")
    print(test_y_)
    return

def k_nearest_neighbor_classification(path: str, target_variable: str, independent_variables: list):

    df = pd.read_csv(path)
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.10, random_state=0)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    clf = RadiusNeighborsClassifier(radius=1.0)
    clf.fit(train_x, train_y)
    test_y_= clf.predict(test_x)
    print("---------------------------------------------------------------------")
    print(test_y_)
    return

path = "C:\Default MLDATA CLS (normalized).csv"
target_variable = 'risk_levels'
independent_variables = ['SMOKING','OBESITY']
k_nearest_neighbor_classification(path,target_variable,independent_variables)

