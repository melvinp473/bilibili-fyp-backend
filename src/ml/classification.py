from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from . import plotting


def decision_trees_classification(path: str, target_variable: str, independent_variables: list, algo_params: dict):
    df = pd.read_csv(path)
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.90, random_state=1)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    clf = tree.DecisionTreeClassifier(**algo_params)
    clf.fit(train_x, train_y)
    test_y_ = clf.predict(test_x)
    test_y_auc = clf.predict_proba(test_x)[:, 1]

    print("---------------------------------------------------------------------")

    # Feature importance chart
    importance_values = clf.feature_importances_
    fig = plotting.plot_importance_figure(importance_values, independent_variables)
    feature_imp_plot = plotting.figure_to_base64(fig)

    # ROC Curve
    roc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)
    fig_roc = roc_disp.figure_
    roc_plot = plotting.figure_to_base64(fig_roc)

    # Precision Recall Curve
    precision, recall, _ = precision_recall_curve(test_y, test_y_)
    pr_disp = PrecisionRecallDisplay.from_estimator(clf, test_x, test_y)
    fig_pr = pr_disp.figure_
    pr_plot = plotting.figure_to_base64(fig_pr)

    # Confusion Matrix
    cm = confusion_matrix(test_y, test_y_)
    cm_disp = ConfusionMatrixDisplay.from_estimator(clf, test_x, test_y)
    fig_cm = cm_disp.figure_
    cm_plot = plotting.figure_to_base64(fig_cm)

    auc = roc_auc_score(test_y, test_y_auc)
    formatted_auc = "{:.4f}".format(auc)
    precision = precision_score(test_y, test_y_)
    formatted_precision = "{:.4f}".format(precision)
    accuracy = accuracy_score(test_y, test_y_.round())
    formatted_accuracy = "{:.4f}".format(accuracy)
    recall = recall_score(test_y, test_y_, average='weighted')
    formatted_recall = "{:.4f}".format(recall)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    formatted_specificity = "{:.4f}".format(specificity)
    f1 = f1_score(test_y, test_y_, average='weighted')
    formatted_f1 = "{:.4f}".format(f1)

    return_dict = {"auc": formatted_auc}
    return_dict.update({"precision": formatted_precision})
    return_dict.update({"accuracy": formatted_accuracy})
    return_dict.update({"recall": formatted_recall})
    return_dict.update({"specificity": formatted_specificity})
    return_dict.update({"f1": formatted_f1})
    return_dict.update({"roc_plot": roc_plot})
    return_dict.update({"pr_plot": pr_plot})
    return_dict.update({"cm_plot": cm_plot})
    return_dict.update({"feature_imp_plot": feature_imp_plot})
    return return_dict


def random_forest_classification(path: str, target_variable: str, independent_variables: list, algo_params: dict):
    df = pd.read_csv(path)
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.90, random_state=1)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    clf = RandomForestClassifier(**algo_params)
    clf.fit(train_x, train_y)
    test_y_ = clf.predict(test_x)
    test_y_auc = clf.predict_proba(test_x)[:, 1]

    print("---------------------------------------------------------------------")

    # Feature importance chart
    importance_values = clf.feature_importances_
    fig = plotting.plot_importance_figure(importance_values, independent_variables)
    feature_imp_plot = plotting.figure_to_base64(fig)

    # ROC Curve
    roc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)
    fig_roc = roc_disp.figure_
    roc_plot = plotting.figure_to_base64(fig_roc)

    # Precision Recall Curve
    precision, recall, _ = precision_recall_curve(test_y, test_y_)
    pr_disp = PrecisionRecallDisplay.from_estimator(clf, test_x, test_y)
    fig_pr = pr_disp.figure_
    pr_plot = plotting.figure_to_base64(fig_pr)

    # Confusion Matrix
    cm = confusion_matrix(test_y, test_y_)
    cm_disp = ConfusionMatrixDisplay.from_estimator(clf, test_x, test_y)
    fig_cm = cm_disp.figure_
    cm_plot = plotting.figure_to_base64(fig_cm)

    auc = roc_auc_score(test_y, test_y_auc)
    formatted_auc = "{:.4f}".format(auc)
    precision = precision_score(test_y, test_y_)
    formatted_precision = "{:.4f}".format(precision)
    accuracy = accuracy_score(test_y, test_y_.round())
    formatted_accuracy = "{:.4f}".format(accuracy)
    recall = recall_score(test_y, test_y_, average='weighted')
    formatted_recall = "{:.4f}".format(recall)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    formatted_specificity = "{:.4f}".format(specificity)
    f1 = f1_score(test_y, test_y_, average='weighted')
    formatted_f1 = "{:.4f}".format(f1)

    return_dict = {"auc": formatted_auc}
    return_dict.update({"precision": formatted_precision})
    return_dict.update({"accuracy": formatted_accuracy})
    return_dict.update({"recall": formatted_recall})
    return_dict.update({"specificity": formatted_specificity})
    return_dict.update({"f1": formatted_f1})
    return_dict.update({"roc_plot": roc_plot})
    return_dict.update({"pr_plot": pr_plot})
    return_dict.update({"cm_plot": cm_plot})
    # return_dict.update({"feature_imp_plot": feature_imp_plot})
    return return_dict


def k_nearest_neighbor_classification(path: str, target_variable: str, independent_variables: list, algo_params: dict):
    df = pd.read_csv(path)
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.90, random_state=1)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    clf = RadiusNeighborsClassifier(**algo_params)
    clf.fit(train_x, train_y)

    test_y_ = clf.predict(test_x)
    test_y_auc = clf.predict_proba(test_x)[:, 1]

    print("---------------------------------------------------------------------")

    # ROC Curve
    roc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)
    fig_roc = roc_disp.figure_
    roc_plot = plotting.figure_to_base64(fig_roc)

    # Precision Recall Curve
    precision, recall, _ = precision_recall_curve(test_y, test_y_)
    pr_disp = PrecisionRecallDisplay.from_estimator(clf, test_x, test_y)
    fig_pr = pr_disp.figure_
    pr_plot = plotting.figure_to_base64(fig_pr)

    # Confusion Matrix
    cm = confusion_matrix(test_y, test_y_)
    cm_disp = ConfusionMatrixDisplay.from_estimator(clf, test_x, test_y)
    fig_cm = cm_disp.figure_
    cm_plot = plotting.figure_to_base64(fig_cm)

    auc = roc_auc_score(test_y, test_y_auc)
    formatted_auc = "{:.4f}".format(auc)
    precision = precision_score(test_y, test_y_)
    formatted_precision = "{:.4f}".format(precision)
    accuracy = accuracy_score(test_y, test_y_.round())
    formatted_accuracy = "{:.4f}".format(accuracy)
    recall = recall_score(test_y, test_y_, average='weighted')
    formatted_recall = "{:.4f}".format(recall)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    formatted_specificity = "{:.4f}".format(specificity)
    f1 = f1_score(test_y, test_y_, average='weighted')
    formatted_f1 = "{:.4f}".format(f1)

    return_dict = {"auc": formatted_auc}
    return_dict.update({"precision": formatted_precision})
    return_dict.update({"accuracy": formatted_accuracy})
    return_dict.update({"recall": formatted_recall})
    return_dict.update({"specificity": formatted_specificity})
    return_dict.update({"f1": formatted_f1})
    return_dict.update({"roc_plot": roc_plot})
    return_dict.update({"pr_plot": pr_plot})
    return_dict.update({"cm_plot": cm_plot})
    # return_dict.update({"feature_imp_plot": feature_imp_plot})
    return return_dict


def gaussian_naive_bayes(path: str, target_variable: str, independent_variables: list):
    df = pd.read_csv(path)
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.90, random_state=1)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    clf = GaussianNB()
    clf.fit(train_x, train_y)
    test_y_ = clf.predict(test_x)
    test_y_auc = clf.predict_proba(test_x)[:, 1]

    print("---------------------------------------------------------------------")

    # ROC Curve
    roc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)
    fig_roc = roc_disp.figure_
    roc_plot = plotting.figure_to_base64(fig_roc)

    # Precision Recall Curve
    precision, recall, _ = precision_recall_curve(test_y, test_y_)
    pr_disp = PrecisionRecallDisplay.from_estimator(clf, test_x, test_y)
    fig_pr = pr_disp.figure_
    pr_plot = plotting.figure_to_base64(fig_pr)

    # Confusion Matrix
    cm = confusion_matrix(test_y, test_y_)
    cm_disp = ConfusionMatrixDisplay.from_estimator(clf, test_x, test_y)
    fig_cm = cm_disp.figure_
    cm_plot = plotting.figure_to_base64(fig_cm)

    auc = roc_auc_score(test_y, test_y_auc)
    formatted_auc = "{:.4f}".format(auc)
    precision = precision_score(test_y, test_y_)
    formatted_precision = "{:.4f}".format(precision)
    accuracy = accuracy_score(test_y, test_y_.round())
    formatted_accuracy = "{:.4f}".format(accuracy)
    recall = recall_score(test_y, test_y_, average='weighted')
    formatted_recall = "{:.4f}".format(recall)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    formatted_specificity = "{:.4f}".format(specificity)
    f1 = f1_score(test_y, test_y_, average='weighted')
    formatted_f1 = "{:.4f}".format(f1)

    return_dict = {"auc": formatted_auc}
    return_dict.update({"precision": formatted_precision})
    return_dict.update({"accuracy": formatted_accuracy})
    return_dict.update({"recall": formatted_recall})
    return_dict.update({"specificity": formatted_specificity})
    return_dict.update({"f1": formatted_f1})
    return_dict.update({"roc_plot": roc_plot})
    return_dict.update({"pr_plot": pr_plot})
    return_dict.update({"cm_plot": cm_plot})
    # return_dict.update({"feature_imp_plot": feature_imp_plot})
    return return_dict

def voting_cls(path: str, target_variable: str, independent_variables: list, algo_params: dict):
    df = pd.read_csv(path)

    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=0.90, random_state=1)
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

    print("---------------------------------------------------------------------")

    # ROC Curve
    roc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)
    fig_roc = roc_disp.figure_
    roc_plot = plotting.figure_to_base64(fig_roc)

    # Precision Recall Curve
    precision, recall, _ = precision_recall_curve(test_y, test_y_)
    pr_disp = PrecisionRecallDisplay.from_estimator(clf, test_x, test_y)
    fig_pr = pr_disp.figure_
    pr_plot = plotting.figure_to_base64(fig_pr)

    # Confusion Matrix
    cm = confusion_matrix(test_y, test_y_)
    cm_disp = ConfusionMatrixDisplay.from_estimator(clf, test_x, test_y)
    fig_cm = cm_disp.figure_
    cm_plot = plotting.figure_to_base64(fig_cm)

    auc = roc_auc_score(test_y, test_y_auc)
    formatted_auc = "{:.4f}".format(auc)
    precision = precision_score(test_y, test_y_)
    formatted_precision = "{:.4f}".format(precision)
    accuracy = accuracy_score(test_y, test_y_.round())
    formatted_accuracy = "{:.4f}".format(accuracy)
    recall = recall_score(test_y, test_y_, average='weighted')
    formatted_recall = "{:.4f}".format(recall)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    formatted_specificity = "{:.4f}".format(specificity)
    f1 = f1_score(test_y, test_y_, average='weighted')
    formatted_f1 = "{:.4f}".format(f1)

    return_dict = {"auc": formatted_auc}
    return_dict.update({"precision": formatted_precision})
    return_dict.update({"accuracy": formatted_accuracy})
    return_dict.update({"recall": formatted_recall})
    return_dict.update({"specificity": formatted_specificity})
    return_dict.update({"f1": formatted_f1})
    return_dict.update({"roc_plot": roc_plot})
    return_dict.update({"pr_plot": pr_plot})
    return_dict.update({"cm_plot": cm_plot})
    # return_dict.update({"feature_imp_plot": feature_imp_plot})
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
