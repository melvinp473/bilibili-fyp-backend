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


def decision_trees_classification(dataframe, target_variable: str, independent_variables: list, algo_params: dict):

    df = dataframe


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
    fig.suptitle("Feature importances (based on Gini importance)")
    fig.tight_layout()
    feature_imp_plot = plotting.figure_to_base64(fig)

    # ROC Curve
    roc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)
    fig_roc = roc_disp.figure_
    fig_roc.suptitle("ROC curve")
    fig_roc.tight_layout()
    roc_plot = plotting.figure_to_base64(fig_roc)

    # Precision Recall Curve
    # precision, recall, _ = precision_recall_curve(test_y, test_y_)
    pr_disp = PrecisionRecallDisplay.from_estimator(clf, test_x, test_y)
    fig_pr = pr_disp.figure_
    fig_pr.suptitle("Precision Recall curve")
    fig_pr.tight_layout()
    pr_plot = plotting.figure_to_base64(fig_pr)

    # Confusion Matrix
    cm = confusion_matrix(test_y, test_y_)
    cm_disp = ConfusionMatrixDisplay.from_estimator(clf, test_x, test_y)
    fig_cm = cm_disp.figure_
    fig_cm.suptitle("Confusion Matrix")
    fig_cm.tight_layout()
    cm_plot = plotting.figure_to_base64(fig_cm)

    auc = roc_auc_score(test_y, test_y_auc)
    precision = precision_score(test_y, test_y_)
    accuracy = accuracy_score(test_y, test_y_.round())
    recall = recall_score(test_y, test_y_, average='weighted')
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    f1 = f1_score(test_y, test_y_, average='weighted')

    return_dict = {"auc": auc}
    return_dict.update({"precision": precision})
    return_dict.update({"accuracy": accuracy})
    return_dict.update({"recall": recall})
    return_dict.update({"specificity": specificity})
    return_dict.update({"f1": f1})
    return_dict.update({"roc_plot": roc_plot})
    return_dict.update({"pr_plot": pr_plot})
    return_dict.update({"cm_plot": cm_plot})
    return_dict.update({"feature_imp_plot": feature_imp_plot})
    return return_dict


def random_forest_classification(dataframe, target_variable: str, independent_variables: list, algo_params: dict):
    df = dataframe
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
    fig.suptitle("Feature importances (based on Gini importance)")
    fig.tight_layout()
    feature_imp_plot = plotting.figure_to_base64(fig)

    # ROC Curve
    roc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)
    fig_roc = roc_disp.figure_
    fig_roc.suptitle("ROC curve")
    fig_roc.tight_layout()
    roc_plot = plotting.figure_to_base64(fig_roc)

    # Precision Recall Curve
    precision, recall, _ = precision_recall_curve(test_y, test_y_)
    pr_disp = PrecisionRecallDisplay.from_estimator(clf, test_x, test_y)
    fig_pr = pr_disp.figure_
    fig_pr.suptitle("Precision Recall curve")
    fig_pr.tight_layout()
    pr_plot = plotting.figure_to_base64(fig_pr)

    # Confusion Matrix
    cm = confusion_matrix(test_y, test_y_)
    cm_disp = ConfusionMatrixDisplay.from_estimator(clf, test_x, test_y)
    fig_cm = cm_disp.figure_
    fig_cm.suptitle("Confusion Matrix")
    fig_cm.tight_layout()
    cm_plot = plotting.figure_to_base64(fig_cm)

    print(test_y_)
    auc = roc_auc_score(test_y, test_y_auc)
    print("AUC-ROC:", auc)
    precision = precision_score(test_y, test_y_)
    print("Precision:", precision)
    accuracy = accuracy_score(test_y, test_y_.round())
    print("Accuracy:", accuracy)
    recall = recall_score(test_y, test_y_, average='weighted')
    print("Recall:", recall)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    print("Specificity:", specificity)
    f1 = f1_score(test_y, test_y_, average='weighted')
    print("f1_score:", f1)

    return_dict = {"auc": auc}
    return_dict.update({"precision": precision})
    return_dict.update({"accuracy": accuracy})
    return_dict.update({"recall": recall})
    return_dict.update({"specificity": specificity})
    return_dict.update({"f1": f1})
    return_dict.update({"roc_plot": roc_plot})
    return_dict.update({"pr_plot": pr_plot})
    return_dict.update({"cm_plot": cm_plot})
    return_dict.update({"feature_imp_plot": feature_imp_plot})
    return return_dict


def k_nearest_neighbor_classification(dataframe, target_variable: str, independent_variables: list, algo_params: dict):
    df = dataframe
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

    print(test_y_)
    auc = roc_auc_score(test_y, test_y_auc)
    print("AUC-ROC:", auc)
    precision = precision_score(test_y, test_y_)
    print("Precision:", precision)
    accuracy = accuracy_score(test_y, test_y_.round())
    print("Accuracy:", accuracy)
    recall = recall_score(test_y, test_y_, average='weighted')
    print("Recall:", recall)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    print("Specificity:", specificity)
    f1 = f1_score(test_y, test_y_, average='weighted')
    print("f1_score:", f1)

    return_dict = {"auc": auc}
    return_dict.update({"precision": precision})
    return_dict.update({"accuracy": accuracy})
    return_dict.update({"recall": recall})
    return_dict.update({"specificity": specificity})
    return_dict.update({"f1": f1})
    return_dict.update({"roc_plot": roc_plot})
    return_dict.update({"pr_plot": pr_plot})
    return_dict.update({"cm_plot": cm_plot})
    return return_dict


def gaussian_naive_bayes(dataframe, target_variable: str, independent_variables: list):
    df = dataframe
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
    print("AUC-ROC:", auc)
    precision = precision_score(test_y, test_y_)
    print("Precision:", precision)
    accuracy = accuracy_score(test_y, test_y_.round())
    print("Accuracy:", accuracy)
    recall = recall_score(test_y, test_y_, average='weighted')
    print("Recall:", recall)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    print("Specificity:", specificity)
    f1 = f1_score(test_y, test_y_, average='weighted')
    print("f1_score:", f1)

    return_dict = {"auc": auc}
    return_dict.update({"precision": precision})
    return_dict.update({"accuracy": accuracy})
    return_dict.update({"recall": recall})
    return_dict.update({"specificity": specificity})
    return_dict.update({"f1": f1})
    return_dict.update({"roc_plot": roc_plot})
    return_dict.update({"pr_plot": pr_plot})
    return_dict.update({"cm_plot": cm_plot})
    return return_dict


def voting_cls(dataframe, target_variable: str, independent_variables: list, algo_params: dict):
    df = dataframe

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

    estimator_counts = {'trees': 0, 'knn': 0, 'random_forest': 0, 'gaussian_nb': 0}

    for estimator_i_params in algo_params["estimators_list"]:
        estimator_params = {key: value for key, value in estimator_i_params["algo_params"].items() if
                            value is not None and value != ''}
        if estimator_i_params['algo_id'] == 'decision_trees_cls':
            estimator_counts['trees'] += 1
            estimator = ('trees_' + str(estimator_counts['trees']), tree.DecisionTreeClassifier(**estimator_params))
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

    auc, roc_plot, pr_plot, cm_plot = None, None, None, None

    # Confusion Matrix
    cm = confusion_matrix(test_y, test_y_)
    cm_disp = ConfusionMatrixDisplay.from_estimator(clf, test_x, test_y)
    fig_cm = cm_disp.figure_
    cm_plot = plotting.figure_to_base64(fig_cm)

    if voting_params["voting"] == "soft":
        test_y_auc = clf.predict_proba(test_x)[:, 1]

        # ROC Curve
        roc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)
        fig_roc = roc_disp.figure_
        roc_plot = plotting.figure_to_base64(fig_roc)

        # Precision Recall Curve
        precision, recall, _ = precision_recall_curve(test_y, test_y_)
        pr_disp = PrecisionRecallDisplay.from_estimator(clf, test_x, test_y)
        fig_pr = pr_disp.figure_
        pr_plot = plotting.figure_to_base64(fig_pr)

        auc = roc_auc_score(test_y, test_y_auc)
        print("AUC-ROC:", auc)

    estimator_results = []

    for name, estimator in clf.named_estimators_.items():

        est_test_y_ = estimator.predict(test_x)

        cm = confusion_matrix(test_y, est_test_y_)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)

        estimator_result_dict = {
            "est_name": name,
            "accuracy": accuracy_score(test_y, est_test_y_),
            "recall": recall_score(test_y, est_test_y_),
            "specificity": specificity,
        }

        if voting_params["voting"] == "soft":
            est_test_y_auc = estimator.predict_proba(test_x)[:, 1]
            estimator_result_dict.update({"auc": roc_auc_score(test_y, est_test_y_auc)})

        estimator_results.append(estimator_result_dict)

    accuracy = accuracy_score(test_y, test_y_.round())
    print("Accuracy:", accuracy)
    recall = recall_score(test_y, test_y_, average='weighted')
    print("Recall:", recall)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)
    print("Specificity:", specificity)
    f1 = f1_score(test_y, test_y_, average='weighted')
    print("f1_score:", f1)

    return_dict = {"auc": auc}
    return_dict.update({"accuracy": accuracy})
    return_dict.update({"recall": recall})
    return_dict.update({"specificity": specificity})
    return_dict.update({"f1": f1})
    return_dict.update({"roc_plot": roc_plot})
    return_dict.update({"pr_plot": pr_plot})
    return_dict.update({"cm_plot": cm_plot})
    return_dict.update({"estimator_results": estimator_results})
    return return_dict


