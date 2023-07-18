from sklearn import tree
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import RocCurveDisplay, roc_curve
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import io
import base64


def decision_trees_classification(path: str, target_variable: str, independent_variables: list, algo_params: dict):
    df = pd.read_csv(path)
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.90, random_state=1)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    clf = tree.DecisionTreeClassifier(**algo_params)
    clf.fit(train_x, train_y)
    test_y_ = clf.predict(test_x)
    test_y_auc = clf.predict_proba(test_x)[:, 1]


    print("---------------------------------------------------------------------")

    rfc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)
    precision, recall, _ = precision_recall_curve(test_y, test_y_)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    cm = confusion_matrix(test_y, test_y_)
    cm_display = ConfusionMatrixDisplay(cm).plot()

    fig_rfc = rfc_disp.figure_
    buffer_rfc = io.BytesIO()
    fig_rfc.savefig(buffer_rfc, format='png')
    buffer_rfc.seek(0)
    plot_rfc_data = base64.b64encode(buffer_rfc.getvalue()).decode()


    fig_disp = disp.figure_
    buffer_disp = io.BytesIO()
    fig_disp.savefig(buffer_disp, format='png')
    buffer_disp.seek(0)
    plot_disp_data = base64.b64encode(buffer_disp.getvalue()).decode()


    fig_cm = cm_display.figure_
    buffer_cm = io.BytesIO()
    fig_cm.savefig(buffer_cm, format='png')
    buffer_cm.seek(0)
    plot_cm_data = base64.b64encode(buffer_cm.getvalue()).decode()


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
    return_dict.update({"rfc_plot": plot_rfc_data})
    return_dict.update({"disp_plot": plot_disp_data})
    return_dict.update({"cm_plot": plot_cm_data})
    return return_dict


def random_forest_classification(path: str, target_variable: str, independent_variables: list, algo_params: dict):
    df = pd.read_csv(path)
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.90, random_state=1)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    clf = RandomForestClassifier(**algo_params)
    clf.fit(train_x, train_y)
    test_y_ = clf.predict(test_x)
    test_y_auc = clf.predict_proba(test_x)[:, 1]

    rfc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)

    print("---------------------------------------------------------------------")

    rfc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)
    precision, recall, _ = precision_recall_curve(test_y, test_y_)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    cm = confusion_matrix(test_y, test_y_)
    cm_display = ConfusionMatrixDisplay(cm).plot()

    fig_rfc = rfc_disp.figure_
    buffer_rfc = io.BytesIO()
    fig_rfc.savefig(buffer_rfc, format='png')
    buffer_rfc.seek(0)
    plot_rfc_data = base64.b64encode(buffer_rfc.getvalue()).decode()

    fig_disp = disp.figure_
    buffer_disp = io.BytesIO()
    fig_disp.savefig(buffer_disp, format='png')
    buffer_disp.seek(0)
    plot_disp_data = base64.b64encode(buffer_disp.getvalue()).decode()

    fig_cm = cm_display.figure_
    buffer_cm = io.BytesIO()
    fig_cm.savefig(buffer_cm, format='png')
    buffer_cm.seek(0)
    plot_cm_data = base64.b64encode(buffer_cm.getvalue()).decode()

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
    return_dict.update({"rfc_plot": plot_rfc_data})
    return_dict.update({"disp_plot": plot_disp_data})
    return_dict.update({"cm_plot": plot_cm_data})
    return return_dict


def k_nearest_neighbor_classification(path: str, target_variable: str, independent_variables: list, algo_params: dict):
    df = pd.read_csv(path)
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.90, random_state=1)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    clf = RadiusNeighborsClassifier(**algo_params)
    clf.fit(train_x, train_y)

    test_y_ = clf.predict(test_x)
    test_y_auc = clf.predict_proba(test_x)[:, 1]

    rfc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)

    print(test_y)
    print("---------------------------------------------------------------------")

    rfc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)
    precision, recall, _ = precision_recall_curve(test_y, test_y_)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    cm = confusion_matrix(test_y, test_y_)
    cm_display = ConfusionMatrixDisplay(cm).plot()

    fig_rfc = rfc_disp.figure_
    buffer_rfc = io.BytesIO()
    fig_rfc.savefig(buffer_rfc, format='png')
    buffer_rfc.seek(0)
    plot_rfc_data = base64.b64encode(buffer_rfc.getvalue()).decode()

    fig_disp = disp.figure_
    buffer_disp = io.BytesIO()
    fig_disp.savefig(buffer_disp, format='png')
    buffer_disp.seek(0)
    plot_disp_data = base64.b64encode(buffer_disp.getvalue()).decode()

    fig_cm = cm_display.figure_
    buffer_cm = io.BytesIO()
    fig_cm.savefig(buffer_cm, format='png')
    buffer_cm.seek(0)
    plot_cm_data = base64.b64encode(buffer_cm.getvalue()).decode()

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
    return_dict.update({"rfc_plot": plot_rfc_data})
    return_dict.update({"disp_plot": plot_disp_data})
    return_dict.update({"cm_plot": plot_cm_data})
    return return_dict


def gaussian_naive_bayes(path: str, target_variable: str, independent_variables: list):
    df = pd.read_csv(path)
    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.90, random_state=1)
    train_x = train_x.to_numpy()
    test_x = test_x.to_numpy()
    train_y = train_y.to_numpy().ravel()
    test_y = test_y.to_numpy().ravel()

    clf = GaussianNB()
    clf.fit(train_x, train_y)
    test_y_ = clf.predict(test_x)
    test_y_auc = clf.predict_proba(test_x)[:, 1]

    rfc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)

    print("---------------------------------------------------------------------")

    rfc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)
    precision, recall, _ = precision_recall_curve(test_y, test_y_)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    cm = confusion_matrix(test_y, test_y_)
    cm_display = ConfusionMatrixDisplay(cm).plot()

    fig_rfc = rfc_disp.figure_
    buffer_rfc = io.BytesIO()
    fig_rfc.savefig(buffer_rfc, format='png')
    buffer_rfc.seek(0)
    plot_rfc_data = base64.b64encode(buffer_rfc.getvalue()).decode()

    fig_disp = disp.figure_
    buffer_disp = io.BytesIO()
    fig_disp.savefig(buffer_disp, format='png')
    buffer_disp.seek(0)
    plot_disp_data = base64.b64encode(buffer_disp.getvalue()).decode()

    fig_cm = cm_display.figure_
    buffer_cm = io.BytesIO()
    fig_cm.savefig(buffer_cm, format='png')
    buffer_cm.seek(0)
    plot_cm_data = base64.b64encode(buffer_cm.getvalue()).decode()

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
    return_dict.update({"rfc_plot": plot_rfc_data})
    return_dict.update({"disp_plot": plot_disp_data})
    return_dict.update({"cm_plot": plot_cm_data})
    return return_dict


def voting_cls(path: str, target_variable: str, independent_variables: list, algo_params: dict):
    df = pd.read_csv(path)

    x = df[independent_variables]
    y = df[[target_variable]]
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.90, random_state=1)
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

    print("---------------------------------------------------------------------")
    rfc_disp = RocCurveDisplay.from_estimator(clf, test_x, test_y)
    precision, recall, _ = precision_recall_curve(test_y, test_y_)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    cm = confusion_matrix(test_y, test_y_)
    cm_display = ConfusionMatrixDisplay(cm).plot()

    fig_rfc = rfc_disp.figure_
    buffer_rfc = io.BytesIO()
    fig_rfc.savefig(buffer_rfc, format='png')
    buffer_rfc.seek(0)
    plot_rfc_data = base64.b64encode(buffer_rfc.getvalue()).decode()

    fig_disp = disp.figure_
    buffer_disp = io.BytesIO()
    fig_disp.savefig(buffer_disp, format='png')
    buffer_disp.seek(0)
    plot_disp_data = base64.b64encode(buffer_disp.getvalue()).decode()

    fig_cm = cm_display.figure_
    buffer_cm = io.BytesIO()
    fig_cm.savefig(buffer_cm, format='png')
    buffer_cm.seek(0)
    plot_cm_data = base64.b64encode(buffer_cm.getvalue()).decode()

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
    return_dict.update({"rfc_plot": plot_rfc_data})
    return_dict.update({"disp_plot": plot_disp_data})
    return_dict.update({"cm_plot": plot_cm_data})
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
