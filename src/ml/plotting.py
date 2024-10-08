from io import BytesIO
from base64 import b64encode
import matplotlib.pyplot as plt


def figure_to_base64(figure):
    """
    Encode scikit-learn figure into base64 format
    """
    buffer = BytesIO()
    figure.savefig(buffer, format='png')
    plot_as_base64 = b64encode(buffer.getvalue()).decode()
    buffer.close()
    return plot_as_base64


def plot_importance_figure(importance_values, independent_variables):
    fig = plt.Figure()
    ax = fig.subplots()
    sorted_data = sorted(zip(importance_values, independent_variables))
    importance_values, independent_variables = zip(*sorted_data)
    ax.bar([independent_variables[x] for x in range(len(importance_values))], importance_values)
    ax.set_xlabel("Risk Factors")
    ax.set_ylabel("Feature Importance Score (Gini Coefficient)")
    ax.set_xticks(independent_variables)
    ax.set_xticklabels(independent_variables, rotation=40, ha='right')
    fig.tight_layout()
    return fig
