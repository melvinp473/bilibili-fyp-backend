from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_squared_log_error, mean_absolute_percentage_error
from sklearn.metrics import median_absolute_error, max_error

def metric_r2(y_true,y_pred):
    return r2_score(y_true, y_pred)

def metric_mean_absolute(y_true,y_pred):
    return mean_absolute_error(y_true, y_pred)

def metric_mean_squared(y_true,y_pred):
    return mean_squared_error(y_true,y_pred)

def metric_mean_squared_log(y_true,y_pred):
    try:
        value = mean_squared_log_error(y_true, y_pred)
    except ValueError:
        value = 'n/a'
    return value

def metric_mean_absolute_percentage(y_true,y_pred):
    return mean_absolute_percentage_error(y_true, y_pred)

def metric_media_absolute(y_true,y_pred):
    return median_absolute_error(y_true, y_pred)

def metric_max_error(y_true,y_pred):
    return max_error(y_true, y_pred)

