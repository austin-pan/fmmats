import numpy as _np
from sklearn.metrics import mean_squared_error as _mean_squared_error

def rmse(y_true: any, y_pred: any) -> float:
    '''Calculate the root mean squared error (RMSE) between the predictions and ground truths.'''
    return _np.sqrt(_mean_squared_error(y_true, y_pred))