import numpy as np


class Regressor():

    def __init__(self,):
        pass

    def loss(self, X_test, y_test, type='relative', aggregate='avg', percentile=None):

        def relative_loss(X_test, y_test):
            return np.abs(np.exp(self.predict(X_test)) - y_test) / y_test

        def abs_loss(X_test, y_test):
            return np.abs(np.exp(self.predict(X_test)) - y_test)

        def mse_loss(X_test, y_test):
            return np.power(self.predict(X_test) - y_test, 2)

        if type == 'relative':
            loss_func = relative_loss
        elif type == 'abs':
            loss_func = abs_loss
        elif type == 'mse':
            loss_func = mse_loss
        else:
            assert False, f"Loss func '{type}' not supported"

        loss = loss_func(X_test, y_test)

        if aggregate == 'avg':
            return np.average(loss)
        elif aggregate == 'max':
            return np.max(loss)
        elif aggregate == 'min':
            return np.min(loss)
        elif aggregate == 'percentile':
            assert percentile is not None, "If use percentile loss, you should provide a percentile"
            return np.percentile(loss, percentile)
        else:
            assert False, f"Aggregate func '{aggregate}' not supported"

    def predict(self, X):
        pass
