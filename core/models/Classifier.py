

class Classifier():

    def __init__(self,):
        self.model = None

    def score(self, X_test, y_test):
        return self.model.score(X_test, y_test)

    def predict(self, X):
        pass
