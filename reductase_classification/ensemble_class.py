from river import tree, compat, ensemble, base
import collections

class EnsembleModel(base.EnsembleMixin):
    def __init__(self, models, classes, weights):
        super().__init__([models])
        self.models = models
        self.classes = classes
        self.weights = weights
    
    def learn_one(self,x,y):
        for model in self.models:
            model.learn_one(x,y)
        return self

    def learn_many(self, X, y):
        for model in self.models:
            model.learn_many(X,y)
        return self
    
    def predict_proba_one(self, x):
        y_pred = collections.Counter()
        for i in range(len(self.models)):
            model = self.models[i]
            metric_value = self.weights[i]
            y_proba_temp = model.predict_proba_one(x)
            if metric_value > 0.0:
                y_proba_temp = {
                    k: val * metric_value for k, val in y_proba_temp.items()
                }
            y_pred.update(y_proba_temp)

        total = sum(y_pred.values())
        return {label: proba / total for label, proba in y_pred.items()}

    def predict_one(self, x):
        y_pred = self.predict_proba_one(x)
        if y_pred:
            return max(y_pred, key=y_pred.get)
        return None

    def _supervised(self):
        return True