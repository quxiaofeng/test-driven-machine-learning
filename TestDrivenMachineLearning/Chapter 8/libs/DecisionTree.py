from sklearn.tree import DecisionTreeRegressor

class Classifier:
    def __init__(self):
      self._decision_tree = DecisionTreeRegressor()
      self._model = None
    def batch_train(self, observations):
        class_labels = [x[0] for x in observations]
        class_inputs = [x[1] for x in observations]
        observations = self._decision_tree.fit(class_inputs, class_labels)
        pass
    def classify(self, observation):
        return self._decision_tree.predict(observation)
