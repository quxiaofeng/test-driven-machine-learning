from sklearn.ensemble import RandomForestClassifier
class Classifier:
  def __init__(self):
    self._forest = RandomForestClassifier(n_estimators = 100)
    self._model = None
  def batch_train(self, observations):
    class_labels = [x[0] for x in observations]
    class_inputs = [x[1] for x in observations]
    self._model = self._forest.fit(class_inputs, class_labels)
  def classify(self, observation):
    return self._model.predict(observation)
