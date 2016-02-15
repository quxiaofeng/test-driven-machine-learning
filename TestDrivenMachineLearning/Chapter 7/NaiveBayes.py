import numpy as np
import operator
from functools import reduce

class Classifier:
  def __init__(self):
    self._classifications = {}
  def batch_train(self, observations):
    for label, observation in observations:
      self.train(label, observation)
  def train(self, classification, observation):
    if not classification in self._classifications:
      self._classifications[classification] = []
    self._classifications[classification].append(observation)
  def probability_of_data_given_class(self, observation, class_observations):
    lists_of_observations = list(zip(*class_observations))
    probabilities = []
    for class_observation_index in range(len(lists_of_observations)):
        some_class_observations = lists_of_observations[class_observation_index]
        the_observation = observation[class_observation_index]
        mean = np.mean(some_class_observations)
        variance = np.var(some_class_observations)
        p_data_given_class = 1/np.sqrt(2*np.pi*variance)*np.exp(-0.5*((the_observation - mean)**2)/variance)
        probabilities.append(p_data_given_class)
    return reduce(operator.mul, probabilities, 1)
  def _probability_of_each_class_given_data(self, observation, classifications):
    all_observations = 1.0*sum([len(class_values) for class_values in classifications.values()])
    class_probabilities = { class_label: len(classifications[class_label])/all_observations
                            for class_label in classifications.keys()}
    sum_of_probabilities = sum([self.probability_of_data_given_class(observation, classifications[class_label])
                                * class_probability
                                for class_label, class_probability in class_probabilities.items()])
    probability_class_given_data = {}
    for class_label, observations in classifications.items():
      class_probability = self.probability_of_data_given_class(observation, observations) \
        * class_probabilities[class_label] \
        / sum_of_probabilities
      probability_class_given_data[class_label] = class_probability
    return probability_class_given_data
  def _any_classes_are_too_small(self, classifications):
    for item in classifications.values():
      if len(item) <= 1:
        return True
    return False
  def classify(self, observation):
    if len(self._classifications.keys()) == 0:
      return None
    elif len(self._classifications.keys()) == 1:
      return list(self._classifications.keys())[0]
    elif self._any_classes_are_too_small(self._classifications):
      return None
    else:
      results = self._probability_of_each_class_given_data(observation, self._classifications)
      return max(results.items(), key=lambda x: x[1])[0]
  def _calculate_model_parameters(self):
    class_metrics = {}
    for class_label, data in self._classifications.items():
      class_metrics[class_label] = []
      columnar_data = zip(*data)
      for column in columnar_data:
        class_metrics[class_label].append({
          "mean": np.mean(column),
          "variance": np.var(column)
        })
    return class_metrics
