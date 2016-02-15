import numpy as np

class Classifier:
  def __init__(self):
    self._classifications = {}
  def train(self, classification, observation):
    if not classification in self._classifications:
      self._classifications[classification] = []
    self._classifications[classification].append(observation)
  def probability_of_data_given_class(self, observation, class_observations):
    mean = np.mean(class_observations)
    variance = np.var(class_observations)
    p_data_given_class = 1/np.sqrt(2*np.pi*variance)*np.exp(-0.5*((observation - mean)**2)/variance)
    return p_data_given_class
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