import collections

class NoClassifierOptionsException(Exception):
  pass

class ClassifierChooser:
    def __init__(self,
                 classifier_options_list,
                 test_label=[],
                 test_input=[],
                 training_labels=[],
                 training_inputs=[]):
        self._classifier_options = classifier_options_list[0]
        highest_score = 0
        for classifier in classifier_options_list:
            classifier.batch_train(list(zip(training_labels, training_inputs)))
            number_right = 0
            for input_value, correct_value in zip(test_input, test_label):
                predicted_label = classifier.classify(input_value)
                if predicted_label == correct_value:
                    number_right += 1
            if number_right > highest_score:
                self._classifier_options = classifier
            print('Classifier: {0}; Number right: {1}'.format(classifier, number_right))

    @staticmethod
    def create_with_single_classifier_option(classifier_option):
        return ClassifierChooser(classifier_options_list=[classifier_option])

    def classify(self, input):
        return self._classifier_options.classify(input)

class AlwaysTrueClassifier:
    def batch_train(self, observations):
        pass
    def classify(self, input):
        return 1

class AlwaysFalseClassifier:
    def batch_train(self, observations):
        pass
    def classify(self, input):
        return 0

class CopyCatClassifier:
    def batch_train(self, observations):
        pass
    def classify(self, input):
        return input

class DictionaryClassifier:
    def __init__(self):
        self._memory = {}
    def batch_train(self, observations):
        for label, observation in observations:
            if not observation in self._memory:
                self._memory[observation] = label
    def classify(self, observation):
        return self._memory[observation]
