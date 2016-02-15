import NaiveBayes

def no_observations_test():
  classifier = NaiveBayes.Classifier()
  classification = classifier.classify(observation=23.2)
  assert classification is None, "Should not classify observations without training examples."

def given_an_observation_for_a_single_class_test():
  classifier = NaiveBayes.Classifier()
  classifier.train(classification='a class', observation=0)
  classification = classifier.classify(observation=23.2)
  assert classification == 'a class', "Should always classify as given class if there is only one."

def given_one_observation_for_two_classes_test():
  classifier = NaiveBayes.Classifier()
  classifier.train(classification='a class', observation=0)
  classifier.train(classification='b class', observation=100)
  classification = classifier.classify(observation=23.2)
  assert classification is None, "Should classify as the nearest class."
  classification = classifier.classify(observation=73.2)
  assert classification is None, "Should classify as the nearest class."

def given_multiple_observations_for_two_classes_with_roughly_same_variance_test():
  classifier = NaiveBayes.Classifier()
  classifier.train(classification='a class', observation=0.0)
  classifier.train(classification='a class', observation=1.0)
  classifier.train(classification='a class', observation=75.0)
  classifier.train(classification='b class', observation=25)
  classifier.train(classification='b class', observation=99)
  classifier.train(classification='b class', observation=100)
  classification = classifier.classify(observation=25)
  assert classification == 'a class', "Should classify as the best fit class."
  classification = classifier.classify(observation=75.0)
  assert classification == 'b class', "Should classify as the best fit class."

def given_multiple_observations_for_two_classes_with_different_variance_test():
  classifier = NaiveBayes.Classifier()
  classifier.train(classification='a class', observation=0.0)
  classifier.train(classification='a class', observation=1.0)
  classifier.train(classification='a class', observation=2.0)

  classifier.train(classification='b class', observation=50)
  classifier.train(classification='b class', observation=75)
  classifier.train(classification='b class', observation=100)

  classifier.train(classification='c class', observation=0.0)
  classifier.train(classification='c class', observation=-1.0)
  classifier.train(classification='c class', observation=-2.0)
  classification = classifier.classify(observation=15)
  assert classification == 'b class', "Because of class b's variance this should be class b."
  classification = classifier.classify(observation=2.5)
  assert classification == 'a class', "Should classify as class a because of tight variance."
  classification = classifier.classify(observation=-2.5)
  assert classification == 'c class', "Should classify as class c because it's the only negative one."

def given_classes_of_different_likelihood_test():
  classifier = NaiveBayes.Classifier()
  observation = 3
  observations = {
    'class a': [1,2,3,4,5],
    'class b': [1,1,2,2,3,3,4,4,5,5]
  }
  results = classifier._probability_of_each_class_given_data(observation, observations)
  print(results)
  assert results['class b'] > results['class a'], "Should classify as class b when class probability is taken into account."
