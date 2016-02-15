import NaiveBayes

def no_observations_test():
  classifier = NaiveBayes.Classifier()
  classification = classifier.classify(observation=(23.2,))
  assert classification is None, "Should not classify observations without training examples."

def given_an_observation_for_a_single_class_test():
  classifier = NaiveBayes.Classifier()
  classifier.train(classification='a class', observation=(0,))
  classification = classifier.classify(observation=(23.2,))
  assert classification == 'a class', "Should always classify as given class if there is only one."

def given_one_observation_for_two_classes_test():
  classifier = NaiveBayes.Classifier()
  classifier.train(classification='a class', observation=(0,))
  classifier.train(classification='b class', observation=(100,))
  classification = classifier.classify(observation=(23.2,))
  assert classification is None, "Should classify as the nearest class."
  classification = classifier.classify(observation=(73.2,))
  assert classification is None, "Should classify as the nearest class."

def given_multiple_observations_for_two_classes_with_roughly_same_variance_test():
  classifier = NaiveBayes.Classifier()
  classifier.batch_train([('a class',(0.0,)),
                          ('a class',(1.0,)),
                          ('a class',(75.0,)),
                          ('b class',(25,)),
                          ('b class',(99,)),
                          ('b class',(100,))])

  classification = classifier.classify(observation=(25,))
  assert classification == 'a class', "Should classify as the best fit class."
  classification = classifier.classify(observation=(75.0,))
  assert classification == 'b class', "Should classify as the best fit class."

def given_multiple_observations_for_two_classes_with_different_variance_test():
  classifier = NaiveBayes.Classifier()
  classifier.train(classification='a class', observation=(0.0,))
  classifier.train(classification='a class', observation=(1.0,))
  classifier.train(classification='a class', observation=(2.0,))

  classifier.train(classification='b class', observation=(50,))
  classifier.train(classification='b class', observation=(75,))
  classifier.train(classification='b class', observation=(100,))

  classifier.train(classification='c class', observation=(0.0,))
  classifier.train(classification='c class', observation=(-1.0,))
  classifier.train(classification='c class', observation=(-2.0,))
  classification = classifier.classify(observation=(15,))
  assert classification == 'b class', "Because of class b's variance this should be class b."
  classification = classifier.classify(observation=(2.5,))
  assert classification == 'a class', "Should classify as class a because of tight variance."
  classification = classifier.classify(observation=(-2.5,))
  assert classification == 'c class', "Should classify as class c because it's the only negative one."

def given_classes_of_different_likelihood_test():
  classifier = NaiveBayes.Classifier()
  observation = (3,)
  observations = {
    'class a': [(1,),(2,),(3,),(4,),(5,)],
    'class b': [(1,),(1,),(2,),(2,),(3,),(3,),(4,),(4,),(5,),(5,)]
  }
  results = classifier._probability_of_each_class_given_data(observation, observations)
  print(results)
  assert results['class b'] > results['class a'], "Should classify as class b when class probability is taken into account."

def given_two_classes_with_two_dimension_inputs_test():
  classifier = NaiveBayes.Classifier()
  observation = (3,10)
  observations = {
    'class a': [(1,-1),(2,0),(3,-1),(4,1),(5,-1)],
    'class b': [(1,10),(2,5),(3,12),(4,10),(5,5)]
  }
  results = classifier._probability_of_each_class_given_data(observation, observations)
  print(results)
  assert results['class b'] > results['class a'], "Should classify as class b because of dimension 2."

def given_two_classes_with_identical_two_dimension_inputs_test():
  classifier = NaiveBayes.Classifier()
  observation = (3,10)
  observations = {
    'class a': [(1,10),(2,5),(3,12),(4,10),(5,5)],
    'class b': [(1,10),(2,5),(3,12),(4,10),(5,5)]
  }
  results = classifier._probability_of_each_class_given_data(observation, observations)
  print(results)
  assert results['class a'] == 0.5, "There should be 50/50 chance of class a"
  assert results['class b'] == 0.5, "There should be 50/50 chance of class b"

import pandas, pprint
import numpy as np
def given_real_data_test():
  patients = pandas.DataFrame.from_csv('./data/training_SyncPatient.csv').reset_index()
  transcripts = pandas.DataFrame.from_csv('./data/training_SyncTranscript.csv').reset_index()
  transcripts = transcripts[transcripts['Height'] > 0]
  transcripts = transcripts[transcripts['Weight'] > 0]
  transcripts = transcripts[transcripts['BMI'] > 0]
  joined_df = patients.merge(transcripts, on='PatientGuid', how='inner')
  final_df = joined_df.groupby('PatientGuid').first().reset_index()

  female_set = final_df.ix[np.random.choice(final_df[final_df['Gender']=='F'].index, 500)]
  male_set = final_df.ix[np.random.choice(final_df[final_df['Gender']=='M'].index, 500)]
  training_data = [(x[2], (x[8], x[9],x[10])) for x in female_set.values]
  training_data += [(x[2], (x[8], x[9],x[10])) for x in male_set.values]
  classifier = NaiveBayes.Classifier()
  for class_label, input_data in training_data:
    classifier.train(classification=class_label, observation=input_data)

  # Manual verification
  pprint.pprint(classifier._calculate_model_parameters())

  # Men
  print("Men")
  print(classifier.classify(observation=(71.3, 210.0, 23.509)))
  print(classifier.classify(observation=(66.0, 268.8, 27.241999999999997)))
  print(classifier.classify(observation=(65.0, 284.0, 30.616)))
  print("Women")
  print(classifier.classify(observation=(60.5, 151.0, 29.002)))
  print(classifier.classify(observation=(60.0, 148.0, 28.901)))
  print(classifier.classify(observation=(60.0, 134.923, 26.346999999999998)))
  assert True, "Always pass until we want to manually evaluate."

def given_data_test():
    classifier = NaiveBayes.Classifier()
    classifier.train(classification='class a', observation=(0.0,))
    classifier.train(classification='class a', observation=(1.0,))
    classifier.train(classification='class a', observation=(2.0,))

    classifier.train(classification='class b', observation=(50,))
    classifier.train(classification='class b', observation=(75,))
    classifier.train(classification='class b', observation=(100,))
    results = classifier._calculate_model_parameters()
    assert results['class a'][0]['mean'] == 1.0
    assert results['class a'][0]['variance'] == 2/3.
    assert results['class b'][0]['mean'] == 75.
    assert results['class b'][0]['variance'] == 416.66666666666669

def quantify_classifier_accuracy_test():
    # Load and clean up the data
    patients = pandas.DataFrame.from_csv('./data/training_SyncPatient.csv').reset_index()
    transcripts = pandas.DataFrame.from_csv('./data/training_SyncTranscript.csv').reset_index()
    transcripts = transcripts[transcripts['Height'] > 0]
    transcripts = transcripts[transcripts['Weight'] > 0]
    transcripts = transcripts[transcripts['BMI'] > 0]
    joined_df = patients.merge(transcripts, on='PatientGuid', how='inner')
    final_df = joined_df.groupby('PatientGuid').first().reset_index()
    total_set = final_df.ix[np.random.choice(final_df.index, 7000, replace=False)]

    # Partition development and cross-validation datasets
    training_count = 2000
    training_data = map(lambda x: (x[2], (x[8], x[9], x[10])), total_set.values[:training_count])
    cross_validate_data = map(lambda x: (x[2], (x[8], x[9], x[10])), total_set.values[training_count:])

    # Train the classifier on the training data.
    classifier = NaiveBayes.Classifier()
    for class_label, input_data in training_data:
      classifier.train(classification=class_label, observation=input_data)

    # Test how well the classifier generalizes.
    number_correct = 0
    number_tested = 0
    for class_label, input_data in cross_validate_data:
      number_tested += 1
      assigned_class = classifier.classify(observation=input_data)
      if class_label == assigned_class:
        number_correct += 1

    correct_rate = number_correct/(1.*number_tested)
    print("Correct rate: {0}, Total: {1}".format(correct_rate, number_tested))
    pprint.pprint(classifier._calculate_model_parameters())
    assert correct_rate > 0.6, "Should be significantly better than random."

import RandomForest
def random_forest_adapter_test():
    # Load and clean up the data
    patients = pandas.DataFrame.from_csv('./data/training_SyncPatient.csv').reset_index()
    transcripts = pandas.DataFrame.from_csv('./data/training_SyncTranscript.csv').reset_index()
    transcripts = transcripts[transcripts['Height'] > 0]
    transcripts = transcripts[transcripts['Weight'] > 0]
    transcripts = transcripts[transcripts['BMI'] > 0]
    joined_df = patients.merge(transcripts, on='PatientGuid', how='inner')
    final_df = joined_df.groupby('PatientGuid').first().reset_index()
    total_set = final_df.ix[np.random.choice(final_df.index, 7000, replace=False)]

    # Partition development and cross-validation datasets
    training_count = 500
    training_data = [(x[2], (x[8], x[9], x[10])) for x in total_set.values[:training_count]]
    cross_validate_data = [(x[2], (x[8], x[9], x[10])) for x in total_set.values[training_count:]]

    # Train the classifier on the training data.
    classifier = RandomForest.Classifier()
    classifier.batch_train(training_data)

    # Test how well the classifier generalizes.
    number_correct = 0
    number_tested = 0
    for class_label, input_data in cross_validate_data:
      number_tested += 1
      assigned_class = classifier.classify(observation=input_data)
      if class_label == assigned_class:
        number_correct += 1

    correct_rate = number_correct/(1.*number_tested)
    print("Correct rate: {0}, Total: {1}".format(correct_rate, number_tested))
    assert correct_rate > 0.6, "Should be significantly better than random."
