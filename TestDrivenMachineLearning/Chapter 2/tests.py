import nose.tools as nt

def the_easy_test():
  nt.assert_true(True)

from perceptron import *

def no_training_data_supplied_test():
  the_perceptron = Perceptron()
  result = the_perceptron.predict([])
  nt.assert_equal(result, None, 'Should have no result with no training data.')

def train_an_OR_function_test():
  the_perceptron = Perceptron()
  the_perceptron.train([
                           [1,1], 
                           [0,1], 
                           [1,0], 
                           [0,0]
                         ], 
                         [1,1,1,0])
  nt.assert_equal(the_perceptron.predict([1,1]), 1)
  nt.assert_equal(the_perceptron.predict([1,0]), 1)
  nt.assert_equal(the_perceptron.predict([0,1]), 1)
  nt.assert_equal(the_perceptron.predict([0,0]), 0)


def detect_values_greater_than_five_test():
  the_perceptron = Perceptron()
  the_perceptron.train([
                           [ 5], 
                           [ 2], 
                           [ 0],
                           [-2], 
                         ], 
                         [1,0,0,0])
  nt.assert_equal(the_perceptron.predict([ 8]),    1)
  nt.assert_equal(the_perceptron.predict([ 5]),    1)
  nt.assert_equal(the_perceptron.predict([ 2]),    0)
  nt.assert_equal(the_perceptron.predict([ 0]),    0)
  nt.assert_equal(the_perceptron.predict([-2]),    0)

import numpy as np 
def detect_a_complicated_example_test():
  # Create random variables
  training_n = 100
  inputs = list(map(list, zip(np.random.uniform(0,100,training_n), 
      np.random.uniform(0,100,training_n), 
      np.random.uniform(0,100,training_n))))
  labels = [int(x[0] + x[1] + x[2] < 150) for x in inputs]
  the_perceptron = Perceptron()
  the_perceptron.train(inputs, labels)

  testing_n = 2500
  test_inputs = list(map(list, zip(np.random.uniform(0,100,testing_n), 
      np.random.uniform(0,100,testing_n), 
      np.random.uniform(0,100,testing_n))))
  test_labels = [int(x[0] + x[1] + x[2] < 150) for x in test_inputs]

  # Create separate test cases
  correctly_classified = 0
  total_classified = 0
  for input, label in zip(test_inputs, test_labels):
    prediction = the_perceptron.predict(input)
    total_classified += 1
    if prediction == 1:
      if label == 1:
        correctly_classified += 1
    else:
      if label == 0:
        correctly_classified += 1
  # Make sure we generated as much data as we'd expect
  nt.assert_equal(total_classified, testing_n)
  assert correctly_classified >= .9*testing_n,  \
         "Perceptron should be much better than random. {0} correct".format(correctly_classified)
