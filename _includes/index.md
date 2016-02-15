[English]({{site.baseurl}}/) | [中文]({{site.baseurl}}/cn/)

Chapter 1
---------

*number_guesser_tests.py*

```python
from NumberGuesser import NumberGuesser

def given_no_information_when_asked_to_guess_test():
  # given
  number_guesser = NumberGuesser()
  # when
  guessed_number = number_guesser.guess()
  # then
  assert guessed_number is None, 'there should be no guess.'

def given_one_datapoint_when_asked_to_guess_test():
  #given
  number_guesser = NumberGuesser()
  previously_chosen_number = 5
  number_guesser.number_was(previously_chosen_number)
  #when
  guessed_number = number_guesser.guess()
  #then
  assert type(guessed_number) is int, 'the answer should be a number'
  assert guessed_number == previously_chosen_number, 'the answer should be the previously chosen number.'

def given_multiple_datapoints_when_asked_to_guess_many_times_test():
  #given
  number_guesser = NumberGuesser()
  previously_chosen_numbers = [1,2,5]
  number_guesser.numbers_were(previously_chosen_numbers)
  #when
  guessed_numbers = [number_guesser.guess() for i in range(0,100)]
  #then
  for guessed_number in guessed_numbers:
    assert guessed_number in previously_chosen_numbers, 'every guess should be one of the previously chosen numbers'
  assert len(set(guessed_numbers)) > 1, "It shouldn't always guess the same number."

def given_a_starting_set_of_observations_followed_by_a_one_off_observation_test():
    #given
  number_guesser = NumberGuesser()
  previously_chosen_numbers = [1,2,5]
  number_guesser.numbers_were(previously_chosen_numbers)
  one_off_observation = 0
  number_guesser.number_was(one_off_observation)
  #when
  guessed_numbers = [number_guesser.guess() for i in range(0,100)]
  #then
  for guessed_number in guessed_numbers:
    assert guessed_number in previously_chosen_numbers + [one_off_observation], 'every guess should be one of the previously chosen numbers'
  assert len(set(guessed_numbers)) > 1, "It shouldn't always guess the same number."

def given_a_one_off_observation_followed_by_a_set_of_observations_test():
  #given
  number_guesser = NumberGuesser()
  print(number_guesser._guessed_numbers)
  previously_chosen_numbers = [1,2]
  one_off_observation = 0
  number_guesser.number_was(one_off_observation)
  number_guesser.numbers_were(previously_chosen_numbers)

  all_observations = previously_chosen_numbers + [one_off_observation]
  #when
  guessed_numbers = [number_guesser.guess() for i in range(0,100)]
  #then
  for guessed_number in guessed_numbers:
    print(guessed_number, all_observations)
    assert guessed_number in all_observations, 'every guess should be one of the previously chosen numbers'
  assert len(set(guessed_numbers)) == len(all_observations), "It should eventually guess every number at least once."
```

*NumberGuesser.py*

```python
import random
class NumberGuesser:
  """Guesses numbers based on the history of your input"""
  def __init__(self):
    self._guessed_numbers = []
  def numbers_were(self, guessed_numbers):
    self._guessed_numbers += guessed_numbers
  def number_was(self, guessed_number):
    self._guessed_numbers.append(guessed_number)
  def guess(self):
    if self._guessed_numbers == []:
      return None
    return random.choice(self._guessed_numbers)
```

An Equation Test
----------------

\\[
w\_{i+1,j} = w\_{i,j} + \\eta \\times w\_{i,j} \\times \\left( t\_j - p\_j \\right) 
\\]

\\[
p\_j = \\sum w\_i \\times x\_i > 0
\\]