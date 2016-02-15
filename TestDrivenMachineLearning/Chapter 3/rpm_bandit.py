import random
import numpy as np

class RPMBandit:
  def __init__(self, treatments):
    self._treatments = treatments
    self._payoffs = {treatment: [] for treatment in treatments}
  def choose_treatment(self):
    max_treatment = self._treatments[0]
    max_value = float('-inf')
    for key, value in self._payoffs.items():
      random_numbers_from_range = np.random.binomial(len(value)+1, 1.0/(len(value)+1))
      generated_data = value + [random.uniform(0,200) for i in range(random_numbers_from_range)]
      sampled_mean = np.random.choice(generated_data, size=len(generated_data)).mean()
      if sampled_mean > max_value:
        max_treatment = key
        max_value = sampled_mean
    return max_treatment
  def log_payout(self, treatment, payout):
    self._payoffs[treatment].append(payout)