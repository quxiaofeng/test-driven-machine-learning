import rpm_bandit, simple_bandit
import numpy as np

class BanditScenario:
  def __init__(self, scenario):
    self._scenario = scenario
    self._scenario_payoffs = {treatment_name:[] for treatment_name in self._scenario.keys()}
    self._bandit_payoffs = []
  def next_visitor(self, show_treatment):
    for key, value in self._scenario.items():
      ordered = np.random.binomial(1, value['conversion_rate'])
      order_average = np.random.normal(loc=value['order_average'], scale=5.00)
      self._scenario_payoffs[key].append(ordered*order_average)
      if key == show_treatment:
          self._bandit_payoffs.append(ordered*order_average)
    return self._scenario_payoffs[show_treatment][-1]

def run_bandit_sim(bandit_algorithm):
    simulated_experiment = BanditScenario({
      'A': {
        'conversion_rate': 1,
        'order_average': 35.00
      }, 
      'B':{
        'conversion_rate': 1,
        'order_average': 50.00
      }
    })

    simple_bandit = bandit_algorithm

    for visitor_i in range(500):
      treatment = simple_bandit.choose_treatment()
      payout = simulated_experiment.next_visitor(treatment)
      simple_bandit.log_payout(treatment, payout)

    return sum(simulated_experiment._bandit_payoffs)

def run_comparison_test():
  simple_bandit_results = np.array([run_bandit_sim(simple_bandit.SimpleBandit(['A', 'B'])) for i in range(300)])
  rpm_bandit_results = np.array([run_bandit_sim(rpm_bandit.RPMBandit(['A', 'B'])) for i in range(300)])
  rpm_better_count = sum(map(lambda x: x[0] > x[1], zip(rpm_bandit_results, simple_bandit_results)))
  assert rpm_better_count/300. > .8, 'The RPM bandit should be better at least 80% of the time.'



