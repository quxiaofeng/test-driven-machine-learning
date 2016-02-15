import rpm_bandit

def given_a_single_treatment_test():
  bandit = rpm_bandit.RPMBandit(['A'])
  chosen_treatment = bandit.choose_treatment()
  assert chosen_treatment == 'A', 'Should choose the only available option.'

def given_a_multiple_treatment_test():
  bandit = rpm_bandit.RPMBandit(['A', 'B'])
  chosen_treatment = bandit.choose_treatment()
  assert chosen_treatment in ['A', 'B'], 'Should choose any option. Doesnt matter.'

def given_a_multiple_treatment_with_a_single_sample_test():
  bandit = rpm_bandit.RPMBandit(['A', 'B'])
  bandit.log_payout('A', 35)
  bandit.log_payout('B', 34)

  treatment_a_chosen_count = sum([bandit.choose_treatment() == 'A' for i in range(50)])
  assert treatment_a_chosen_count < 50, "Each treatment should be assigned randomly"

def given_a_multiple_treatment_with_data_weighing_towards_a_treatment_test():
  bandit = rpm_bandit.RPMBandit(['A', 'B'])
  bandit.log_payout('A', 100)
  bandit.log_payout('A', 120)
  bandit.log_payout('A', 150)
  bandit.log_payout('B', 34)
  bandit.log_payout('B', 35)
  bandit.log_payout('B', 32)
  treatment_a_chosen_count = sum([bandit.choose_treatment() == 'A' for i in range(1000)])
  assert treatment_a_chosen_count > 900, 'Treatment A should be chosen much more than 50% of the time.'

