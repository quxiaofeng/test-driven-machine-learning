class SimpleBandit:
  def __init__(self, treatments):
  	self._treatments = treatments
  	self._selection_count = 0
  	self._exploitation_count = 0
  	self._payouts = {treatment: 0 for treatment in treatments}
  def choose_treatment(self):
  	self._selection_count += 1
  	if self._selection_count <= 5*len(self._treatments):
  		return self._treatments[int((self._selection_count-1) / 5)]
  	else:
  		self._exploitation_count += 1
  		if self._exploitation_count == 5:
  			self._exploitation_count = 0
  			self._selection_count = 0
  		return sorted(self._payouts.items(), key=lambda x: x[1], reverse=True)[0][0]
  def log_payout(self, treatment, amount):
    self._payouts[treatment] += amount