class Perceptron:
  def train(self, inputs, labels):
    dummied_inputs = [x + [-1] for x in inputs]
    self._weights = [0.2] * len(dummied_inputs[0])
    for _ in range(5000):
      for input, label in zip(dummied_inputs, labels):
        label_delta = (label - self.predict(input))
        for index, x in enumerate(input):
          self._weights[index] += .1 * x * label_delta
  def predict(self, input):
    if len(input) == 0:
      return None
    input = input + [-1]
    return int(0 < sum([x[0]*x[1] for x in zip(self._weights, input)]))