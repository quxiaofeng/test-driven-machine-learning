import pandas 
import sklearn.metrics
import statsmodels.formula.api as smf
import numpy as np

def logistic_regression_test():
  df = pandas.DataFrame.from_csv('./generated_logistic_data.csv')

  generated_model = smf.logit('y ~ variable_a + variable_b + variable_c', df)
  generated_fit = generated_model.fit()
  roc_data = sklearn.metrics.roc_curve(df['y'], generated_fit.predict(df))
  auc = sklearn.metrics.auc(roc_data[0], roc_data[1])
  print(generated_fit.summary())
  print("AUC score: {0}".format(auc))
  assert auc > .8, 'AUC should be significantly above random'