import pandas
import statsmodels.formula.api as sm

def vanilla_model_test():
  df = pandas.read_csv('./generated_data.csv')
  model_fit = sm.ols('dependent_var ~ ind_var_a + ind_var_b + ind_var_c + ind_var_e + ind_var_b * ind_var_c', data=df).fit()
  print(model_fit.summary())
  assert model_fit.f_pvalue <= 0.05, "Prob(F-statistic) should be small enough to reject the null hypothesis."
  assert model_fit.rsquared_adj >= 0.95, "Model should explain 95% of the variation in the sampled data or more."

def final_model_cross_validation_test():
  df = pandas.read_csv('./generated_data.csv')
  df['predicted_dependent_var'] = 25.6266 \
                                + 2.7083*df['ind_var_a'] \
                                - 1.5527*df['ind_var_b'] \
                                - 0.3917*df['ind_var_c'] \
                                - 0.2006*df['ind_var_e'] \
                                + 5.6450*df['ind_var_b'] * df['ind_var_c']
  df['diff'] = (df['dependent_var'] - df['predicted_dependent_var']).abs()
  print(df['diff'])
  print('===========')
  cv_df = pandas.read_csv('./generated_data_cv.csv')
  cv_df['predicted_dependent_var'] = 25.6266 \
                                + 2.7083*cv_df['ind_var_a'] \
                                - 1.5527*cv_df['ind_var_b'] \
                                - 0.3917*cv_df['ind_var_c'] \
                                - 0.2006*cv_df['ind_var_e'] \
                                + 5.6450*cv_df['ind_var_b'] * cv_df['ind_var_c']
  cv_df['diff'] = (cv_df['dependent_var'] - cv_df['predicted_dependent_var']).abs()
  print(cv_df['diff'])
  print(cv_df['diff'].sum()/df['diff'].sum())

  assert cv_df['diff'].sum()/df['diff'].sum() - 1 <= .05, "Cross-validated data should have roughly the same error as original model."