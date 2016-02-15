import pandas 
import statsmodels.formula.api as smf
import numpy as np

def generate_data():
    observation_count = 1000
    intercept = -1.6
    beta1 = -0.03
    beta2 = 0.1
    beta3 = -0.15
    variable_a = np.random.uniform(0, 100, size=observation_count)
    variable_b = np.random.uniform(50, 75, size=observation_count)
    variable_c = np.random.uniform(3, 10, size=observation_count)
    variable_d = np.random.uniform(3, 10, size=observation_count)
    variable_e = np.random.uniform(11, 87, size=observation_count)
    x = zip(variable_a, variable_b, variable_c, variable_d, variable_e)
    x_prime = [np.exp(intercept + beta1 * x_i[0] + beta2 * x_i[1] + beta3 * x_i[2]) / (1 + np.exp(intercept + beta1 * x_i[0] + beta2 * x_i[1] + beta3 * x_i[2])) for x_i in x]
    y = [np.random.binomial(1, x_prime_i, size=1)[0] for x_prime_i in x_prime]
    df = pandas.DataFrame({
        'variable_a':variable_a, 
        'variable_b': variable_b, 
        'variable_c': variable_c,
        'variable_d': variable_d,
        'variable_e': variable_e, 
        'y':y});
    return df

generate_data().to_csv('./generated_logistic_data.csv')