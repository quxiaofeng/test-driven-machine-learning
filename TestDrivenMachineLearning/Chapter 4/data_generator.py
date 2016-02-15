import numpy

variable_a = numpy.random.uniform(-100, 100, 30)
variable_b = numpy.random.uniform(-5, 5, 30)
variable_c = numpy.random.uniform(0, 37, 30)
variable_d = numpy.random.uniform(121, 213, 30)
variable_e = numpy.random.uniform(-1000, 100, 30)
variable_f = numpy.random.uniform(-100, 100, 30)
variable_g = numpy.random.uniform(-25, 75, 30)
variable_h = numpy.random.uniform(1, 27, 30)

independent_variables = zip(variable_a, variable_b, variable_c, variable_d, variable_e, variable_f, variable_g, variable_h)
dependent_variables = map(lambda x: 3*x[0] - 2*x[1] - .25*x[4] + 5.75*x[1]*x[2] + numpy.random.normal(0, 50), independent_variables)

full_dataset = map(lambda x: x[0] + (x[1],), zip(independent_variables, dependent_variables))

import csv
with open('generated_data.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['ind_var_a', 'ind_var_b', 'ind_var_c', 'ind_var_d', 'ind_var_e', 'ind_var_f', 'ind_var_g', 'ind_var_h', 'dependent_var'])
    writer.writerows(full_dataset)
