import json
from sklearn.linear_model import LogisticRegression
import statsmodels.api

treatment_codes = {
    'control': 1,
    'variant': 2
}
zipcode_codes = {
    '60626': 1,
    '60602': 2,
    '98006': 3
}
gender_codes = {
    'M': 1,
    'F': 2,
    'U': 3
}

def create_profit_inputs_and_outputs(fake_data):
    test_inputs = []
    for x in fake_data:
        if not x['ordered']:
            continue
        input = (treatment_codes[x['treatment']],
           zipcode_codes[x['zipcode']],
           gender_codes[x['gender']],
           x['orders_in_last_6_months'],
           x['customer_service_contacts'])
        test_inputs.append(input)

    test_labels = [x['profit_from_order'] for x in fake_data if x['ordered']]
    return test_labels, test_inputs

def create_order_inputs_and_outputs(fake_data):
    test_inputs = []
    for x in fake_data:
        input = (treatment_codes[x['treatment']],
           zipcode_codes[x['zipcode']],
           gender_codes[x['gender']],
           x['orders_in_last_6_months'],
           x['customer_service_contacts'])
        test_inputs.append(input)

    test_labels = [x['ordered'] for x in fake_data]
    return test_labels, test_inputs

def logistic_regression_hello_world_test():
    fake_data = json.load(open('./fake_data.json', 'r'))
    test_order_labels, test_order_inputs = create_order_inputs_and_outputs(fake_data)
    model = LogisticRegression()
    fitted_model = model.fit(test_order_inputs, test_order_labels)
    print(fitted_model)
    print(model.score(test_order_inputs, test_order_labels))
    print(model.predict_proba(test_order_inputs))
    #assert False

def linear_regression_test():
    fake_data = json.load(open('./fake_data.json', 'r'))
    test_labels, test_inputs = create_profit_inputs_and_outputs(fake_data)
    profit_model = statsmodels.api.OLS(test_labels, test_inputs)
    results = profit_model.fit()
    print(results.summary())
    #assert False

def combined_test():
    fake_data = json.load(open('./fake_data.json', 'r'))
    test_order_labels, test_order_inputs = create_order_inputs_and_outputs(fake_data)
    test_profit_labels, test_profit_inputs = create_profit_inputs_and_outputs(fake_data)

    order_model = LogisticRegression()
    fitted_order_model = order_model.fit(test_order_inputs, test_order_labels)

    profit_model = statsmodels.api.OLS(test_profit_labels, test_profit_inputs)
    fitted_profit_model = profit_model.fit()

    for input in test_order_inputs:
        order_probability = fitted_order_model.predict_proba(input)[0][1]
        profit_if_ordered = fitted_profit_model.predict(input)[0]
        print(input, order_probability * profit_if_ordered)
    #assert False
