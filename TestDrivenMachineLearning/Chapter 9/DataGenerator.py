import random, numpy, itertools, json
# Data generator

zipcode_options = ['60602', '60626', '98006']
gender_options = ['M', 'U', 'F']
option_combinations = set(reduce(lambda x,y: x+y,
                             [zip(x,gender_options)
                              for x in itertools.permutations(zipcode_options,
                                                              len(gender_options))]))
enhancers_by_segment = {
    ('60602', 'U'): {
                        'control':{'ordering': 1.00, 'profit': 1.00},
                        'variant':{'ordering': 1.35, 'profit': 1.20},
                    },
    ('98006', 'M'): {
                        'control':{'ordering': 1.00, 'profit': 1.00},
                        'variant':{'ordering': 1.00, 'profit': 1.00},
                    },
    ('60602', 'M'): {
                        'control':{'ordering': 1.00, 'profit': 1.00},
                        'variant':{'ordering': 1.45, 'profit': 1.24},
                    },
    ('60626', 'M'): {
                        'control':{'ordering': 1.00, 'profit': 1.00},
                        'variant':{'ordering': 0.6, 'profit': 0.8},
                    },
    ('60626', 'F'): {
                        'control':{'ordering': 1.00, 'profit': 1.00},
                        'variant':{'ordering': 1.00, 'profit': 1.00},
                    },
    ('60602', 'F'): {
                        'control':{'ordering': 1.00, 'profit': 1.00},
                        'variant':{'ordering': 1.00, 'profit': 1.00},
                    },
    ('98006', 'F'): {
                        'control':{'ordering': 1.10, 'profit': 1.25},
                        'variant':{'ordering': 1.10, 'profit': 1.30},
                    },
    ('60626', 'U'): {
                        'control':{'ordering': 1.00, 'profit': 1.00},
                        'variant':{'ordering': 1.00, 'profit': 1.00},
                    },
    ('98006', 'U'): {
                        'control':{'ordering': 1.00, 'profit': 1.00},
                        'variant':{'ordering': 1.00, 'profit': 1.00},
                    }
}

auto_customer_data = []
for i in range(10000):
    likelihood_to_order = 0.25
    profit_from_order = 12.25
    customer_data = {
        'zipcode': random.choice(zipcode_options),
        'gender': random.choice(gender_options),
        'orders_in_last_6_months': numpy.random.poisson(1.0),
        'customer_service_contacts': numpy.random.poisson(0.5),
        'profit_from_order': profit_from_order,
        'ordered': False,
        'treatment': 'control'
    }
    segment_key = (customer_data['zipcode'], customer_data['gender'])
    enhancer = enhancers_by_segment[segment_key]

    buff = enhancer['control']
    in_variant = numpy.random.binomial(1, 0.5)
    if in_variant:
        buff = enhancer['variant']
        customer_data['treatment'] = 'variant'
    likelihood_to_order = likelihood_to_order * buff['ordering'] \
                          * (1.0 + customer_data['orders_in_last_6_months']/10.0) \
                          / (1.0 + customer_data['customer_service_contacts'])
    ordered = numpy.random.binomial(1, likelihood_to_order)
    profit_from_order = profit_from_order * buff['profit'] \
                       - 0.5 * customer_data['customer_service_contacts'] \
                       + 0.25 * customer_data['orders_in_last_6_months']
    customer_data['ordered'] = (ordered == 1)
    customer_data['profit_from_order'] = profit_from_order if customer_data['ordered'] else 0.00

    #auto_customer_data[i] = customer_data
    auto_customer_data.append(customer_data)
json.dump(auto_customer_data, open('./fake_data.json', 'w'))
