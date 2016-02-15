import numpy
import nose.tools
from sklearn.tree import DecisionTreeRegressor

def decision_tree_can_predict_perfect_linear_relationship_test():
    decision_tree = DecisionTreeRegressor()
    decision_tree.fit([[1],[1.1],[2]], [[0],[0],[1]])
    predicted_value = decision_tree.predict([[-1],[5]])
    print(predicted_value)
    assert list(predicted_value) == [0,1]

@nose.tools.raises(Exception)
def decision_tree_can_not_predict_strings_test():
    decision_tree = DecisionTreeRegressor()
    decision_tree.fit([[1],[1.1],[2]], [['class a'],['class a'],['class b']])
    predicted_value = decision_tree.predict([[-1],[5]])


def exploring_decision_trees_test():
    decision_tree = DecisionTreeRegressor()

    class_a_variable_a = numpy.random.normal(loc=51, scale=5, size=1000)
    class_a_variable_b = numpy.random.normal(loc=5, scale=1, size=1000)
    class_a_input = list(zip(class_a_variable_a, class_a_variable_b))
    class_a_label = [0]*len(class_a_input)

    class_b_variable_a = numpy.random.normal(loc=60, scale=7, size=1000)
    class_b_variable_b = numpy.random.normal(loc=8, scale=2, size=1000)
    class_b_input = list(zip(class_b_variable_a, class_b_variable_b))
    class_b_label = [1]*len(class_b_input)

    decision_tree.fit(class_a_input[:50] + class_b_input[:50],
                      class_a_label[:50] + class_b_label[:50])

    predicted_labels_for_class_a = decision_tree.predict(class_a_input[50:1000])
    predicted_labels_for_class_b = decision_tree.predict(class_b_input[50:1000])

    print("Class A correct: {0}; Class B correct: {1}".format(
            list(predicted_labels_for_class_a).count(0),
            list(predicted_labels_for_class_a).count(1)))

    assert list(predicted_labels_for_class_a).count(0) > list(predicted_labels_for_class_a).count(1), "For some reason when the decision tree guesses class a it's usually right way more than when it guesses class b."
