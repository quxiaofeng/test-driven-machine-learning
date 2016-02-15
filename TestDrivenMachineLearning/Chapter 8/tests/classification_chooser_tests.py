import nose.tools, numpy
from choosey import *
import libs.NaiveBayes
import libs.RandomForest
import libs.DecisionTree

@nose.tools.raises(Exception)
def given_no_model_options_test():
    classifier_chooser = ClassifierChooser()

def given_a_single_classifier_option_that_does_not_require_training_test():
    classifier_chooser = ClassifierChooser.create_with_single_classifier_option(CopyCatClassifier())
    input_value = 42
    predicted_label = classifier_chooser.classify(input_value)
    assert predicted_label == input_value, "Should predict input value."

    input_value = 11
    predicted_label = classifier_chooser.classify(input_value)
    assert predicted_label == input_value, "Should predict input value."

def given_a_different_single_classifier_option_that_does_not_require_training_test():
    classifier_chooser = ClassifierChooser.create_with_single_classifier_option(AlwaysTrueClassifier())
    input_value = 42
    predicted_label = classifier_chooser.classify(input_value)
    assert predicted_label, "Should always predict True."

def given_a_CopyCatClassifier_test():
    classifier = CopyCatClassifier()
    input_value = 12.5
    predicted_label = classifier.classify(input_value)
    assert predicted_label == input_value, "Should predict the value to be what the input is."

    input_value = 77
    predicted_label = classifier.classify(input_value)
    assert predicted_label == input_value, "Should predict the value to be what the input is."

def given_an_AlwaysTrueClassifier_test():
    classifier = AlwaysTrueClassifier()
    predicted_label = classifier.classify(55)
    assert predicted_label == 1, "Should always predict one."

def given_an_AlwaysFalseClassifier_test():
    classifier = AlwaysFalseClassifier()
    predicted_label = classifier.classify(55)
    assert predicted_label == 0, "Should always predict zero."

def given_multiple_classifier_options_test():
    classifier_chooser = ClassifierChooser(classifier_options_list=[
            AlwaysTrueClassifier(),
            AlwaysFalseClassifier()
        ],
        test_input=[78],
        test_label=[1])
    predicted_label = classifier_chooser.classify(0)
    assert predicted_label == 1, "Should choose best classifier option to classify with."

def given_multiple_classifier_options_and_several_test_data_test():
    classifier_chooser = ClassifierChooser(classifier_options_list=[
            AlwaysFalseClassifier(),
            AlwaysTrueClassifier()
        ],
        test_input=[78,22,12],
        test_label=[1,1,0])
    predicted_label = classifier_chooser.classify(0)
    assert predicted_label == 1, "Should choose best classifier option to classify with."


def given_multiple_classifier_options_and_several_test_data_with_training_test():
    classifier_chooser = ClassifierChooser(classifier_options_list=[
            DictionaryClassifier(),
            AlwaysFalseClassifier(),
            AlwaysTrueClassifier()
        ],
        test_input=[(1,2), (3,4)],
        test_label=[3,7],
        training_inputs=[(1, 2), (3, 4), (5, 6)],
        training_labels=[3, 7, 11])
    predicted_label = classifier_chooser.classify((5,6))
    assert predicted_label == 11, "Should choose best classifier option to classify with."

def given_a_dictionary_classifier_test():
    classifier = DictionaryClassifier()
    classifier.batch_train([
        (42, (1,2,3)),
        (2, (2,3,4)),
    ])
    assert classifier.classify((1,2,3)) == 42
    assert classifier.classify((2,3,4)) == 2

def given_real_classifiers_and_random_data_test():
    class_a_variable_a = numpy.random.normal(loc=51, scale=5, size=1000)
    class_a_variable_b = numpy.random.normal(loc=5, scale=1, size=1000)
    class_a_input = list(zip(class_a_variable_a, class_a_variable_b))
    class_a_label = [0]*len(class_a_input)

    class_b_variable_a = numpy.random.normal(loc=60, scale=7, size=1000)
    class_b_variable_b = numpy.random.normal(loc=8, scale=2, size=1000)
    class_b_input = list(zip(class_b_variable_a, class_b_variable_b))
    class_b_label = [1]*len(class_b_input)

    classifier_chooser = ClassifierChooser(classifier_options_list=[
            CopyCatClassifier(),
            libs.NaiveBayes.Classifier(),
            libs.RandomForest.Classifier(),
            libs.DecisionTree.Classifier()
        ],
        test_input=class_a_input[50:500] + class_b_input[50:500],
        test_label=class_a_label[50:500] + class_b_label[50:500],
        training_inputs=class_a_input[:50] + class_b_input[:50],
        training_labels=class_a_label[:50] + class_b_label[:50])
    print(classifier_chooser._classifier_options)
    assert True
