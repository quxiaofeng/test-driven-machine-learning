import libs.DecisionTree
import sklearn

def decision_tree_can_predict_perfect_linear_relationship_test():
    decision_tree = libs.DecisionTree.Classifier()
    observations = decision_tree.batch_train(((44, (1,2)), ((10, (45, 49)))))
    answer = decision_tree.classify((1,2))
    assert answer == 44, "Should be the answer it was trained on."
