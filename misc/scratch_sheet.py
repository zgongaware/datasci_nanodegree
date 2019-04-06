
# Quiz 2.7.8 - ADABoost Weight
from math import log


def calculate_adaboost_weight(accuracy):
    return log(accuracy / (1 - accuracy))


m1 = calculate_adaboost_weight(7/8)
m2 = calculate_adaboost_weight(4/8)
m3 = calculate_adaboost_weight(2/8)

# Quiz 2.8.13 - F1 Score


def calculate_f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)

a = calculate_f1_score(0.556, 0.833)