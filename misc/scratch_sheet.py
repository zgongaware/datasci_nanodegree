
# Quiz 2.7.8
from math import log


def calculate_adaboost_weight(accuracy):
    return log(accuracy / (1 - accuracy))


m1 = calculate_adaboost_weight(7/8)
m2 = calculate_adaboost_weight(4/8)
m3 = calculate_adaboost_weight(2/8)
