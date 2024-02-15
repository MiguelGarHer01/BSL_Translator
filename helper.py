import numpy as np


def normalize_expressions(expressions):
    expressions = np.asarray(expressions)

    return expressions


def normalize_labels(labels):
    labels = np.asarray(labels)

    return labels

def number_classes(labels):

    return max(labels) + 1
