import numpy as np


def normalize_expressions(expressions):
    expressions = np.asarray(expressions)

    return expressions


def normalize_labels(labels):

    # Create temp list to store all the integer values
    temp = []

    # Loop through all the labels and convert them from str into int
    for label in labels:
        temp.append(int(label))

    labels = np.array(temp)

    return labels


def number_classes(labels):
    return max(labels) + 1
