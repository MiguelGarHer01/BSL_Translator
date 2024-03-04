import numpy as np
import cv2


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

def image_preprocessing(frame, sentence):

    final_sentence = ""
    width = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (width, 45), (0, 0), -1)
    cv2.rectangle(frame, (0, 0), (width-2, 45), (255, 255, 255), 2)

    if sentence != "":

        for word in sentence:

            final_sentence = final_sentence + " " + word

        cv2.putText(frame, final_sentence, (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 1,
                    cv2.LINE_AA)

    return frame

