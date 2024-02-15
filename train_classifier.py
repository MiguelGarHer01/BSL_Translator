import os
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
from model import get_model

from helper import normalize_expressions, normalize_labels, number_classes

EPOCHS = 150

data_dict = pickle.load(open('expressions.pickle', 'rb'))

expressions = normalize_expressions(data_dict['expressions'])
labels = normalize_labels(data_dict['labels'])

num_classes = number_classes(labels)

X_train, X_test, y_train, y_test = train_test_split(expressions, labels)

classifier = get_model(num_classes)

classifier.summary()

classifier.fit(X_train, y_train, epochs=EPOCHS)

classifier.evaluate(X_test, y_test)

predictions = classifier.predict(X_test)

print(predictions)


