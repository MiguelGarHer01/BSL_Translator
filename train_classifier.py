import os
import tensorflow as tf
import pickle

from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from model import get_model
from keras import callbacks

from helper import normalize_expressions, normalize_labels, number_classes

EPOCHS = 80

data_dict = pickle.load(open('expressions.pickle', 'rb'))

expressions = normalize_expressions(data_dict['expressions'])
labels = normalize_labels(data_dict['labels'])

num_classes = number_classes(labels)

X_train, X_test, y_train, y_test = train_test_split(expressions, labels)

classifier = get_model(num_classes)

es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)

classifier.summary()

classifier.fit(X_train, y_train, epochs=EPOCHS, callbacks=[es])

classifier.evaluate(X_test, y_test)

predictions = classifier.predict(X_test)

print(predictions)

classifier.save('BSL_Expressions.keras')

