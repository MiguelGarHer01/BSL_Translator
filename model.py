from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

'''
Sequential model definition for the expression classifier
'''


def get_model(num_class):
    model = Sequential()
    model.add(LSTM(64, input_shape=(20, 63), return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(256, return_sequences=True, activation='relu'))
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(num_class, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
