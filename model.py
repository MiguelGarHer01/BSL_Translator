from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout


'''
Sequential model definition for the expression classifier
'''
def get_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=(20, 42), return_sequences=True))
    model.add(LSTM(32, return_sequences=True))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(input_shape, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
