import tensorflow as tf
import keras 
from keras import layers 
from tensorflow import Sequential
from tensorflow import Dense, LSTM


#need to declare sequential, dense, and lstm models 
def create_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
