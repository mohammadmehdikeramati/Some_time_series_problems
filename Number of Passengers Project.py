# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 17:33:43 2022

@author: Mehdi Keramati
"""


import numpy as np
import pandas as pd
from io import StringIO
import keras
from keras import layers
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

f = open('C:/Users/Mehdi Keramati/Desktop/RNN class/Session 3/airline-passengers.txt')

content = f.read()


data = StringIO(content)
Data = pd.read_csv(data, sep=",")

x=Data.drop('Month', axis=1)

num_train_samples = int(0.67 * len(x))
num_val_samples = int(0.17 * len(x))
num_test_samples = len(x) - num_train_samples - num_val_samples




##################  Normalization #################

'''
mean = x[:num_train_samples].mean(axis=0)
x -= mean
std = x[:num_train_samples].std(axis=0)
x /= std
'''

################# Transormation ####################

sampling_rate = 1
sequence_length = 10
delay = sampling_rate * (sequence_length) ######## -1 ????
batch_size = 10



test_1= x[:-delay]
test_2=x[delay:]



train_dataset = keras.utils.timeseries_dataset_from_array(
    x[:-delay],
    targets=x[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    
    batch_size=batch_size,
    
    
    start_index=0,
    end_index=num_train_samples)


val_dataset = keras.utils.timeseries_dataset_from_array(
    x[:-delay],
    targets=x[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    
    batch_size=batch_size,
    
      
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples)


test_dataset = keras.utils.timeseries_dataset_from_array(
    x[:-delay],
    targets=x[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples)




############## How about my classificxation problem ?????


######################### Model ###################

inputs = keras.Input(shape=(sequence_length, x.shape[-1]))
x = layers.LSTM(32, recurrent_dropout=0.5)(inputs)

#x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

model.compile(optimizer="adam", loss="mse", metrics=["mae"])
mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

history = model.fit(train_dataset,
                    epochs=300,
                    validation_data=val_dataset, callbacks=[es, mc])

loss = history.history['loss']
val_loss = history.history['val_loss']
mae = history.history['mae']
val_mae = history.history['val_mae']
epochs = range(1, len(loss) + 1)


########### Visualizing loss and mae ##############
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.figure()

plt.plot(epochs, mae, 'r', label='Training mae')
plt.plot(epochs, val_mae, 'b', label='Validation mae')
plt.title('Training and validation mae')
plt.legend()

plt.show()


############### Load model ##############
from keras.models import load_model
saved_model = load_model('best_model.h5')


############## Prediction ###############
for samples, targets in test_dataset:
    
    predication= saved_model.predict(samples)
    
    print("predictions:", predication)
    print("targets:", targets)
    plt.figure()
    plt.plot(range(len(predication)), targets,'r', label='Target')
    plt.plot(range(len(predication)), predication,'b', label='Prediction')
    plt.title('Prediction and target')
    plt.legend()
    plt.show()
    
    
    
    break   

