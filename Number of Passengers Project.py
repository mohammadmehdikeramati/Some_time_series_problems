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

#################### Visualization #####################

'''
plt.plot(range(len(x)), x)
plt.show()
'''


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


######################### Model ########################################

inputs = keras.Input(shape=(sequence_length, x.shape[-1]))
x = layers.LSTM(32)(inputs)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)


model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(train_dataset,
                    epochs=100,
                    validation_data=val_dataset)



for samples, targets in test_dataset:
    
    predication= model.predict(samples)
    
    print("samples shape:", samples)
    print("targets shape:", targets)
    plt.plot(range(len(predication)), predication)
    plt.plot(range(len(targets)), targets)
    plt.show()
    
    
    
    break   

