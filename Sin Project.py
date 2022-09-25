# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 22:05:06 2022

@author: Mohammad Mehdi Keramati
"""


import keras
import numpy as np
import pandas as pd

import keras
from keras import layers

t = np.arange(1500)
x = np.sin(0.08 * t)

x=pd.DataFrame(x)

num_train_samples = int(0.5 * len(x))
num_val_samples = int(0.25 * len(x))
num_test_samples = len(x) - num_train_samples - num_val_samples



sampling_rate = 1
sequence_length = 100
delay = sampling_rate * (sequence_length) ######## -1 ????
batch_size = 10



test_1= x[:-delay]
test_2=x[delay:]



train_dataset = keras.utils.timeseries_dataset_from_array(
    x[:-delay],
    targets=x[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    
    
    start_index=0,
    end_index=num_train_samples)


val_dataset = keras.utils.timeseries_dataset_from_array(
    x[:-delay],
    targets=x[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    
      
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples)

test_dataset = keras.utils.timeseries_dataset_from_array(
    x[:-delay],
    targets=x[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    
    
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples)



######################### Model ########################################

inputs = keras.Input(shape=(sequence_length, x.shape[-1]))
x = layers.GRU(32, recurrent_dropout=0.25)(inputs)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)


model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = model.fit(train_dataset,
                    epochs=10,
                    validation_data=val_dataset)
