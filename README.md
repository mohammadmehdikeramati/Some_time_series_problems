# Some_time_series_problems
In this repository, I am gonaa to work on some data stes, which are available online (their links will be attached) with time series features.  
## First problem: hyperparameters tunning
In this attempt I investigated the effect of changing hyperparameters such as numbers of units, drop out and learning rate value, optimizer, time steps and also adding some dense layer instead of one layer in out put to acheive a better performance. The investigations were done on a public dataset, named Jena Climate Dataset (it is accessable via https://www.kaggle.com/datasets/mnassrib/jena-climate). It is a timeseries dataset recorded at the Weather Station of the Max Planck Institute in Jena, Germany. Also, the script of this part uploaded as 'Temperature Prediction'.
### First: changing numbers of units
As the results show As the results show decreasing numbers of units can cause the reduction rate of loss, mae, validation loss and validation mae to reduce. 
LSTM32:

![Normal results](https://user-images.githubusercontent.com/42337253/192144926-c5aac0bd-2a0a-4c5a-9269-d045e3452caf.PNG)




