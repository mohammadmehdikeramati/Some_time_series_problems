# Description
In this repository, I am gonaa to work on some data stes, which are available online (their links will be attached) with time series features.  
## First problem: hyperparameters tunning
In this attempt I investigated the effect of changing hyperparameters such as numbers of units, drop out and learning rate value, optimizer, time steps and also adding some dense layer instead of one layer in out put to acheive a better performance. The investigations were done on a public dataset, named Jena Climate Dataset (it is accessable via https://www.kaggle.com/datasets/mnassrib/jena-climate). It is a timeseries dataset recorded at the Weather Station of the Max Planck Institute in Jena, Germany. Also, the script of this part uploaded as 'Temperature Prediction'.
### First: changing numbers of units
As the results show As the results show decreasing numbers of units can cause the reduction rate of loss, mae, validation loss and validation mae to reduce. 

LSTM32:

![Normal results](https://user-images.githubusercontent.com/42337253/192144926-c5aac0bd-2a0a-4c5a-9269-d045e3452caf.PNG)

LSTM16:

![Results (LSTM16)](https://user-images.githubusercontent.com/42337253/192145101-9750c954-d8b9-481c-b2d3-b6ba9bffd8fa.PNG)

LSTM2:

![Results (LSTM2)](https://user-images.githubusercontent.com/42337253/192145120-1faac9ba-7fb9-4b2d-834a-4a73261f43ae.PNG)

But increasing numbers of units can be an effective way if overfit phenomenon not to be happened. Hence, drop out layers and also recurrent drop out in LSTM layers should be added to prevent overfit along with improve the reduction rate of loss, mae, mse, validation loss and validation mae.

LSTM64:

![Results (LSTM64)](https://user-images.githubusercontent.com/42337253/192145272-70771191-7586-458b-ac86-60de4cf5311d.PNG)

LSTM64 and drop out 0.75 and recurrent drop pout 0.5:

![Results (LSTM64 and D0 75 and R D0 5)](https://user-images.githubusercontent.com/42337253/192145321-5d86ce48-c7b5-4a7c-ba53-5d7449e2a023.PNG)

LSTM64 and drop out 0.75 and recurrent drop pout 0.75:

![Results-best (LSTM64 and D0 75 and R D0 75)](https://user-images.githubusercontent.com/42337253/192145341-1dd77df3-a8eb-4e18-9f5e-fa78c96f6598.PNG)

### Second: Changing learning rate of rmsprop and then utilizing adam instead of it
Based on our investigation considering 0.001 as learning rate value for rmsprop act better than other learning rate values.

LSTM32 and default learning rate value:

![Normal results](https://user-images.githubusercontent.com/42337253/192145527-0d49cb09-202d-477e-8de1-4092986cfae5.PNG)

LSTM32 and learning rate 0.1:

![Results (LSTM32 and lr0 1)](https://user-images.githubusercontent.com/42337253/192145599-7f234a08-54cc-4b9f-b143-56fa955423b0.PNG)

LSTM32 and learning rate 0.01:

![Results (LSTM32 and lr0 01)](https://user-images.githubusercontent.com/42337253/192145637-3e9947a3-ad8f-4040-b09c-117e5b34888b.PNG)

LSTM32 and learning rate 0.001:

![Results- best (LSTM32 and lr0 001)](https://user-images.githubusercontent.com/42337253/192145796-574ef178-5f85-4aed-a722-51f18a09e1ab.PNG)

LSTM32 and learning rate 0.0001:

![Results (LSTM32 and lr0 0001)](https://user-images.githubusercontent.com/42337253/192145732-1ba3b010-7a7d-44c8-8e0e-7ba2f364316e.PNG)

LSTM32 and learning rate 0.00001:

![Results (LSTM32 and lr0 00001)](https://user-images.githubusercontent.com/42337253/192145768-893feb50-dc89-45da-ab42-6dbf7c859067.PNG)



