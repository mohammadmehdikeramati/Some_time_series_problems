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

And it is worth mentioning the results depict even though rmsprop optimizer had better performance on training data, validation loss and validation mae reduction was better when adam were implemented. It can be concluded, using adam can reduce the chance of overfitting.

LSTM32 and rmsprop:

![Results (LSTM32 and rmsprop and 10 epochs)](https://user-images.githubusercontent.com/42337253/192146051-c3029679-4816-4f1d-871e-d9a02a18ca18.PNG)

LSTM32 and adam:

![Results (LSTM32 and adam and 10 epochs)](https://user-images.githubusercontent.com/42337253/192146067-69f14f80-4315-44c1-8ae0-e4b9e8e538e1.PNG)

### Third: Utilizing several dense layers
As the results show, adding some dense layers (respectively 128,64 and 32) can improve the performance of our architecture.

LSTM32 and a dense (one unit) output layer: 

![Results (LSTM32 and rmsprop and 10 epochs)](https://user-images.githubusercontent.com/42337253/192146149-4cb8105d-9e27-4fbc-a5d7-6ed3a55873b6.PNG)

LSTM32 and adding three extra dense (respectively 128, 64 and 32) layers: 

![Results (LSTM32 and two denses128 64 32 and 10 epochs)](https://user-images.githubusercontent.com/42337253/192146197-23befc3b-41ca-4693-9a3a-fcaa7b226ea8.PNG)

 ### Fourth: Changing sequence length
Comparison among different sequence lengths illustrate that longer sequence length not only increase the process time, but also raise the loss and mae on both training and validation set. In fact, considering longer sequence length required changing other hyper parameters such as adding extra layers to our architecture, otherwise it may reduce its performance.

LSTM32 and sequence length 120:

![Results (LSTM32 and rmsprop and 10 epochs)](https://user-images.githubusercontent.com/42337253/192146899-2439e1c1-6a26-4e89-ae10-69cea10ee73e.PNG)

LSTM32 and sequence length 80:

![Results (LSTM32-and sequence80 and 10epochs)](https://user-images.githubusercontent.com/42337253/192146959-04a09035-9609-4604-aca1-40255eb898ea.PNG)

LSTM32 and sequence length 160:

![Results (LSTM32-and sequence160 and 10epochs)](https://user-images.githubusercontent.com/42337253/192146979-e2ec6a52-d43a-4361-ab0e-e2b67ffbf6e4.PNG)

## Second problem: using dense layer instead of LSTM to analyze a time series
In this section I used a dense layer instead of LSTM to predict last section time series. The results demonstrated a huge difference. This section code uploaded as 'Dense Instead of LSTM'.

LSTM32:

![Results (LSTM32 and rmsprop and 10 epochs)](https://user-images.githubusercontent.com/42337253/192147355-342be303-ff9b-4fa2-bd35-f96e9d396f85.PNG)

Dense instead of LSTM32:

![Dense instead of LSTM](https://user-images.githubusercontent.com/42337253/192147414-162e7f37-b6ec-49ee-85f7-552828a9544d.PNG)

## Third problem: Comparison among LSTM, GRU and SimpleRNN performance in a sine wave prediction. 
In this part, I tried to make a comparison among performance of LSTM, GRU and SimpleRNN via constracting a sine wave and test these algorithms performance in prediction of the constructed wave. The results show the better performance of GRU. This issue was according our expectation but GRU better convergence and lower loss, mae, validation loss and validation mae in comparison with LSTM was a bit strange. I think this is because this series is not complicated enough and therefore it is no need to utilize LSTM. The script of this section uploaded as 'Sin Project'.

GRU:

![GRU-Sin project result](https://user-images.githubusercontent.com/42337253/192147775-e1493d30-673f-42c8-9ec5-86e44dc6a0a8.PNG)

LSTM:

![LSTM-Sin project result](https://user-images.githubusercontent.com/42337253/192147791-aff2434f-b703-4187-92f5-bc86912b021a.PNG)

SimpleRNN:

![SimpleRnn-Sin project result](https://user-images.githubusercontent.com/42337253/192147806-9d3d3744-211b-4952-886f-f2a4c3c173cd.PNG)

## Fourth problem: Forecasting numbers of an international flight passengers between 1949 and 1960.
This is a time series anticipation project. It seems there is not fundamental differences among this dataset and the previous one but implementing last scrip (with little changes) did not show as good result as last part. In fact, even though by increasing epochs the loss, mae, validation loss and validation mae reduced, the algorithm could not convergent within 300 epochs. This issue is because dataset values were huge. Hence, I decided to normalize data and you can see the effect of normalization on convergence below. This part code uploaded as 'Number of Passengers Project'.

LSTM training and validation loss without normalizing: 

![LSTM loss-300epochs- whitout normalizing](https://user-images.githubusercontent.com/42337253/192319925-e378af55-ea06-4caf-8693-c9af1155eb26.png)

LSTM training and validation mae without normalizing: 

![LSTM mae-300epochs- whitout normalizing](https://user-images.githubusercontent.com/42337253/192322309-0afd13e4-8f80-43b2-a4a0-70428e94c2ba.png)


LSTM prediction without normalizing:

![LSTM prediction-300epochs whitout normalizing](https://user-images.githubusercontent.com/42337253/192321571-f06beff1-7bce-429f-83b7-f6418e0d4565.png)

LSTM training and validation loss with normalizing:

![LSTM loss-300epochs](https://user-images.githubusercontent.com/42337253/192337314-dd18a541-2a58-44a8-9729-6387dfd91b56.png)


LSTM training and validation mae with normalizing: 

![LSTM mae-300epochs](https://user-images.githubusercontent.com/42337253/192337463-fff9844d-0899-4bc2-80ca-3a36240d3418.png)


LSTM prediction with normalizing:

![LSTM prediction-300epochs- DO0 5](https://user-images.githubusercontent.com/42337253/192337605-16bc5aa4-3352-4b68-b4e2-599c72ba8c04.png)

I also tried to improve the performance of my network via add drop out layer and recurrent drop out in LSTM layer, but actually this issue did not improve the performance of our network at all.

Previous architecture with adding a drop out (0.5) layer:

![LSTM -300epochs- DO0 5](https://user-images.githubusercontent.com/42337253/192339643-0eeb668a-9093-4b36-a2c9-8c4d035b1b72.png)

![LSTM loss-300epochs- DO0 5](https://user-images.githubusercontent.com/42337253/192339685-657be1f7-cbc2-4491-aca2-831d957ac136.png)

![LSTM mae-300epochs- DO0 5](https://user-images.githubusercontent.com/42337253/192339740-f1237f8c-591f-47a8-9ae4-737a8a7985eb.png)

![LSTM prediction-300epochs- DO0 5](https://user-images.githubusercontent.com/42337253/192339859-4fdb7910-5eec-4166-9a3b-1bf0bb61f521.png)


Previous architecture with adding recurrent drop out (0.5) in LSTM layer:

![LSTM -300epochs- RDO0 5](https://user-images.githubusercontent.com/42337253/192339858-58efddb9-5853-40af-b44e-d50c91a17892.png)

![LSTM loss-300epochs- RDO0 5](https://user-images.githubusercontent.com/42337253/192339913-ac992cb7-c6fa-4892-9642-d343e09733ae.png)

![LSTM mae-300epochs- RDO0 5](https://user-images.githubusercontent.com/42337253/192340040-acdc053d-3bb7-4b34-8404-b31ef4799f82.png)

![LSTM prediction-300epochs- RDO0 5](https://user-images.githubusercontent.com/42337253/192340056-31e007e3-08fa-40f9-9fdd-58ca8cdaab29.png)




