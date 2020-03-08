import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from pandas import datetime
from keras.models import Sequential
from keras.layers import LSTM,GRU
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D


def parser(x):
    return datetime.strptime(x,'%d-%m-%Y')

####LSTM####

dataset=pd.read_csv(r"D:\Time Series Paper\stock_train.csv",index_col=0,parse_dates=[0],date_parser=parser)
dataset

open = dataset[dataset.columns[0:1]]
print(open.head())

data=np.array(open)

#Normalizing input features
scaler=MinMaxScaler(feature_range=(0,1))
scaled=scaler.fit_transform(data)

scaled=pd.DataFrame(scaled)
scaled.head(10)

data=np.array(scaled)

X = []
y = []
for i in range(60, len(scaled)):
    X.append(data[i-60:i, 0])#0:for all col
    y.append(data[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1],1))

###LSTM###

regressor_lstm = Sequential()
regressor_lstm.add(LSTM(units = 20,return_sequences = True, input_shape = (X.shape[1], 1)))
regressor_lstm.add(LSTM(units = 20,return_sequences = True))
regressor_lstm.add(LSTM(units = 20,return_sequences = True))
regressor_lstm.add(LSTM(units = 20))
regressor_lstm.add(Dense(units = 1))
regressor_lstm.compile(loss='mse', optimizer='adam')
regressor_lstm.summary()

#Fitting the model
history_lstm = regressor_lstm.fit(X, y, epochs=50, batch_size=30, validation_split=0.1 ,shuffle=False)

print(history_lstm)

plt.plot(history_lstm.history['loss'],label='LSTM train',color='red')
plt.plot(history_lstm.history['val_loss'],label='LSTM test',color='blue')
plt.title('model for LSTM')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

###GRU###
regressor_gru = Sequential() #object is created of class sequential

regressor_gru.add(GRU(units = 20, return_sequences = True, input_shape = (X.shape[1], 1)))
regressor_gru.add(GRU(units = 20, return_sequences = True))
regressor_gru.add(GRU(units = 20, return_sequences = True))
regressor_gru.add(GRU(units = 20))
regressor_gru.add(Dense(units = 1))
regressor_gru.compile(loss='mse', optimizer='adam')
regressor_gru.summary()

#Fitting the model
history_gru = regressor_gru.fit(X, y, epochs=50, batch_size=30, validation_split=0.1,  shuffle=False)

plt.plot(history_gru.history['loss'],label='GRU train',color='red')
plt.plot(history_gru.history['val_loss'],label='GRU test',color='blue')
plt.title('model for GRU')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

###CNN###
model = Sequential()

model.add(Conv1D(filters=20, kernel_size=2, activation='relu', input_shape=(X.shape[1],1)))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=20, kernel_size=2,activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=20, kernel_size=2,activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=20, kernel_size=2,activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.summary()

#Fitting the model
history_cnn = model.fit(X, y, epochs=50, batch_size=30, validation_split=0.1,  shuffle=False)

plt.plot(history_cnn.history['loss'],label='CNN train',color='yellow')
plt.plot(history_cnn.history['val_loss'],label='CNN test',color='violet')
plt.title('model for CNN')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

###MLP###Vanilla neural netwrok

regressor = Sequential()
regressor.add(Dense(units = 200, activation='relu', input_shape = (X.shape[1], 1)))
regressor.add(Flatten())
regressor.add(Dense(units = 1))
regressor.compile(loss='mse', optimizer='adam')
regressor.summary()

history = regressor.fit(X, y, epochs=50, batch_size=30, validation_split=0.1,  shuffle=False)

plt.plot(history.history['loss'],label='MLP train',color='blue')
plt.plot(history.history['val_loss'],label='MLP test',color='red')
plt.title('model for NN')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()