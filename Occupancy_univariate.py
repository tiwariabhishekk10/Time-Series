import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from pandas import datetime
from keras.models import Sequential
from keras.layers import LSTM,GRU
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Activation
from math import sqrt

def parser(x):
    return datetime.strptime(x,'%d-%m-%Y %H:%M')

def rmse(y_true,y_pred):
    return K.sqrt(K.mean(K.square(y_pred-y_true),axis=-1))

url='https://raw.githubusercontent.com/tiwariabhishekk10/Time-Series/master/Occupancy.csv'
dataset=pd.read_csv(url,index_col=0,parse_dates=[0],date_parser=parser)
dataset.head(10)
temperature = dataset[dataset.columns[0:1]]
print(temperature.head())
values=np.array(temperature)

#Normalizing input features
scaler=MinMaxScaler(feature_range=(0,1))
scaled=scaler.fit_transform(values)

scaled=pd.DataFrame(scaled)
scaled.head(10)

train=scaled[:int(0.9*(len(scaled)))]
train.shape
train=np.array(train)

valid=scaled[int(0.9*(len(scaled))):]
valid.shape
valid=np.array(valid)

X_train = []
y_train = []
for i in range(200, len(train)):
    X_train.append(train[i-200:i, 0])
    y_train.append(train[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

X_test = []
y_test = []
for i in range(200, len(valid)):
    X_test.append(valid[i-200:i, 0])
    y_test.append(valid[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

###LSTM###
regressor_lstm = Sequential()
regressor_lstm.add(LSTM(units = 32,return_sequences = True, activation='relu',input_shape = (X_train.shape[1], X_train.shape[2])))
regressor_lstm.add(Dropout(0.2))
regressor_lstm.add(LSTM(units = 32,activation='relu',return_sequences = True))
regressor_lstm.add(Dropout(0.2))
regressor_lstm.add(LSTM(units = 32,activation='relu',return_sequences = True))
regressor_lstm.add(Dropout(0.2))
regressor_lstm.add(LSTM(units = 32,activation='relu'))
regressor_lstm.add(Dropout(0.2))
regressor_lstm.add(Dense(units = 1,activation='linear'))
regressor_lstm.compile(loss='mse', optimizer='adam',metrics=[rmse])
regressor_lstm.summary()

#Fitting the model
history_lstm = regressor_lstm.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test,y_test),  shuffle=False)

plt.plot(history_lstm.history['rmse'],label='LSTM train',color='red')
plt.plot(history_lstm.history['val_rmse'],label='LSTM test',color='blue')
plt.title('model for LSTM')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

###GRU###

regressor_gru = Sequential()
regressor_gru.add(GRU(units = 32,return_sequences = True,activation='relu', input_shape = (X_train.shape[1], X_train.shape[2])))
regressor_gru.add(Dropout(0.2))
regressor_gru.add(GRU(units = 32,activation='relu',return_sequences = True))
regressor_gru.add(Dropout(0.2))
regressor_gru.add(GRU(units = 32,activation='relu',return_sequences = True))
regressor_gru.add(Dropout(0.2))
regressor_gru.add(GRU(units = 32,activation='relu'))
regressor_gru.add(Dropout(0.2))
regressor_gru.add(Dense(units = 1,activation='linear'))
regressor_gru.compile(loss='mse', optimizer='adam',metrics=[rmse])
regressor_gru.summary()

#Fitting the model
history_gru = regressor_gru.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test,y_test),  shuffle=False)

plt.plot(history_gru.history['rmse'],label='GRU train',color='green')
plt.plot(history_gru.history['val_rmse'],label='GRU test',color='black')
plt.title('model for GRU')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

###CNN###
model = Sequential()

model.add(Conv1D(filters=20, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=20, kernel_size=2,activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=20, kernel_size=2,activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(filters=20, kernel_size=2,activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1,activation='linear'))

model.compile(loss='mse', optimizer='adam',metrics=[rmse])
model.summary()

#Fitting the model
history_cnn = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test,y_test),  shuffle=False)

plt.plot(history_cnn.history['rmse'],label='CNN train',color='green')
plt.plot(history_cnn.history['val_rmse'],label='CNN test',color='black')
plt.title('model for CNN')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

###MLP### Vanilla neural netwrok

regressor = Sequential()
regressor.add(Dense(units = 200, activation='relu', input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Flatten())
regressor.add(Dense(units = 1,activation='linear'))
regressor.compile(loss='mse', optimizer='adam',metrics=[rmse])
regressor.summary()

history = regressor.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test,y_test),  shuffle=False)

plt.plot(history.history['rmse'],label='MLP train',color='blue')
plt.plot(history.history['val_rmse'],label='MLP test',color='red')
plt.title('model for NN')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()