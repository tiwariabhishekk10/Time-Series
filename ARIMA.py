import pandas as pd
from pandas import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.model_selection import train_test_split
from pmdarima.arima import auto_arima

def parser(x):
    return datetime.strptime(x,'%d-%m-%Y')

#Indexing of dates is neccessary for times series anlysis
birth=pd.read_csv(r"D:\Time Series\ARIMA.csv",index_col=0,parse_dates=[0],date_parser=parser)
birth.head()
birth.plot()
#Stationary: mean, var and covar is contant over period
#Augmented Dickey fuller test to test wether data is stationary or not
result=adfuller(birth.Births)
result
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1]+)
plot_acf(birth)
help(adfuller)

help(auto)
#Model
model = auto_arima(birth, trace=True, error_action='ignore', suppress_warnings=True,stationary=True)
model.fit(birth)

forecast = model.predict(n_periods=len(birth))
forecast = pd.DataFrame(forecast,index = birth.index,columns=['Prediction'])
forecast
#plot the predictions for validation set
plt.plot(birth, label='birth')
plt.plot(forecast, label='Prediction')
plt.show()

#calculate rmse
from math import sqrt
from sklearn.metrics import mean_squared_error

rms = sqrt(mean_squared_error(birth,forecast))
print(rms)