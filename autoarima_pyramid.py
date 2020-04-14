# Import Libraries
import csv
import math
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import psycopg2
import pyramid
import random
import seaborn as sns
import statsmodels.tsa.stattools as ts
from pyramid.arima import auto_arima
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

import os;
path="directory"
os.chdir(path)
os.getcwd()

from pandas import Series
from matplotlib import pyplot
series = Series.from_csv('mly532.csv', header=0)
series = np.log(series)
series.plot()
plt.xlabel('Year')
plt.ylabel('Maximum Temperature')
plt.title('Maximum Air Temperature in Dublin, Ireland')
pyplot.show()

decomposition=seasonal_decompose(series, model='multiplicative')
trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid

plt.subplot(221)
plt.plot(series,color='#ff0000', label='Series')
plt.legend(loc='best')
plt.subplot(222)
plt.plot(trend,color='#1100ff', label='Trend')
plt.legend(loc='best')
plt.tight_layout()
plt.subplot(223)
plt.plot(residual,color='#00ff1a', label='Residual')
plt.legend(loc='best')
plt.tight_layout()
plt.subplot(224)
plt.plot(seasonal,color='#de00ff', label='Seasonality')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

from pyramid.arima.stationarity import ADFTest
adf_test = ADFTest(alpha=0.05)
adf_test.is_stationary(series)

train, test = series[1:741], series[742:927]
train.shape
test.shape
plt.plot(train)
plt.plot(test)
plt.title("Training and Test Data")
plt.show()

Arima_model=auto_arima(train, start_p=1, start_q=1, max_p=8, max_q=8, start_P=0, start_Q=0, max_P=8, max_Q=8, m=12, seasonal=True, trace=True, d=1, D=1, error_action='warn', suppress_warnings=True, random_state = 20, n_fits=30)

Arima_model.summary()

prediction=pd.DataFrame(Arima_model.predict(n_periods=185), index=test.index)
prediction.columns = ['Predicted_Temperature']

plt.figure(figsize=(15,10))
plt.plot(train, label='Training')
plt.plot(test, label='Test')
plt.plot(prediction, label='Predicted')
plt.legend(loc = 'upper center')
plt.show()

predictions=prediction['Predicted_Temperature']
predictions=np.exp(predictions)
test=np.exp(test)
mse=(predictions-test)/test
np.mean(mse)
mse=abs(mse)
below10=mse[mse < 0.10].count()
all=mse.count()
accuracy=below10/all
