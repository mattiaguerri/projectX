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

import statsmodels.api as sm  
from statsmodels.tsa.stattools import acf  
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose

train, test = series[1:741], series[742:927]
train.shape
test.shape
plt.plot(train)
plt.plot(test)
plt.title("Training and Test Data")
plt.show()

model=sm.tsa.statespace.SARIMAX(endog=train,order=(1,0,0),seasonal_order=(2,1,0,12),trend='c',enforce_invertibility=False)
results=model.fit()
print(results.summary())

predictions=results.predict(741, 925, typ='levels')
predictions=np.exp(predictions)
test=np.exp(test)
mse=(predictions-test)/test
mse=abs(mse)
below10=mse[mse < 0.10].count()
plt.hist(below10)
plt.show()

all=mse.count()
accuracy=below10/all
