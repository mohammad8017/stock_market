from os import sep
from numpy.core.numeric import normalize_axis_tuple
import pandas
import matplotlib.pyplot as plt
import numpy as np


def findLower_Upper(data:list):
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    med = np.median(data)
    iqr = q3-q1
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)

    return lower_bound, upper_bound

def normalize(data):
    return [((item - min(data)) / ((max(data)-min(data)))) for item in data]




data = pandas.read_csv('ADANIPORTS.csv')

date, lastClose, open, high, low, last, close, volume = data['Date'], data['Prev Close'], data['Open'], data['High'], data['Low'], data['Last'], data['Close'], data['Volume']

#normalize values
lastClose = normalize(lastClose)
open = normalize(open)
high = normalize(high)
low = normalize(low)
last = normalize(last)
close = normalize(close)
volume = normalize(volume)


plt.subplot(1,3,1)
plt.plot(open)
plt.ylabel('price')
plt.title('open')

plt.subplot(1,3,2)
plt.plot(last)
plt.title('last')

plt.subplot(1,3,3)
plt.plot(close)
plt.title('close')

plt.show()

features = []
for i in range(len(date)):
    tempList = []
    tempList.append(lastClose[i])
    tempList.append(high[i])
    tempList.append(low[i])
    tempList.append(last[i])
    tempList.append(close[i])
    if close[i] - lastClose[i] >= 0:
        tempList.append('Positive')
    else:
        tempList.append('Negative')    
    features.append(tempList) 

print(features)       





    