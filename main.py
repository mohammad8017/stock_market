from os import sep
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



data = pandas.read_csv('ADANIPORTS.csv')

date, lastClose, open, high, low, last, close, volume = data['Date'], data['Prev Close'], data['Open'], data['High'], data['Low'], data['Last'], data['Close'], data['Volume']

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