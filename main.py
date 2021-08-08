from os import sep
from numpy.core.numeric import normalize_axis_tuple
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split



#-----------------------find q1, q3 of box plot(doesn't use)---------------------------
def findLower_Upper(data:list):
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    med = np.median(data)
    iqr = q3-q1
    upper_bound = q3+(1.5*iqr)
    lower_bound = q1-(1.5*iqr)

    return lower_bound, upper_bound

#-----------------------function for normalize values---------------------------
def normalize(data):
    return [((item - min(data)) / ((max(data)-min(data)))) for item in data]



#-----------------------read data from file---------------------------
data = pandas.read_csv('ADANIPORTS.csv')

date, lastClose, open, high, low, last, close, volume, Turnover = data['Date'], data['Prev Close'], data['Open'], data['High'], data['Low'], data['Last'], data['Close'], data['Volume'], data['Turnover']
volume = volume.astype(float)

#-----------------------normalize values---------------------------
lastClose = normalize(lastClose)
open = normalize(open)
high = normalize(high)
low = normalize(low)
last = normalize(last)
close = normalize(close)
volume = normalize(volume)
Turnover = normalize(Turnover)

#-----------------------visualization---------------------------
# plt.subplot(1,3,1)
# plt.plot(open)
# plt.ylabel('price')
# plt.title('open')

# plt.subplot(1,3,2)
# plt.plot(last)
# plt.title('last')

# plt.subplot(1,3,3)
# plt.plot(close)
# plt.title('close')

# plt.show()


#-----------------------create and set value for features and labels---------------------------
label = []
features = []

for i in range(len(close)):
    if close[i] - lastClose[i] > 0:
        label.append('Positive')
    else: 
        label.append('Negative')  
label.pop(-1)

for i in range(len(date)):
    tempList = [] 
    # tempList.append(high[i])
    # tempList.append(low[i])
    tempList.append(last[i])
    tempList.append(close[i])
    tempList.append(volume[i])
    tempList.append(Turnover[i])
    try: #for handle error of divide by zero
        tempList.append(volume[i]/Turnover[i])
    except:
        tempList.append(0.0) 
    try:
        tempList.append(close[i]/last[i])
    except:
        tempList.append(0.0)    

    features.append(tempList)
features.pop(-1)    



features = np.array(features)
y = np.array(label)

#-----------------------create test array and train array (80% of data => train)---------------------------

X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.09)
print('shape of y_train:', y_train.shape)
print('shape of y_test:', y_test.shape)
print('=======================')


#-----------------------use KNN algorithm and predict values---------------------------
scores = {}

for i in range(1,31):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)

    scores[i] = metrics.accuracy_score(y_test, pred)

print('result: (k : predicted result)')
print(scores)
print('=======================')

Keymax = max(scores, key=scores.get)
print(str(Keymax),':',scores[Keymax])

#-----------------------plot result of KNN---------------------------
plt.plot(range(1,31), scores.values())
plt.show()


    