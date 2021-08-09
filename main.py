from os import sep
from numpy.core.numeric import normalize_axis_tuple
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from itertools import combinations



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

#-----------------------function for combination of lists---------------------------
def comb(lst:list):
    allFeatures = []
    combs = []
    for i in range(len(lst)):
        l = list(combinations(lst, i+1))
        for j in range(len(l)):
            tmp = []
            tmpStr = ''
            for k in range(i+1):
                indexList = l[j][k]
                if k == 0:
                    for m in range(len(indexList)):
                        tmp.append([indexList[m]])
                else:
                    for m in range(len(indexList)):
                        tmp[m].append(indexList[m])
                tmpStr += str(lst.index(indexList)) + ' '        
            tmp.pop(-1)
            allFeatures.append(np.array(tmp))   
            combs.append(tmpStr)
    return allFeatures, combs       
                
#-----------------------function for create model---------------------------
def createModel(allFeatures, y, combs):
    Max = -1
    bestFeature = ''
    predMax = -1
    predTmp = ''
    featureMax = ''
    j = 0
    for feature in allFeatures:
        X_train, X_test, y_train, y_test = train_test_split(feature, y, test_size=0.09)
        print(combs[j], ':')

        scores = {}
        for i in range(1,31):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            pred = knn.predict(X_test)

            scores[i] = metrics.accuracy_score(y_test, pred)
            if predMax < scores[i]:
                predMax = scores[i]
                predTmp = pred

        Keymax = max(scores, key=scores.get)
        print(str(Keymax),':',scores[Keymax])
        if scores[Keymax] > Max: 
            Max = scores[Keymax]
            bestFeature = combs[j]
            featureMax = y_test
        print('=======================')
        j += 1
    print('number of all possible features:', len(allFeatures))    
    print('best fatures:', bestFeature)
    print('max of all:', Max)

    for i in range(len(predTmp)):
        if predTmp[i] == featureMax[i]:
            plt.plot(featureMax[i], 'bo', markersize=1)
        else:
            plt.plot(predTmp[i], 'ro', markersize=1)

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


#-------------------------------initial value for volome/turnover-----------------------------------------

volume_Turnover = []
close_last = []
for i in range(len(date)):
    try: #for handle error of divide by zero
        volume_Turnover.append(volume[i]/Turnover[i])
    except:
        volume_Turnover.append(0.0) 
    try:
        close_last.append(close[i]/last[i])
    except:
        close_last.append(0.0)

#-----------------------create and set value for features and labels---------------------------
label = []
features = []

for i in range(len(close)):
    if close[i] - lastClose[i] > 0:
        label.append('Positive')
    else: 
        label.append('Negative')  
label.pop(-1)


allFeatures, combs = comb([high, low, last, close, volume, Turnover, volume_Turnover, close_last])
y = np.array(label)

createModel(allFeatures, y, combs)

