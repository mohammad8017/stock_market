from os import sep
from numpy.core.numeric import normalize_axis_tuple
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from itertools import combinations



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



data = pandas.read_csv('ADANIPORTS.csv')
close = data['Close']

avg7, avg14, avg21, deviation = [], [], [], []


for i in range(len(close)+1):
	if i == 0:
		avg7.append(close[i])
		avg14.append(close[i])
		avg21.append(close[i])
		deviation.append(0.0)
	else:	
		if i < 7:
			avg7.append(sum(close[:i]) / i)
		else:
			avg7.append(sum(close[i-7:i]) / 7)

		if i < 14:
			avg14.append(sum(close[:i]) / i)
		else:
			avg14.append(sum(close[i-14:i]) / 14) 

		if i < 21:
			avg21.append(sum(close[:i]) / i)
		else:
			avg21.append(sum(close[i-21:i]) / 21)  

		if i < 7: 
			variance = sum([((x - avg7[i]) ** 2) for x in close[:i]]) / i
			deviation.append(variance**0.5)
		else:
			variance = sum([((x - avg7[i]) ** 2) for x in close[i-7:i]]) / 7


close, avg7, avg14, avg21, deviation = normalize(close), normalize(avg7), normalize(avg14), normalize(avg21), normalize(deviation)
