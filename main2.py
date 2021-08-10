import math
from operator import le
from os import sep
from numpy import testing
from numpy.core.numeric import normalize_axis_tuple
from numpy.lib.function_base import average
import pandas
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from itertools import combinations
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


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

	# for i in range(len(predTmp)):
	# 	if predTmp[i] == featureMax[i]:
	# 		plt.plot(featureMax[i], 'bo', markersize=1)
	# 	else:
	# 		plt.plot(predTmp[i], 'ro', markersize=1)

	return predTmp, featureMax


def model(allFeatures, y, combs):
	j = 0
	scores = []
	predicts = []
	
	for feature in allFeatures:
		X_train, X_test, y_train, y_test = train_test_split(feature, y, test_size=0.09)
		print(combs[j], ':')

		dtree = DecisionTreeRegressor()
		dtree = dtree.fit(X_train, y_train)

		pred = dtree.predict(X_test)
		predicts.append(pred)
		#dtree.score(y_test, pred)
		scores.append(metrics.r2_score(y_test, pred))
		# scores[j] = metrics.accuracy_score(y_test, pred)

		print(scores[j])
		print('========================')
		j += 1

	print('num of all combinations:',j)

	print('max:')
	print(combs[scores.index(max(scores))], ':', max(scores))	
	
	return y_test, predicts[scores.index(max(scores))]


def RFModel(features, y):
	X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.09, shuffle=False)

	randF = RandomForestRegressor(n_estimators=4,max_depth=3)
	randF.fit(X_train, y_train)

	pred = randF.predict(X_test)

	print(metrics.r2_score(y_test, pred))

	# plt.plot(close[-299:], 'b')
	# plt.plot(pred, 'r')
	# plt.show()

	# plt.plot(close[-299:], 'bo')
	# plt.plot(pred, 'rx')
	# plt.show()
	
	
	plt.plot(y_test, 'b')
	plt.plot(pred, 'r')
	plt.show()

	plt.plot(y_test, 'bo')
	plt.plot(pred, 'rx')
	plt.show()

def findMSL(y_test, pred):
	return math.sqrt(sum([((pred[i] - y_test[i])**2) for i in range(len(pred))]) / len(pred))

def UseSVR(features, y):
	X_train, X_test, y_train, y_test = train_test_split(features, y, test_size=0.09, shuffle=False)

	reg = SVR()
	reg.fit(X_train, y_train)

	pred = reg.predict(X_test)

	print(metrics.r2_score(y_test, pred))

	# plt.plot(y_test, 'b')
	# plt.plot(pred, 'r')
	# plt.show()
	return y_test
	# plt.plot(y_test, 'bo')
	# plt.plot(pred, 'rx')
	# plt.show()




data = pandas.read_csv('ADANIPORTS.csv')
close = data['Close']

avg7, avg14, avg21, deviation = [], [], [], []


for i in range(len(close)):
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
			deviation.append(variance**0.5)

#-------------------------------for decision tree---------------------------------------
# close, avg7, avg14, avg21, deviation = normalize(close), normalize(avg7), normalize(avg14), normalize(avg21), normalize(deviation)

#-------------------------------for RandomForest and SVM---------------------------------------
allFeatures = []
for i in range(len(close)):
	tmp = []
	tmp.append(close[i])
	tmp.append(avg7[i])
	tmp.append(avg14[i])
	tmp.append(avg21[i])
	tmp.append(deviation[i])
	allFeatures.append(tmp)
allFeatures.pop(-1)	


#-------------------------------for decision tree---------------------------------------
# allFeatures, combination = comb([close, avg7, avg14, avg21, deviation])


allFeatures = np.array(allFeatures)
print(allFeatures)
y = data['Close']
y.pop(0)
#y = normalize(y)
y = np.array(y)

#-------------------------------for RandomForest and SVM---------------------------------------
#RFModel(allFeatures, y)

y_test = UseSVR(allFeatures, y)

print(metrics.r2_score(y_test, avg7[-299:]))
plt.plot(y_test, 'b')
plt.plot(avg7[-299:], 'r')
plt.show()


#-------------------------------for decision tree---------------------------------------
# y_test, pred = model(allFeatures, y, combination)

# print(findMSL(y_test, pred))
# print(average(close[-299:]))

# plt.plot(close[-299:], 'bo')
# plt.plot(pred, 'rx')
# plt.show()
