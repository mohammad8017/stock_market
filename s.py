from itertools import combinations
def comb(lst:list):
    allFeatures = []
    for i in range(len(lst)):
        l = list(combinations(lst, i+1))
        
        for j in range(len(l)):
            tmp = []
            for k in range(i+1):
                indexList = l[j][k]
                if k == 0:
                    for m in range(len(indexList)):
                        tmp.append([indexList[m]])
                else:
                    for m in range(len(indexList)):
                        tmp[m].append(indexList[m])
            allFeatures.append(tmp) 


a = [1,2,3]
b = [4,5,6]
c = [7,8,9]

comb([a,b,c])