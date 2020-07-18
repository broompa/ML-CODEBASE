import tensorflow as tf 
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import sys



ProteinDict={'A':0,'R':1,'N':2,'D':3,'C':4,'Q':5,'E':6,'G':7,'H':8,'I':9,'L':10,'K':11,'M':12,'F':13,'P':14,'S':15,'T':16,'W':17,'Y':18,'V':19,'U':20,'O':21,'X':22}

def getVector(ls, wSize): # list of Protein Sequences, wSize - windows Size(must be odd)
	assert wSize%2==1
	k=(wSize-1)//2
	vectors=[]
	label=[]
	c=0
	for i in range(len(ls)):
		for j in range(len(ls[i])):
			if (j<k):
				s="X"*(k-j)+ls[i][0:j+k+1]
			elif (len(ls[i])-j<wSize):
				s=ls[i][j:len(ls[i])]+"X"*(wSize+j-len(ls[i]))
			else :
				s = ls[i][j:j+wSize]
			vectors.append(list(map(lambda x : ProteinDict[x.upper()],list(s))))
			label.append(str('+1') if s[k].islower() else str('-1'))
	return vectors,label 



if __name__=='__main__':
	train = "train.data"
	test = 'test1.txt'
	with open(train,'r') as file:
		d = list(map(lambda x : x.split(',')[1].replace('\n',''),file.read().split('>')[1:]))
	trainDataX,trainDataY = getVector(d,17)
	print(trainDataX[:6],sep='\n')


#	clf = RandomForestClassifier(n_estimators=100,criterion="entropy")

	#---------------Uncomment while testing, Comment while generating result for kaggle submission
	# X_train,X_test,Y_train,Y_test=train_test_split(trainDataX,trainDataY,test_size=0.60)
	# clf.fit(X_train,Y_train)
	# print(accuracy_score(Y_test,clf.predict(X_test)))
	#-------------- Testing Section Ends here !-----------------------
	
	#----------------------------Uncomment while generating result for kaggle competition, comment while testing.
	# clf.fit(trainDataX,trainDataY)
	# with open(test,'r') as file:
	# 	testRawData=file.read().split('\n')[1:-1]
	# 	indexes=list(map(lambda x: x.split(',')[0] ,testRawData))
	# 	testRawData=list(map(lambda x: x.split(',')[1] ,testRawData))
	# 	testRawData=[''.join(testRawData)]
	# test_X=getVector(testRawData,17)
	# prediction = clf.predict(test_X[0]).tolist()
	# s='\n'.join([indexes[i]+","+str(prediction[i]) for i in range(len(indexes))])
	# s="ID,Lable\n"+"\n".join([indexes[i]+","+str(prediction[i]) for i in range(len(indexes))])
	# with open("submission.txt",'w') as file:
	# 	file.write(s)
	#------------------------
