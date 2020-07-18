"""	
Author: Lavanya Verma
Date: 22 March 2020
ML Assignment 1 of BIO213 (Winter 2020),IIITD
Code uses supervised classification algorithm (random forest) to classify the antifungal nature of sequence of amino acids (proteins)
Competition had been conducted on kaggle.
Accuracy: 91.502%
"""
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
import sys
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame
import matplotlib.pyplot as plt


train="train.csv"
test ="test.csv"
output = "Data.csv"

ProteinDict={'A':0,'R':1,'N':2,'D':3,'C':4,'Q':5,'E':6,'G':7,'H':8,'I':9,'L':10,'K':11,'M':12,'F':13,'P':14,'S':15,'T':16,'W':17,'Y':18,'V':19,'U':20,'O':21} # Amino acid one letter mapping to index of Vector input 
ProteinList=['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', 'U', 'O']
p=['AA', 'AR', 'AN', 'AD', 'AC', 'AQ', 'AE', 'AG', 'AH', 'AI', 'AL', 'AK', 'AM', 'AF', 'AP', 'AS', 'AT', 'AW', 'AY', 'AV', 'RA', 'RR', 'RN', 'RD', 'RC', 'RQ', 'RE', 'RG', 'RH', 'RI', 'RL', 'RK', 'RM', 'RF', 'RP', 'RS', 'RT', 'RW', 'RY', 'RV', 'NA', 'NR', 'NN', 'ND', 'NC', 'NQ', 'NE', 'NG', 'NH', 'NI', 'NL', 'NK', 'NM', 'NF', 'NP', 'NS', 'NT', 'NW', 'NY', 'NV', 'DA', 'DR', 'DN', 'DD', 'DC', 'DQ', 'DE', 'DG', 'DH', 'DI', 'DL', 'DK', 'DM', 'DF', 'DP', 'DS', 'DT', 'DW', 'DY', 'DV', 'CA', 'CR', 'CN', 'CD', 'CC', 'CQ', 'CE', 'CG', 'CH', 'CI', 'CL', 'CK', 'CM', 'CF', 'CP', 'CS', 'CT', 'CW', 'CY', 'CV', 'QA', 'QR', 'QN', 'QD', 'QC', 'QQ', 'QE', 'QG', 'QH', 'QI', 'QL', 'QK', 'QM', 'QF', 'QP', 'QS', 'QT', 'QW', 'QY', 'QV', 'EA', 'ER', 'EN', 'ED', 'EC', 'EQ', 'EE', 'EG', 'EH', 'EI', 'EL', 'EK', 'EM', 'EF', 'EP', 'ES', 'ET', 'EW', 'EY', 'EV', 'GA', 'GR', 'GN', 'GD', 'GC', 'GQ', 'GE', 'GG', 'GH', 'GI', 'GL', 'GK', 'GM', 'GF', 'GP', 'GS', 'GT', 'GW', 'GY', 'GV', 'HA', 'HR', 'HN', 'HD', 'HC', 'HQ', 'HE', 'HG', 'HH', 'HI', 'HL', 'HK', 'HM', 'HF', 'HP', 'HS', 'HT', 'HW', 'HY', 'HV', 'IA', 'IR', 'IN', 'ID', 'IC', 'IQ', 'IE', 'IG', 'IH', 'II', 'IL', 'IK', 'IM', 'IF', 'IP', 'IS', 'IT', 'IW', 'IY', 'IV', 'LA', 'LR', 'LN', 'LD', 'LC', 'LQ', 'LE', 'LG', 'LH', 'LI', 'LL', 'LK', 'LM', 'LF', 'LP', 'LS', 'LT', 'LW', 'LY', 'LV', 'KA', 'KR', 'KN', 'KD', 'KC', 'KQ', 'KE', 'KG', 'KH', 'KI', 'KL', 'KK', 'KM', 'KF', 'KP', 'KS', 'KT', 'KW', 'KY', 'KV', 'MA', 'MR', 'MN', 'MD', 'MC', 'MQ', 'ME', 'MG', 'MH', 'MI', 'ML', 'MK', 'MM', 'MF', 'MP', 'MS', 'MT', 'MW', 'MY', 'MV', 'FA', 'FR', 'FN', 'FD', 'FC', 'FQ', 'FE', 'FG', 'FH', 'FI', 'FL', 'FK', 'FM', 'FF', 'FP', 'FS', 'FT', 'FW', 'FY', 'FV', 'PA', 'PR', 'PN', 'PD', 'PC', 'PQ', 'PE', 'PG', 'PH', 'PI', 'PL', 'PK', 'PM', 'PF', 'PP', 'PS', 'PT', 'PW', 'PY', 'PV', 'SA', 'SR', 'SN', 'SD', 'SC', 'SQ', 'SE', 'SG', 'SH', 'SI', 'SL', 'SK', 'SM', 'SF', 'SP', 'SS', 'ST', 'SW', 'SY', 'SV', 'TA', 'TR', 'TN', 'TD', 'TC', 'TQ', 'TE', 'TG', 'TH', 'TI', 'TL', 'TK', 'TM', 'TF', 'TP', 'TS', 'TT', 'TW', 'TY', 'TV', 'WA', 'WR', 'WN', 'WD', 'WC', 'WQ', 'WE', 'WG', 'WH', 'WI', 'WL', 'WK', 'WM', 'WF', 'WP', 'WS', 'WT', 'WW', 'WY', 'WV', 'YA', 'YR', 'YN', 'YD', 'YC', 'YQ', 'YE', 'YG', 'YH', 'YI', 'YL', 'YK', 'YM', 'YF', 'YP', 'YS', 'YT', 'YW', 'YY', 'YV', 'VA', 'VR', 'VN', 'VD', 'VC', 'VQ', 'VE', 'VG', 'VH', 'VI', 'VL', 'VK', 'VM', 'VF', 'VP', 'VS', 'VT', 'VW', 'VY', 'VV']

# def ProteinToVector(s,l=15): # Binary Profile based
# 	l1 = [0]*22
# 	l2 = [0]*22
# 	for i in range(0,int(min(len(s),l))):
# 		l1[ProteinDict[s[i]]] +=1
# 		l2[ProteinDict[s[len(s)-1-i]]]+=1
# 	l1.extend(l2)
# 	return l1

def ProteinToVector1(s,l=20): # Binary Profile based
	l1 = [0]*22
	l2 = [0]*22
	for i in range(0,int(min(len(s),l))):
		l1[ProteinDict[s[i]]] +=1
		l2[ProteinDict[s[len(s)-1-i]]]+=1
	
	l1.extend(l2)
	l1=(np.array(l1)*100)/len(s)
	return l1

def ProteinToVector2(s):
	ls=[0]*400
	for k in range(400):
		i=k//20
		j = k%20
		# ls[k]+=s.count(ProteinList[i]+ProteinList[j])
		ls[k]+=s.count(p[k])
	# ls = (np.array(ls)*100)/((len(s)//2)*2)
	arr = ProteinToVector1(s)
	ls = np.concatenate((ls,arr),axis=0)
	return ls 

def ProteinToVector(s):
	l=[]
	for i in range(20):
		for j in range(20):
			for k in range(20):
				l.append(s.count(ProteinList[i]+ProteinList[j]+ProteinList[k]))
	l =np.array(l)*100/((len(s)//3)*3)
	# arr = ProteinToVector2(s)
	# l = np.concatenate((l,arr),axis=0)
	return l 
if __name__=='__main__':
	with open(train,"r") as file:
		trainRawData = file.read().splitlines()[1:]
	trainData=list(map(lambda x : x.split(',')[-1],trainRawData )) 
	trainDataY=(list(map(lambda x : x.split(',')[1],trainRawData )))#labels
	trainDataX=(list(map( ProteinToVector,trainData))) #features 
	
	trainDataY=np.ravel(trainDataY)
	clf = RandomForestClassifier(n_estimators=400,criterion="entropy")


	#---------------Uncomment while testing, Comment while generating result for kaggle submission
	# X_train,X_test,Y_train,Y_test=train_test_split(trainDataX,trainDataY,test_size=0.30)
	# clf.fit(X_train,Y_train)
	# print(accuracy_score(Y_test,clf.predict(X_test)))
	#-------------- Testing Section Ends here !-----------------------
	
	
	
	#----------------------------Uncomment while generating result for kaggle competition, comment while testing.
	clf.fit(trainDataX,trainDataY)
	with open(test,'r') as file:
		testRawData=file.read().splitlines()
	test_X = list(map(lambda x : ProteinToVector(x.split(',')[1]),testRawData[1:] ))
	Ids= list(map(lambda x: x.split(',')[0],testRawData[1:]))
	data = clf.predict(test_X)
	res = [i +','+ j for i,j in zip(Ids,data) ]
	res= 'ID,label\n'+'\n'.join(res)
	with open(output,'w') as file:
		file.write(res)
	#------------------------


