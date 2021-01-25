import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tabulate import tabulate


def scatterPlot2d(features,labels):
	plt.subplots_adjust(top=0.98,
						bottom=0.06,
						left=0.045,
						right=0.97,
						hspace=0.39,
						wspace=0.175)
	n = features.shape[1] 
	# plt.subplot(n,n-1,)
	indx = 1
	for i in range(n):
		for j in range(i,n):
			if not(i == j):
				if n%2 == 0:
					plt.subplot(n//2,n-1,indx)
				else :
					plt.subplot(n,(n-1)//2,indx)
				plt.scatter(features[:,i],features[:,j],c=labels,cmap=plt.cm.Set1,edgecolor='k')
				plt.xlabel('f'+str(i+1)); plt.ylabel('f'+str(j+1))
				indx += 1
	plt.show()

def tsnePlot2d(features,labels):
	x_emb = TSNE(n_components=2).fit_transform(features)
	plt.scatter(x_emb[:,0],x_emb[:,1],c=labels,cmap=plt.cm.Set1,edgecolor='k')
	plt.show()

def pcaPlot2d(features,labels):
	f = features.copy()
	f = (f - np.mean(f,axis=0))/np.array([np.std(f[:,i]) for i in range(features.shape[1])])
	pca = PCA(n_components = 2)
	x_pca = pca.fit_transform(f)
	plt.scatter(x_pca[:,0],x_pca[:,1],c=labels,cmap=plt.cm.Set1,edgecolor='k')
	plt.show()


def histogram(features,labels):
	for i in range(4):
		plt.subplot(2,2,i+1)
		plt.hist(features[:,i],bins=20,color='green')
		plt.xlabel('f'+str(i+1))
	plt.show()


def qFiveData(features,labels):
	head = ['feature','Max Value','Average','Standard deviation']
	dat = []
	av = np.mean(features,axis=0)
	std = [np.std(features[:,i]) for i in range(features.shape[1])]
	mx = np.max(features,axis=0)
	f = np.array(['f1','f2','f3','f4'])
	dat = np.array([f,mx,av,std]).T
	return head,dat




#Q1
p = pd.read_csv('iris.data',names=['f1','f2','f3','f4','label'])
print(p.head(5))
print("Samples: "+ str(p.shape[0]))
print("Number of attributes: "+str(len(p.axes[1])))

#Q2
nData = p.values
features = nData[:,0:4]
labels = nData[:,4]
uLabels = np.unique(labels) 
labelToNumbers = {d:k for (d,k) in zip(uLabels,range(len(uLabels)))}
for d in uLabels:
	labels[labels == d] = labelToNumbers[d]

#Q3
# scatterPlot2d(features,labels)
# tsnePlot2d(features,labels)
# pcaPlot2d(features,labels)

#Q4
histogram(features,labels)

#Q5
h,d = qFiveData(features,labels)
table = tabulate(d,headers=h)
print(table)

