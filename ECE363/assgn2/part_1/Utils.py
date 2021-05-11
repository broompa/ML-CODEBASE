import numpy as np
import joblib
import scipy.special
import sys
import matplotlib.pyplot as  plt
from mlxtend.data import loadlocal_mnist
from Templates import MLPClassifier
import pickle
import bz2
from sklearn.manifold import TSNE

def load_dataset(dPath):
    x_train, y_train = loadlocal_mnist(images_path = dPath + 'train-images-idx3-ubyte', labels_path = dPath +  'train-labels-idx1-ubyte')
    x_test, y_test = loadlocal_mnist(images_path = dPath + 't10k-images-idx3-ubyte', labels_path = dPath + 't10k-labels-idx1-ubyte')   
    return x_train , y_train , x_test , y_test

def save_model(model,fName):
    f = open(fName, 'wb')
    pickle.dump(model,f)
    f.close()
    print('Model saved at '+fName)

def load_model(fName):
    f = open(fName, 'rb')
    model = pickle.load(f)
    f.close()
    return model


def tsnePlot2d(features,labels,title = ''):
    plt.rcParams['figure.figsize'] = [5,3]
    plt.rcParams['figure.dpi'] = 100
    
    x_emb = TSNE(n_components=2).fit_transform(features)
    y_unique = np.unique(labels)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for i in y_unique:
        indexes = np.where(labels == i)
        plt.scatter(x_emb[indexes,0], x_emb[indexes,1], c=colors[i], edgecolor='k', label = i)
    plt.legend()
    plt.title('TSNE '+title)
    plt.savefig("TSNE_" + title+".png", dpi=300, bbox_inches='tight')
    plt.show()
