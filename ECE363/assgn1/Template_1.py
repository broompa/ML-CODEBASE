import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
class DatasetLoader():
    def __init__(self,arg):
        """ arg -> 0 (abalone)
        """
        self.arg = arg
    def loadDataset(self):
        if (self.arg == 0):
            dataset = pd.read_csv('./dataset/abalone/rings/Prototask.data',delimiter = ' ')
            dataset.columns = ['sex','length','diameter','height','whole_weight','shucked_weight','viscera_weight','shell_weight','rings']
            df = pd.get_dummies(dataset)
            dfn = (df - df.min())/(df.max() - df.min())
            x = np.array(dfn.loc[:,dfn.columns != 'rings'])
            y = np.array(dfn.loc[:,dfn.columns == 'rings'])
            return x,y
        elif (self.arg == 1):
            df = pd.read_csv('./dataset/data.csv',delimiter=',')
            df = (df - df.min())/(df.max() - df.min())
            x = np.array(df.loc[:,df.columns != 'Body_Weight'])
            y = np.array(df.loc[:,df.columns == 'Body_Weight'])
            return x,y
        elif(self.arg == 2):
            ds1 = pd.read_csv('./dataset/2_1/train.csv')
            ds1.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','Income']
            x_train = pd.get_dummies(ds1.loc[:,ds1.columns != 'Income'])
            y_train = pd.get_dummies(ds1.loc[:,ds1.columns == 'Income'])

            x_train = (x_train - x_train.min() )/ (x_train.max() - x_train.min())
            y_train = (y_train - y_train.min() )/ (y_train.max() - y_train.min())

            x_train = np.array(x_train)
            y_train = np.array(y_train)[:,1].reshape((y_train.shape[0],1))


            ds2 = pd.read_csv('./dataset/2_1/test.csv')
            ds2.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','Income']
            x_test = pd.get_dummies(ds2.loc[:,ds2.columns != 'Income'])
            y_test = pd.get_dummies(ds2.loc[:,ds2.columns == 'Income'])
            x_test = (x_test - x_test.min() )/ (x_test.max() - x_test.min())
            y_test = (y_test - y_test.min() )/ (y_test.max() - y_test.min())
            x_test = np.array(x_test)
            y_test = np.array(y_test)[:,1].reshape((y_test.shape[0],1))
            

            return x_train,y_train,x_test,y_test
            
class LinearRegressorGradient():
    def __init__(self,lam,lr=0.0001,reg = None,epochs = 1000):
        self.params = dict()
        self.params['fitCalled'] = False
        if reg not in [None, 'l1','l2']:
            raise Exception("Provide appropriate value of reg")
        self.params['reg'] = reg
        self.params['lambda'] = lam
        self.params['lr'] = lr 
        self.params['epochs'] = epochs
    def fit(self,x_train,y_train,x_val=None,y_val=None,valSet = False):

        x_train = np.append(x_train,np.ones((x_train.shape[0],1)),axis = 1)
        trainRMSE = np.zeros(self.params['epochs'])
        trainLoss = np.zeros(self.params['epochs'])
        if valSet:
            x_val = np.append(x_val,np.ones((x_val.shape[0],1)),axis = 1)
            valRMSE  = np.zeros(self.params['epochs'])
        weights = np.random.random((x_train.shape[1],1))
        
        for i in range(self.params['epochs']):
            y_train_predicted = np.matmul(x_train,weights)
            train_loss = self.loss(y_train,y_train_predicted,weights)
            trainLoss[i] = train_loss 
            trainRMSE[i] = np.sqrt(np.square(y_train - y_train_predicted).mean())
            if valSet:
                y_val_predicted = np.matmul(x_val,weights)
                valRMSE[i] = np.sqrt(np.square(y_val - y_val_predicted).mean())
            weights = weights - self.params['lr']*self.gradient(y_train,y_train_predicted,x_train,weights)
        
        self.params['weights'] = weights
        self.params['trainRMSE'] = trainRMSE
        if (valSet):
            self.params['valRMSE'] = valRMSE
        self.params['affine'] = True
        self.params['inputDim'] = x_train.shape[1]
        self.params['fitCalled'] = True
        self.params['trainLoss'] = trainLoss
    
    def predict(self,x_test):
        if (self.params['fitCalled']):
            x_test = np.append(x_test,np.ones((x_test.shape[0],1)),axis = 1)
            return np.matmul(x_test,self.params['weights'])
        raise Exception("Call fit first")
    
    def getParams(self):
        return self.params

    def loss(self,y_true,y_predicted,weights):
        loss = np.square(y_true - y_predicted).mean()
        if self.params['reg'] ==  'l1':
            loss = loss + self.params['lambda']*np.linalg.norm(weights,ord = 1)
        elif self.params['reg'] == 'l2':
            loss = loss + self.params['lambda']*np.linalg.norm(weights,ord = 2)
        return loss
    def gradient(self,y_true,y_predicted,x,weights):
        grad = -2*np.matmul(x.T,(y_true - y_predicted))
        if self.params['reg'] ==  'l1':
            grad =  grad + 2*self.params['lambda']*np.array(list(map(lambda x: 1 if x>=0 else -1,weights))).reshape(weights.shape)
        elif self.params['reg'] == 'l2':
            grad = grad + 2*self.params['lambda']*weights
        return grad
    
    def rmseLoss(self,y_val,x_val):
        return np.sqrt(np.square(y_val - self.predict(x_val)).mean())

class LinearRegressorNormal():
    def __init__(self):
        self.params = dict()
        self.params['fitCalled'] = False
    def fit(self,x_train,y_train,x_val = None,y_val = None,valSet = False):
        x_train = np.append(x_train,np.ones((x_train.shape[0],1)),axis = 1)
        xtx = np.matmul(x_train.T,x_train)
        if (np.linalg.det(xtx) == 0):
            raise Exception("Singular Matrix")
        
        weights = np.matmul(np.linalg.inv(xtx), np.matmul(x_train.T,y_train))
        
        y_train_predicted = np.matmul(x_train,weights)
        trainRMSE = np.sqrt(np.square(y_train - y_train_predicted).mean())
        
        if (valSet):
            x_val = np.append(x_val,np.ones((x_val.shape[0],1)),axis = 1)
            y_val_predicted = np.matmul(x_val,weights)
            valRMSE = np.sqrt(np.square(y_val - y_val_predicted).mean())
            self.params['valRMSE'] = valRMSE

        self.params['weights'] = weights
        self.params['trainRMSE'] = trainRMSE
        self.params['fitCalled'] = True
        self.params['inputDim'] = x_train.shape[1]
        self.params['affine'] = True
    def predict(self,x_test):
        if (self.params['fitCalled']):
            x_test = np.append(x_test,np.ones((x_test.shape[0],1)),axis = 1)
            return np.matmul(x_test,self.params['weights'])
        raise Exception("Call fit first")
    def getParams(self):
        return self.params

class CrossValidate():
    def __init__(self,x,y):
        self.params = dict()
        self.params['x'] = x
        self.params['y'] = y

    @classmethod
    def k_split(cls,x_data,y_data,k,folds):
        assert k < folds
        d = x_data.shape[0] // folds
        valSetIndicies = np.arange(k*d , k*d + d -1 )
        mask = np.ones(x_data.shape[0],bool)
        mask[valSetIndicies] = False
        x_val = x_data[~mask,:]
        x_train = x_data[mask,:]
        y_val = y_data[~mask,:]
        y_train = y_data[mask,:]
        return x_train,y_train,x_val,y_val

        
    def k_fold(self,folds):
        Params_G = []
        Params_N = []
        trainRMSE_G = 0
        valRMSE_G = 0
        trainRMSE_N = []
        valRMSE_N = []
        for k in range(folds):
            x_train, y_train, x_val , y_val = CrossValidate.k_split(self.params['x'],self.params['y'],k,folds)

            model = LinearRegressorGradient(lam = None)
            model.fit(x_train = x_train , y_train = y_train,x_val= x_val,y_val= y_val,valSet=True)
            p = model.getParams()
            trainRMSE_G = trainRMSE_G + p['trainRMSE'] # Mean
            valRMSE_G  = valRMSE_G + p['valRMSE'] # Mean
            Params_G.append(p)
            epochs = p['epochs']

            model = LinearRegressorNormal()
            try :
                model.fit(x_train = x_train , y_train = y_train,x_val= x_val,y_val= y_val,valSet=True)
                p = model.getParams()
                Params_N.append(p)
                trainRMSE_N.append(p['trainRMSE'])
                valRMSE_N.append(p['valRMSE'])
            except Exception:
                print("Singular Matrix")
                Params_N.append("Invalid Model")
                trainRMSE_N.append(np.nan)
                valRMSE_N.append(np.nan)        
        trainRMSE_G = trainRMSE_G/ folds # Mean
        valRMSE_G = valRMSE_G/ folds # Mean

        self.params['folds'] = folds
        self.params['trainRMSE_G'] = trainRMSE_G
        self.params['valRMSE_G']  = valRMSE_G
        self.params['trainRMSE_N'] = trainRMSE_N
        self.params['valRMSE_N'] = valRMSE_N
        self.params['epochs'] = epochs
        self.params['Models_G'] = Params_G
    
    def leastValRMSE(self):
        val = np.inf; ind = None
        for i in range(self.params['folds']):
            if(val > min(self.params['Models_G'][i]['valRMSE'])):
                val = min(self.params['Models_G'][i]['valRMSE'])
                ind = i
        return ind # index of fold which have lowest valset rmse
        
    def newVal2Test(self,fN):
        """ fN: Fold number which have lowest val rmse """
        x_train_val,y_train_val, x_test,y_test = CrossValidate.k_split(self.params['x'],self.params['y'],fN,self.params['folds'])

        self.params['x_new'] = x_train_val
        self.params['y_new'] = y_train_val
        self.params['x_ntest'] = x_test
        self.params['y_ntest'] = y_test

        # return x_train_val,y_train_val,x_test,y_test
    
    def GridSearch(self, model, hyperParams, folds=5):
        t1  = time.time()
        def helperDictParams(x,k):
            s = []
            for i in range(len(x)):
                s.append(str(k[i]) + '=' + str(x[i]))
            s = ','.join(s)
            return "dict({})".format(s)

        keys = list(hyperParams.keys())
        vProduct = list(itertools.product(*list(hyperParams.values())))
        scores = []
        d = self.params['x_new'].shape[0]
        for i in range(len(vProduct)):
            rmse = 0
            for k in range(folds):
                x_train, y_train, x_val , y_val = CrossValidate.k_split(self.params['x_new'],self.params['y_new'],k,folds)
                d = helperDictParams(vProduct[i],keys)
                m = model(**eval(d))
                m.fit(x_train = x_train, y_train  = y_train)
                rmse = m.rmseLoss(y_val,x_val)
            rmse = rmse/folds
            scores.append(rmse)
        ind = scores.index(min(scores))
        self.params['bestConfig'] = helperDictParams(vProduct[ind],keys)
        print("time taken: %f"%(time.time() - t1))
   
    def getBestConfig(self):
        return self.params['bestConfig']
        

class LogisticRegressor():
    def __init__(self,lam,lr=0.0001,reg = None,epochs = 1000):
        self.params = dict()
        self.params['fitCalled'] = False
        if reg not in [None, 'l1','l2']:
            raise Exception("Provide appropriate value of reg")
        self.params['reg'] = reg
        self.params['lambda'] = lam
        self.params['lr'] = lr 
        self.params['epochs'] = epochs
    def fit(self,x_train,y_train,x_val=None,y_val=None,valSet = False):
        x_train = np.append(x_train,np.ones((x_train.shape[0],1)),axis = 1)
        trainAccuracy = np.zeros(self.params['epochs'])
        trainLoss = np.zeros(self.params['epochs'])
        if valSet:
            x_val = np.append(x_val,np.ones((x_val.shape[0],1)),axis = 1)
            valAccuracy  = np.zeros(self.params['epochs'])
        weights = np.random.random((x_train.shape[1],1))
        
        for i in range(self.params['epochs']):
            print(i)
            y_train_predicted = self.sigmoid(np.matmul(x_train,weights))
            trainLoss[i] = self.loss(y_train,y_train_predicted,weights)
            trainAccuracy[i] = self.accuracy(y_train,y_train_predicted)
            if valSet:
                y_val_predicted = self.sigmoid(np.matmul(x_val,weights))
                valAccuracy[i] = self.accuracy(y_val,y_val_predicted)
            weights = weights - self.params['lr']*self.gradient(y_train,y_train_predicted,x_train,weights)
        
        self.params['weights'] = weights
        self.params['trainAccuracy'] = trainAccuracy
        if (valSet):
            self.params['valAccuracy'] = valAccuracy
        self.params['affine'] = True
        self.params['inputDim'] = x_train.shape[1]
        self.params['fitCalled'] = True
        self.params['trainLoss'] = trainLoss
    
    def accuracy(self,y_true,y_predicted):
        # y_predicted = np.array(list(map(lambda x : 1 if x > 0.5 else 0 , y_predicted))).reshape(y_predicted.shape)
        y_predicted = y_predicted > 0.5
        return np.sum(np.equal(y_true , y_predicted))/y_true.shape[0]

    def sigmoid(self,z):
        return 1/(1 + np.exp(-z))

    def predict(self,x_test):
        if (self.params['fitCalled']):
            x_test = np.append(x_test,np.ones((x_test.shape[0],1)),axis = 1)
            return self.sigmoid(np.matmul(x_test,self.params['weights']))
        raise Exception("Call fit first")
    
    def getParams(self):
        return self.params

    def loss(self,y_true,y_predicted,weights):
        noise = 1e-9
        loss = (-y_true*np.log(y_predicted + noise) - (1 - y_true)*np.log(1 - y_predicted + noise)).mean()
        if self.params['reg'] ==  'l1':
            loss = loss + self.params['lambda']*np.linalg.norm(weights,ord = 1)
        elif self.params['reg'] == 'l2':
            loss = loss + self.params['lambda']*np.linalg.norm(weights,ord = 2)
        return loss

    def gradient(self,y_true,y_predicted,x,weights):
        grad = -1*np.matmul(x.T,(y_true - y_predicted))
        if self.params['reg'] ==  'l1':
            grad =  grad + 2*self.params['lambda']*np.array(list(map(lambda x: 1 if x>=0 else -1,weights))).reshape(weights.shape)
        elif self.params['reg'] == 'l2':
            grad = grad + 2*self.params['lambda']*weights
        return grad



    
        

        
    


    


