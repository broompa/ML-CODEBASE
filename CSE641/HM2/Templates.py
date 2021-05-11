import numpy as np
class MLPClassifier:

    acti_fns = ['relu', 'sigmoid', 'tanh']
    weight_inits = ['zero', 'random', 'normal']
    
    def __init__(self,n_layers ,layer_sizes, activation='relu',  learning_rate = 1e-5, weight_init="random", batch_size=64, num_epochs = 100):
        
        if (activation not in self.acti_fns):
            raise Exception("Incorrect Weight Activation Function")
        if (weight_init not in self.weight_inits):
            raise Exception("Incorrect Weight Initialization")
        
        self.n_layers = n_layers
        
        self.weights = []
        self.weights_grad = []
        self.inputs = []
        if weight_init == 'zero':
            self.zero_weights(layer_sizes)
        elif weight_init == 'random':
            self.random_weights(layer_sizes)
        elif weight_init == 'normal':
            self.normal_weights(layer_sizes)

        if activation == 'relu':
            self.acti_fns = self.relu
            self.acti_fns_grad = self.relu_grad
        elif activation == 'sigmoid':
            self.acti_fns = self.sigmoid
            self.acti_fns_grad = self.sigmoid_grad
        elif activation == 'tanh':
            self.acti_fns = self.tanh
            self.acti_fns_grad = self.tanh_grad
    
        self.optimizer = self.gradient_descent
        self.lr = learning_rate

        self.batch_size = batch_size
        self.epochs = num_epochs
    
    def zero_weights(self,layer_sizes):
        for i in range(len(layer_sizes)-1):
            w = np.zeros((layer_sizes[i]+1,layer_sizes[i+1]),dtype = np.float128)
            self.weights.append(w)
            self.weights_grad.append(np.zeros(w.shape,dtype=np.float128))
    
    def random_weights(self,layer_sizes):
        for i in range(len(layer_sizes)-1):
            w = np.random.random((layer_sizes[i]+1,layer_sizes[i+1]))*0.01
            self.weights.append(w)
            self.weights_grad.append(np.zeros(w.shape,dtype=np.float128))
    
    def normal_weights(self,layer_sizes):
        for i in range(len(layer_sizes)-1):
            w = np.random.normal(scale = 0.01 , size = (layer_sizes[i]+1,layer_sizes[i+1]))
            self.weights.append(w)
            self.weights_grad.append(np.zeros(w.shape,dtype=np.float128))
    

    def fit(self,X,Y,verbose_idx= None):
        
        # make it loop over batches <---
        training_loss = []
        iterations = []
        accuracy = []
        
        if (verbose_idx == None):
            verbose_idx = X.shape[0]//self.batch_size//20
        

        for e in range(self.epochs):
            sample_indicies = np.random.permutation(X.shape[0])
            for idx in range(X.shape[0]//self.batch_size):
                rows = sample_indicies[idx*self.batch_size: (idx+1)*(self.batch_size) ]
                # print(len(rows))
                # if idx == 6:
                #     return
                # continue
                
                x_batch = X[rows,:]
                y_batch = Y[rows]
                one_hot = np.eye(9 + 1)[y_batch] # D(s)
                
                probs = self.forward_pass(x_batch) # Y(s)           
                predicted_labels = np.argmax(probs,axis = 1)
                loss = self.loss(probs,one_hot)
                acc = self.accuracy(predicted_labels,y_batch)           
                
                if idx% verbose_idx == 0:
                    print("---> Epoch: {}/{}[{:.0f}%], Batch Size: {},Loss:{} ,Accuracy: {}".format(e+1,self.epochs,self.batch_size*idx*100/X.shape[0],self.batch_size,loss.sum(),acc))
                
                self.backward_pass(probs,one_hot)
                self.optimizer()
            # training_loss.append(loss.sum())
            # iterations.append(e+1)
            # accuracy.append(acc)
            
                
            
            
        # self.iteration = iterations
        # self.training_loss = training_loss
        # self.training_accuracy = accuracy
            
    def forward_pass(self,X):
        self.inputs =[] # Yi(s)
        
        for i in range(len(self.weights)):

            X = np.append(X,np.ones((X.shape[0],1)),axis=1)
            self.inputs.append(X)
            X = np.matmul(X,self.weights[i])
            if (i < len(self.weights) - 1):
                X = self.acti_fns(X)
            else :
                X = self.softmax(X)
        self.inputs.append(X)
        return X
        
    def backward_pass(self,Y,D):
        self.zero_grad()
        dl_dy = self.loss_grad(Y,D)
        for l in range(len(self.inputs)-1,0,-1):

            if (l == len(self.inputs) -1 ):
                # grad = self.softmax_grad(self.inputs[l])
                dl_dz = Y - D
            else: 
                grad = self.acti_fns_grad(self.inputs[l])
                
                if not (grad.isfinite().any()):
                    raise Exception('check grad')
                
                dl_dz = np.multiply(grad,dl_dy) 
            
            if not (dl_dz.isfinite().any()):
                raise Exception('check dl_dz')
                
                
            if (l == len(self.inputs) - 1):
              self.weights_grad[l-1] += np.matmul(self.inputs[l-1].T,dl_dz)
              dl_dy = np.matmul(dl_dz,self.weights[l-1].T)
            else :
              self.weights_grad[l-1] += np.matmul(self.inputs[l-1].T,dl_dz[:,:-1])
              dl_dy = np.matmul(dl_dz[:,:-1],self.weights[l-1].T)
            
            if not (dl_dl.isfinite().any()):
                raise Exception('check grad')
                
            
    def gradient_descent(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr*self.weights_grad[i]

    def momentum(self):
        for i in range(len(self.weights)):
            self.m[i] = self.beta*self.m[i] - self.lr*self.weights_grad[i]
            self.weights[i] = self.weights[i]  + self.m[i]
    
    
    def predict(self,X):  
        probs = self.predict_proba(X)
        return np.argmax(probs,axis=1)
    
    def predict_proba(self,X):
        return self.forward_pass(X)
        
    def get_params(self):
        return self.weights
        
    def score(self,X,Y):
        probs = self.forward_pass(X)
        return accuracy(probs, Y)
        
    def zero_grad(self):
        for i in range(len(self.weights_grad)):
            self.weights_grad[i] = np.zeros(self.weights_grad[i].shape)

    def loss(self,Y,D):
        ''' Cross_Entropy Loss
            Y: probs (n_instances,M)
            D: Actual Labels (n_instances,M); M:number of neurons in the last layer
        '''
        noise = 1e-9
        return -1*np.mean(D*np.log(Y+noise))
    
    def loss_grad(self,Y,D):
        '''Y : probabilites (n_instances,M)
           D : Actual Labels(n_instances,M); categorical cross entropy (not used)
        '''
        N = Y + 1e-9
        N = -1/N
        return N
    
    
    def relu(self,X):
        return np.maximum(X,0)

    def relu_grad(self,X):
        r = np.zeros(X.shape)
        r[X>0] = 1
        return r
    
    def tanh(self,X):
        return np.tanh(X)
    
    def tanh_grad(self,X):
        return 1 - self.tanh(X)**2
    
    def sigmoid(self,X):
        return 1/(1 + np.exp(-X))
    
    def sigmoid_grad(self,X):
        a = self.sigmoid(X)
        return a*(1 - a)
    
    def softmax(self,X):
        
        exp = np.exp(X - np.max(X))
        return exp / exp.sum(axis = 1,keepdims = True)

                
    def softmax_grad(self,X):
        ''' there is an other of derivative, however for gradient computation this term suffices'''
        S = self.softmax(X)
        out = S*(1- S)
        return out
    
    def accuracy(self,predicted_labels,true_labels):
        s = predicted_labels == true_labels
        return s.sum()/s.shape[0]
        
                  