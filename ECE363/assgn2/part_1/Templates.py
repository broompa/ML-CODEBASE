import numpy as np
class MLPClassifier:

    acti_fns = ['relu', 'sigmoid', 'tanh','linear']
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
        elif activation == 'linear':
            self.acti_fns = self.linear
            self.acti_fns_grad = self.linear_grad
    
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
    

    def fit(self,X,Y,verbose_idx= None,X_val = None,Y_val = None):

        training_loss = []
        training_accuracy = []
        validation_loss =  []
        validation_accuracy = []
        val = False
        
        if (type(X_val) == np.ndarray  and type(Y_val) == np.ndarray):
            val = True
            
        
        if (verbose_idx is None):
            verbose_idx = X.shape[0]//self.batch_size//5 # 5 verbose output in each epoch

        for e in range(self.epochs):
            sample_indicies = np.random.permutation(X.shape[0])
            running_loss = 0
            for idx in range(X.shape[0]//self.batch_size):
                rows = sample_indicies[idx*self.batch_size: (idx+1)*(self.batch_size) ]
                
                x_batch = X[rows,:]
                y_batch = Y[rows]
                one_hot = np.eye(9 + 1)[y_batch] # D(s)
                
                probs = self.forward_pass(x_batch) # Y(s)           
                predicted_labels = np.argmax(probs,axis = 1)
                loss = self.loss(probs,one_hot)  
                acc = self.accuracy(predicted_labels,y_batch)
                # running_loss += loss
                
                self.backward_pass(probs,one_hot)
                self.optimizer()
                
                if idx% verbose_idx == 0:
                    print("Epoch: {}/{}[{:.0f}%], Batch Size: {},Loss:{}, Acc:{} ".format(e+1,self.epochs,self.batch_size*idx*100/X.shape[0],self.batch_size,loss.sum(),acc))

            train_loss, train_acc = self.loss_n_score(X,Y)

            training_accuracy.append(train_acc)
            training_loss.append(train_loss)
            out = " === Train Loss: {:.5f} , Train Accuracy: {:.5f}".format(train_loss, train_acc)

            if val :
                val_loss, val_acc = self.loss_n_score(X_val, Y_val)
                validation_accuracy.append(val_acc)
                validation_loss.append(val_loss)
                out = out + ", Val Loss: {:.5f} , Val Accuracy: {:.5f} ===".format(val_loss, val_acc)
            print(out)

        self.training_loss = training_loss
        self.training_accuracy = training_accuracy
        self.validation_accuracy = validation_accuracy
        self.validation_loss = validation_loss


            
                
            
            
    
            
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
                dl_dz = Y - D 
            else: 
                grad = self.acti_fns_grad(self.inputs[l])
                dl_dz = np.multiply(grad,dl_dy) 
                           
                
            if (l == len(self.inputs) - 1):
              self.weights_grad[l-1] += np.matmul(self.inputs[l-1].T,dl_dz)
              dl_dy = np.matmul(dl_dz,self.weights[l-1].T)
            else :
              self.weights_grad[l-1] += np.matmul(self.inputs[l-1].T,dl_dz[:,:-1])
              dl_dy = np.matmul(dl_dz[:,:-1],self.weights[l-1].T)
            
    def gradient_descent(self):
        for i in range(len(self.weights)):
            self.weights[i] -= self.lr*self.weights_grad[i]
        self.inputs = []
        self.zero_grad()
        # self.weights_grad = []

    
    
    def predict(self,X):  
        probs = self.predict_proba(X)
        return np.argmax(probs,axis=1)
    
    def predict_proba(self,X):
        return self.forward_pass(X)
        
    def get_params(self):
        return self.weights
        
    def score(self,X,Y):
        predicted = self.predict(X)
        return self.accuracy(predicted, Y)
        
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
    
    def linear(self,X):
        return X
    
    def linear_grad(self,X):
        return np.ones(X.shape)
    
    def softmax(self,X):
        exp = np.exp(X - np.max(X,axis = 1,keepdims = True))
        exp = exp / exp.sum(axis = 1,keepdims = True)
        return exp

                
    def softmax_grad(self,X):
        ''' there is an other of derivative, however for gradient computation this term suffices'''
        if not (np.isfinite(X).all()):
            raise Exception('Softmax grad')
        S = self.softmax(X)
        out = S*(1- S)
        return out
    
    def accuracy(self,predicted_labels,true_labels):
        s = predicted_labels == true_labels
        return s.sum()/s.shape[0]
    
    def loss_n_score(self,X,Y):
        one_hot = np.eye(9 + 1)[Y] # D(s)
        probs = self.forward_pass(X) # Y(s)           
        predicted_labels = np.argmax(probs,axis = 1)
        loss = self.loss(probs,one_hot)
        acc = self.accuracy(predicted_labels , Y)  
        return loss, acc

        
                  
