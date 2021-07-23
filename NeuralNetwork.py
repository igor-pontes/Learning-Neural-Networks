import numpy as np
import math
class NeuralNetwork(object):
    def __init__(self, *args): 
        self.layers = [np.ones((1,args[x]+1)) if x < len(args)-1 else np.ones((1,args[x])) for x in range(len(args))]
        # He-et-al Initialization
        self.weights = [np.random.randn(self.layers[x+1].shape[1],self.layers[x].shape[1])*np.sqrt(2/self.layers[x].shape[1]) for x in range(len(args)-1)]
    
    def sigmoidgradient(self, x):
        # X with sigmoid func
        return x*(1-x) 
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def forwardprop(self, image = None):
        temp = [1]
        temp.extend(image)
        self.layers[0] = np.transpose(np.asarray(temp).reshape(len(temp),1))
        for i, w in zip(range(1,len(self.layers)), self.weights):
            self.layers[i] = self.sigmoid(np.asarray(np.matmul(self.layers[i-1], np.transpose(w)), dtype=np.float128))
        return self.layers[-1]

    def backprop(self, m):
        delta = [0, 0]
        m = m[0:30000]
        for i in m:
            error = []
            self.forwardprop(i[0])
            error.extend(self.layers[-1]-i[1])

            for l, w, e in zip(range(len(self.layers)-2, 0, -1), range(len(self.weights)-1, 0, -1), range(0, len(self.layers)-2)):
                error[e] = error[e].reshape(len(error[e]), 1)
                error.extend([np.matmul(np.transpose(self.weights[w][:,:]),error[e])*self.sigmoidgradient(np.transpose(self.layers[l][:,:]))])
            for d, l in zip(range(0, len(error)), range(len(self.layers)-2, -1,-1)):
                delta[d] += np.matmul(error[d],self.layers[l])
        
        for w, d in zip(range(0, len(delta)), range(len(delta)-1, -1, -1)):
            self.weights[w] = (1/len(m)) * delta[d]

    def predict(self, image):
        l = self.forwardprop(image)

        return np.argmax(l)

    def sgd(self): # TODO
        None
