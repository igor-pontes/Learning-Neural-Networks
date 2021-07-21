import numpy as np
import math
class NeuralNetwork(object):
    def __init__(self, *args): 
        self.layers = [np.ones((1,args[x]+1)) if x < len(args)-1 else np.ones((1,args[x])) for x in range(len(args))]
        self.weights = [np.random.randn(self.layers[x+1].shape[1],self.layers[x].shape[1])*np.sqrt(2/self.layers[x].shape[1]) for x in range(len(args)-1)] # He-et-al Initialization
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def fowardprop(self, image = None):
        temp = [1]
        temp.extend(image)
        self.layers[0] = np.transpose(np.asarray(temp, dtype=np.float32).reshape(len(temp),1))
        for i, w in zip(range(1,len(self.layers)), self.weights):
            self.layers[i] = self.sigmoid(np.matmul(self.layers[i-1], np.transpose(w)))
            print(self.layers[i])
       # return np.where(self.layers[-1] == np.amax(self.layers[-1]))[1]

    def backprop(self, m):
        None 
    
    def sgd(self): # TODO
        None
