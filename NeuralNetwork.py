import numpy as np
import math
class NeuralNetwork(object):
    def __init__(self, *args): 
        self.layers = [x for x in args]
        self.weights = [np.random.randn(self.layers[x+1],self.layers[x]+1)*np.sqrt(2/self.layers[x]) for x in range(len(args)-1)]
    def sigmoidgradient(self, x):
        # X with sigmoid func
        return x*(1-x) 
    
    def sigmoid(self, x):
        return 1./(1.+np.exp(-x))
    
    def forwardprop(self, image = None):
        a = np.empty(len(self.layers), dtype = object)
        temp = [1]
        temp.extend(image)
        a[0] = np.asarray(np.transpose(np.asarray(temp).reshape(len(temp),1)), dtype=np.float128)
        for i, w in zip(range(1,len(self.layers)), self.weights):
            if i == len(self.layers)-1:
                a[i] = self.sigmoid(np.asarray(np.matmul(a[i-1], np.transpose(w)), dtype=np.float128))
            else:
                a[i] = np.concatenate((np.ones((1,1)), self.sigmoid(np.asarray(np.matmul(a[i-1], np.transpose(w)), dtype=np.float128))), axis=1)
        return a

    def backprop(self, m, lamb):
        delta = np.zeros(len(self.weights), dtype = object)
        m = m[0:10000]
        for i in m:
            error = np.empty(len(self.weights), dtype = object)
            a = self.forwardprop(i[0])
            error[0] = np.asarray((a[-1]-i[1]).transpose(), dtype=np.float128)
           
            for e, w, l in zip(range(1, len(self.weights)), range(len(self.weights)-1, -1, -1), range(len(a)-2, -1, -1)):
                #error[e] = error[e].reshape(error[e].shape[1], 1)
                error[e] = np.asarray(np.matmul(self.weights[w].transpose()[1:,:], error[e-1]) * self.sigmoidgradient(a[l].transpose()[1:,:]), dtype=np.float128)
                  
            for d, l in zip(range(0, len(error)), range(len(self.layers)-2, -1,-1)):
                #if d == 0:
                #delta[d] = np.zeros((error[d].shape[0], a[l].shape[1]), dtype = np.float128)
                delta[d] += np.matmul(error[d],a[l])
                #else:
                #    delta[d] += np.matmul(error[d],a[l])[1:]
        for w, d in zip(range(0, len(delta)), range(len(delta)-1, -1, -1)):
            self.weights[w] = np.concatenate(( (1/len(m)) * delta[d][:,0].reshape(delta[d][:,0].shape[0],1), (lamb/len(m)) * delta[d][:,1:]), axis=1)

    def predict(self, image):
        l = self.forwardprop(image)[-1]
        return np.argmax(l)

    def sgd(self): # TODO
        None
