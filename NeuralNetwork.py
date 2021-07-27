import numpy as np
import math
class NeuralNetwork(object):
    def __init__(self, *args): 
        self.layers = [x for x in args]
        self.weights = [np.random.randn(x, y + 1) for x, y in zip(self.layers[1:], self.layers[:-1])]
        self.sizes = [x + 1 if x != args[-1] else x for x in args]
   
    def sigmoidprime(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x)) 
    
    def sigmoid(self, x):
        x = np.clip(x, -900, 900)
        return 1./(1.+np.exp(-x))
    
    def forwardprop(self, image = None):
        a = [np.ones((x,1), dtype = np.float128) for x in self.sizes]
        z = [np.ones((x,1), dtype = np.float128) for x in self.sizes]

        a[0][1:] = np.asarray(image).reshape(len(image), 1)
        z[0] = a[0]
        
        for w, l in zip(self.weights[:-1], range(1,len(a)-1)):
            a[l] = np.vstack((np.ones((1,1)), self.sigmoid(np.dot(w, a[l-1]))))
            z[l] = np.vstack((np.ones((1,1)), np.dot(w, a[l-1])))
        
        a[-1] = self.sigmoid(np.dot(self.weights[-1], a[-2]))
        z[-1] = np.dot(self.weights[-1], a[-2])

        return (z, a)

    def gd(self, data, epochs, lr, lamb):
        for x in range(0, epochs):
            self.backprop(data, lr, lamb)
            print("Epoch {0} accuracy: {1}%".format(x, self.accuracy(data[0:1000])))

    def backprop(self, m, lr, lamb):
        m = m[0:60000]
        size = len(m)
        delta = [np.zeros(w.shape, dtype = np.float128) for w in self.weights]
        for i in m:
            error = [np.zeros(self.sizes[x], dtype = np.float128) for x in range(1, len(self.sizes))]    
            output = np.asarray([0 if x != i[1] else 1 for x in range(0,self.layers[-1])]).reshape(self.layers[-1], 1)
            z, a = self.forwardprop(i[0])
            error[-1] = a[-1] - output

            for w, e in zip(range(1, len(self.weights)), range(1, len(self.layers))):
                error[-e-1] = np.dot(self.weights[-w].transpose(), error[-e]) * self.sigmoidprime(z[-e-1])
            for e, l in zip(range(0, len(error)-1), range(0, len(self.layers)-1)):
                delta[e] += np.dot(error[e][1:,:], a[l].transpose())

            delta[-1] += np.dot(error[-1], a[-2].transpose())
        
        for w in range(0, len(self.weights)):
            temp = np.hstack(((1/size)*delta[w][:,0].reshape(delta[w][:,0].shape[0], 1), (1/size)*delta[w][:,1:] + (lamb/size)*self.weights[w][:,1:]))
            self.weights[w] = self.weights[w] - lr*temp

    def predict(self, image):
        _, l = self.forwardprop(image)
        return np.argmax(l[-1])

    def accuracy(self, data):
        r = 0
        for x in data:
           i = self.predict(x[0])
           if i == x[1]:
               r += 1
        return r/len(data)*100
    
