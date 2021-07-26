import matplotlib
from PIL import Image
import numpy as np
from process import processfile
from NeuralNetwork import NeuralNetwork
data_i = processfile("train-images-idx3-ubyte")
data_l = processfile("train-labels-idx1-ubyte")
images = [(np.asarray(Image.frombytes(mode = "L", size=(len(data_i[0]),1), data=data_i[x]), dtype=np.float32).flatten(), data_l[x]) for x in range(len(data_i))]

n = NeuralNetwork(len(images[0][0]), 15, 10)
n.backprop(images, 3)
image = images[1]
print(n.predict(image[0]))
print("---")
print(image[1])

