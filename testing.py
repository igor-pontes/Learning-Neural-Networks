import matplotlib
from PIL import Image
import numpy as np
from process import processfile
from NeuralNetwork import NeuralNetwork
data = processfile("train-images-idx3-ubyte")
images = [np.asarray(Image.frombytes(mode = "L", size=(len(data[0]),1), data=data[x]), dtype=np.float32).flatten() for x in range(len(data))]
#i_array = np.asarray(image).flatten()
n = NeuralNetwork(len(images[0]),20,10)
lol = n.fowardprop(images[0])
#print(lol)
#image.save("number.jpeg")

