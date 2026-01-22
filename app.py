import random
import math
import numpy as np

class Layer():
    def __init__(self,input_size,output):
        self.input_vals = np.zeros(input_size)
        self.output = np.zeros(output)
        self.biases = np.zeros(output)
        self.weights = np.random.uniform(-1,1,size=(input_size,output))
        
    
    @staticmethod
    def feedForward(layer,input_vals): 
        layer.input_vals = np.array(input_vals)
        layer.output = Relu(np.dot(layer.input_vals,layer.weights) + layer.biases) #Relu(dot product + biases)
            
        return layer.output


class Neural_network():
    def __init__(self,nodes):
        self.layers = [Layer(nodes[i],nodes[i+1]) for i in range(len(nodes)-1)]
    
    @staticmethod
    def feedForward(layers,input_vals):
        x = Layer.feedForward(layers[0],input_vals)
        for i in range(1,len(layers)):
            x = Layer.feedForward(layers[i],x)
        return x

def Relu(arr):
    return np.maximum(0,arr)

if __name__=='__main__':
    network = Neural_network([4,7,2])
    print(Neural_network.feedForward(network.layers,[1,1,1,1]))