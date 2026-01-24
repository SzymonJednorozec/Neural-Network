import random
import math
import numpy as np

class Layer_Relu():
    def __init__(self,input_size,output):
        self.input_vals = np.zeros(input_size)
        self.output = np.zeros(output)
        self.biases = np.zeros(output)
        self.weights = np.random.uniform(-1,1,size=(input_size,output))
           
    def feedForward(self,input_vals): 
        self.input_vals = np.atleast_2d(input_vals)
        self.output = Relu(np.dot(self.input_vals,self.weights) + self.biases) #Relu(dot product + biases)       
        return self.output
    
class Layer_softmax(Layer_Relu):
    def feedForward(self,input_vals):
        self.input_vals = np.atleast_2d(input_vals)
        self.output = softmax(np.dot(self.input_vals,self.weights) + self.biases) 
        return self.output

class Neural_network():
    def __init__(self,nodes):
        self.layers=[]
        for i in range(len(nodes)-1):
            if i == len(nodes)-2:
                self.layers.append(Layer_softmax(nodes[i],nodes[i+1]))
            else:
                self.layers.append(Layer_Relu(nodes[i],nodes[i+1]))
    
    def feedForward(self,input_vals):
        x = self.layers[0].feedForward(input_vals)
        for i in range(1,len(self.layers)):
            x = self.layers[i].feedForward(x)
        return x

def Relu(arr):
    return np.maximum(0,arr)

def softmax(arr):
    exponential_vals = np.exp(arr)
    outcome = exponential_vals/np.sum(exponential_vals,axis=1)
    return outcome

if __name__=='__main__':
    network = Neural_network([4,7,2])
    print(network.feedForward([1,1,1,1]))