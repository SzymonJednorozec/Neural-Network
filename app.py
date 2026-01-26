import random
import math
import numpy as np

class Layer_Relu():
    def __init__(self,input_size,output):
        self.input_vals = np.zeros(input_size)
        self.output = np.zeros(output)
        self.biases = np.zeros(output)
        self.weights = np.random.uniform(-1,1,size=(input_size,output))

        self.d_biases 
        self.d_weights
        self.d_inputs
           
    def feedForward(self,input_vals): 
        self.input_vals = np.atleast_2d(input_vals)
        self.output = Relu(np.dot(self.input_vals,self.weights) + self.biases) #Relu(dot product + biases)       
        return self.output
    
    def backPropagate(self,dvalues): #dvalues - accumulated derivatives to this point
        dvalues = dvalues.copy()
        dvalues = np.where(self.output > 0, dvalues, 0)
        self.d_biases = np.sum(dvalues,axis=0,keepdims=True)
        self.d_weights = np.dot(self.input_vals.T,dvalues)
        self.d_inputs = np.dot(dvalues,self.weights.T)
        return self.d_inputs
    
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
    outcome = exponential_vals/np.sum(exponential_vals,axis=1,keepdims=True)
    return outcome

def cross_entropy(arr,target):
    if np.ndim(target) == 2:
        outcome = -np.sum(target*np.log(arr),axis=1)
    elif np.ndim(target) == 1:
        batch_size = arr.shape[0]
        outcome =  -np.log(arr[np.arange(batch_size), target])
    
    return outcome


if __name__=='__main__':
    network = Neural_network([4,7,2])
    print(cross_entropy(network.feedForward([1,1,1,1]),[1]))