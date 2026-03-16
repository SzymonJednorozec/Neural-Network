import random
import math
import numpy as np
from .activations import *

class Layer_Relu():
    def __init__(self,input_size,output):
        self.input_vals = np.zeros(input_size)
        self.output = np.zeros(output)
        self.biases = np.zeros((1, output))
        self.weights = np.random.uniform(-1,1,size=(input_size,output))
           
    def feedForward(self,input_vals,target): 
        self.input_vals = np.atleast_2d(input_vals)
        self.output = Relu(np.dot(self.input_vals,self.weights) + self.biases) #Relu(dot product + biases)       
        return self.output
    
    def backPropagate(self,dvalues, target): #dvalues - accumulated derivatives to this point
        dvalues = dvalues.copy()
        dvalues = np.where(self.output > 0, dvalues, 0)
        self.d_biases = np.sum(dvalues,axis=0,keepdims=True)
        self.d_weights = np.dot(self.input_vals.T,dvalues)
        self.d_inputs = np.dot(dvalues,self.weights.T)
        return self.d_inputs
    
class Layer_softmax_plus_crossentropy():
    def __init__(self, input_size, output):
        self.input_vals = np.zeros(input_size)
        self.output = np.zeros(output)
        self.biases = np.zeros((1, output))
        self.weights = np.random.uniform(-1,1,size=(input_size,output))

    def feedForward(self,input_vals,target):
        self.input_vals = np.atleast_2d(input_vals)
        self.output = softmax(np.dot(self.input_vals,self.weights) + self.biases) 
        loss = cross_entropy(self.output,target)
        return loss
    
    def backPropagate(self, dvalues, target): #calculating derivative for cross_entorpy(softmax(x),target)
        if np.ndim(target)==1:
            target_matrix = np.zeros(shape=(self.output.shape))
            batch_size = target.shape[0]
            target_matrix[np.arange(batch_size),target]=1
        else:
            target_matrix=target
        dvalues = (self.output - target_matrix) / self.output.shape[0]
        self.d_biases = np.sum(dvalues,axis=0,keepdims=True)
        self.d_weights = np.dot(self.input_vals.T,dvalues)
        self.d_inputs = np.dot(dvalues,self.weights.T)
        return self.d_inputs