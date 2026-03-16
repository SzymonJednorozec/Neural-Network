from .layers import *

class Neural_network():
    def __init__(self,nodes):
        self.layers=[]
        for i in range(len(nodes)-1):
            if i == len(nodes)-2:
                self.layers.append(Layer_softmax_plus_crossentropy(nodes[i],nodes[i+1]))
            else:
                self.layers.append(Layer_Relu(nodes[i],nodes[i+1]))
    
    def feedForward(self,input_vals,target):
        x = self.layers[0].feedForward(input_vals,target)
        for i in range(1,len(self.layers)):
            x = self.layers[i].feedForward(x,target)
        return x
    
    def backPropagate(self,target):
        x = self.layers[-1].backPropagate(None,target)
        for i in range(len(self.layers)-2,-1,-1):
            x = self.layers[i].backPropagate(x,target)