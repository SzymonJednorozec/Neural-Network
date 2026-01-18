import random
import math

class Layer():
    def __init__(self,input_size,output):
        self.input_vals = [ 0.0 for _ in range(input_size)]
        self.output = [ 0.0 for _ in range(output)]
        self.biases = [ 0.0 for _ in range(output)]
        self.weights = [[random.uniform(-1,1) for _ in range(input_size)] for _ in range(output)]
        
    
    @staticmethod
    def feedForward(layer,input_vals): 
        layer.input_vals = input_vals
        layer.output = [0.0] * len(layer.output) 
        for o in range(len(layer.output)):
            for i in range(len(layer.input_vals)):
                layer.output[o] += layer.input_vals[i] * layer.weights[o][i]
            layer.output[o]+=layer.biases[o]
            layer.output[o] = 1 / (1 + math.exp(-layer.output[o])) 
            
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

if __name__=='__main__':
    network = Neural_network([4,7,2])
    print(Neural_network.feedForward(network.layers,[1,1,1,1]))