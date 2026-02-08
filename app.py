import random
import math
import numpy as np

class Layer_Relu():
    def __init__(self,input_size,output):
        self.input_vals = np.zeros(input_size)
        self.output = np.zeros(output)
        self.biases = np.zeros(output)
        self.weights = np.random.uniform(-1,1,size=(input_size,output))

        # self.d_biases 
        # self.d_weights
        # self.d_inputs
           
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
    
class Layer_softmax_plus_crossentropy(Layer_Relu):
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
        dvalues = dvalues.copy()
        dvalues = (self.output - target_matrix) / self.output.shape[0]
        self.d_biases = np.sum(dvalues,axis=0,keepdims=True)
        self.d_weights = np.dot(self.input_vals.T,dvalues)
        self.d_inputs = np.dot(dvalues,self.weights.T)
        return self.d_inputs

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
        
class Optimizer():
    def __init__(self,alfa,decay=0.001):
        self.learning_rate=alfa
        self.current_lr=alfa
        self.decay=decay
        self.iteratrions=0 
    
    def optimize(self,neural_network):
        self.current_lr = self.learning_rate/(1+self.decay*self.iteratrions)
        for layer in neural_network.layers:
            layer.weights += -(layer.d_weights*self.learning_rate)
            layer.biases += -(layer.d_biases*self.learning_rate)


        self.iteratrions+=1

class Adam_optimizer():
    def __init__(self,learning_rate=1.,decay=0.,epsilon=1e-7,beta1=0.9,beta2=0.99,w_lambda2=0,b_lambda2=0):
        self.learning_rate=learning_rate
        self.decay=decay
        self.current_learning_rate=learning_rate
        self.iterations=0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.w_lambda2 = w_lambda2
        self.b_lambda2 = b_lambda2

    def optimize(self ,neural_network):
        if self.decay:
            self.current_learning_rate = self.learning_rate * 1./(1.+self.decay*self.iterations)

        for layer in neural_network.layers:
            if not hasattr(layer,'weight_momentum'):
                layer.weight_momentum = np.zeros_like(layer.weights)
                layer.bias_momentum = np.zeros_like(layer.biases)
                layer.weight_cache = np.zeros_like(layer.weights)
                layer.bias_cache = np.zeros_like(layer.biases)


            layer.weight_momentum = (self.beta1 * layer.weight_momentum + (1 - self.beta1) * layer.d_weights)
            layer.bias_momentum = (self.beta1 * layer.bias_momentum + (1 - self.beta1) * layer.d_biases)

            t=self.iterations+1
            w_momentum_term = (layer.weight_momentum) / (1-self.beta1**t)
            b_momentum_term = (layer.bias_momentum) / (1-self.beta1**t)


            layer.weight_cache = (self.beta2 * layer.weight_cache + (1-self.beta2) * np.square(layer.d_weights))
            layer.bias_cache = (self.beta2 * layer.bias_cache + (1-self.beta2) * np.square(layer.d_biases))

            w_cache_term = (layer.weight_cache) / (1-self.beta2**t)
            b_cache_term = (layer.bias_cache) / (1-self.beta2**t)


            w_alfa_factor = w_momentum_term/(np.sqrt(w_cache_term) + self.epsilon)
            b_alfa_factor = b_momentum_term/(np.sqrt(b_cache_term) + self.epsilon)
            # --------------------------

            layer.weights += -(self.current_learning_rate * w_alfa_factor)
            layer.biases += -(self.current_learning_rate * b_alfa_factor)   
        
        self.iterations += 1



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
    print(network.feedForward([1,1,1,1],[1]))