import numpy as np

class SDG_ptimizer():
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