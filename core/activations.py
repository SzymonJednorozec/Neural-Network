import numpy as np

def Relu(arr):
    return np.maximum(0,arr)

def softmax(arr):
    exponential_vals = np.exp(arr)
    outcome = exponential_vals/np.sum(exponential_vals,axis=1,keepdims=True)
    return outcome

def cross_entropy(arr,target):
    arr = np.clip(arr, 1e-7, 1 - 1e-7)
    if np.ndim(target) == 2:
        outcome = -np.sum(target*np.log(arr),axis=1)
    elif np.ndim(target) == 1:
        batch_size = arr.shape[0]
        outcome =  -np.log(arr[np.arange(batch_size), target])
    
    return outcome