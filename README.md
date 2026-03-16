# Numpy Neural Engine

A lightweight, from-scratch neural network library implemented in Python using only NumPy. This project demonstrates deep understanding of neural network fundamentals, backpropagation, optimization algorithms, and software engineering principles.

## Features

### Core Components
- **Neural Network Architecture**: Flexible multi-layer perceptron with configurable layer sizes
- **Activation Functions**: ReLU and Softmax implementations with numerical stability
- **Loss Functions**: Cross-entropy loss optimized for classification tasks
- **Optimizers**: 
  - Stochastic Gradient Descent (SGD) with learning rate decay
  - Adam optimizer with momentum and adaptive learning rates

### Key Capabilities
- Forward and backward propagation through the network
- Automatic differentiation for gradient computation
- Batch processing support
- Modular design for easy extension

## Architecture

```
core/
├── __init__.py          # Package initialization and exports
├── activations.py       # ReLU, Softmax, Cross-entropy functions
├── layers.py           # Layer implementations (ReLU, Softmax+CrossEntropy)
├── network.py          # NeuralNetwork class with feedforward/backprop
└── optimizers.py       # SGD and Adam optimizers
```

## Installation

```bash
pip install -e .
```

## Usage

### Basic Classification Example

```python
from core import Neural_network, Adam_optimizer
import numpy as np

# Create synthetic data
X = np.random.randn(100, 2)
y = (X[:, 0] < X[:, 1]).astype(int)

# Initialize network and optimizer
network = Neural_network([2, 8, 2])  # 2 inputs, 8 hidden, 2 outputs
optimizer = Adam_optimizer(learning_rate=0.05, decay=1e-3)

# Training loop
for epoch in range(201):
    loss_vector = network.feedForward(X, y)
    loss = np.mean(loss_vector)

    network.backPropagate(y)
    optimizer.optimize(network)

    if epoch % 20 == 0:
        predictions = np.argmax(network.layers[-1].output, axis=1)
        accuracy = np.mean(predictions == y)
        print(f"Epoch {epoch:3} | Loss: {loss:.4f} | Accuracy: {accuracy:.2%}")
```

### Running Examples

```bash
cd examples
python simple_classification.py
```

## Technical Implementation

### Forward Propagation
- Input flows through layers sequentially
- Each layer applies weights, biases, and activation functions
- Loss computed at output layer for training

### Backpropagation
- Gradients computed using chain rule
- Derivatives flow backward through layers
- Weights and biases updated based on computed gradients

### Optimization
- **SGD**: Simple gradient descent with configurable learning rate decay
- **Adam**: Adaptive moment estimation with bias correction and L2 regularization support

## Dependencies

- **NumPy**: Core numerical computations and array operations


## License

MIT License - feel free to use in your projects!
