from core.network import Neural_network
from core.optimizers import Adam_optimizer
import numpy as np


if __name__=='__main__':
    X = np.random.randn(100, 2)
    y = (X[:, 0] < X[:, 1]).astype(int)

    network = Neural_network([2, 8, 2])
    optimizer = Adam_optimizer(learning_rate=0.05, decay=1e-3)

    for epoch in range(201):
        loss_vector = network.feedForward(X, y)
        loss = np.mean(loss_vector)

        network.backPropagate(y)

        optimizer.optimize(network)

        if epoch % 20 == 0:
            predictions = np.argmax(network.layers[-1].output, axis=1)
            accuracy = np.mean(predictions == y)
            print(f"Epoch {epoch:3} | Loss: {loss:.4f} | Accuracy: {accuracy:.2%}")