import numpy as np

class SimpleNNVectorized:
    """A small 2-layer neural network vectorized with NumPy for binary classification.

    API:
      - forward(X): returns probabilities shape (n_samples, 1)
      - backward(X, y): performs a gradient step in place
      - compute_loss(y, y_pred): binary cross-entropy
      - train_batch(X, y, epochs): returns loss history list
      - predict(X): returns probabilities
    """

    def __init__(self, input_size, hidden_size, output_size=1, learning_rate=0.01, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.lr = learning_rate
        # +1 for bias in weights (we will handle bias by concatenating ones)
        self.W1 = np.random.uniform(-0.5, 0.5, size=(input_size + 1, hidden_size))
        self.W2 = np.random.uniform(-0.5, 0.5, size=(hidden_size + 1, output_size))

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(a):
        return a * (1 - a)

    @staticmethod
    def add_bias_np(X):
        # X: (n_samples, n_features) -> add column of ones on the left
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def forward(self, X):
        # X: (n_samples, n_features)
        self.A1 = self.add_bias_np(X)  # (n_samples, input_size+1)
        self.Z2 = self.A1.dot(self.W1)  # (n_samples, hidden_size)
        self.A2_nobias = self.sigmoid(self.Z2)  # (n_samples, hidden_size)
        self.A2 = self.add_bias_np(self.A2_nobias)  # (n_samples, hidden_size+1)
        self.Z3 = self.A2.dot(self.W2)  # (n_samples, output_size)
        self.A3 = self.sigmoid(self.Z3)  # (n_samples, output_size)
        return self.A3

    def backward(self, X, y):
        # X: (n_samples, n_features), y: (n_samples,) or (n_samples,1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        m = X.shape[0]

        # forward pass already sets A1, A2, A3
        # Output error
        output_errors = (self.A3 - y) * self.sigmoid_derivative(self.A3)  # (n_samples, output_size)

        # Hidden error (exclude bias of W2)
        W2_no_bias = self.W2[1:, :]  # (hidden_size, output_size)
        hidden_errors = output_errors.dot(W2_no_bias.T) * self.sigmoid_derivative(self.A2_nobias)  # (n_samples, hidden_size)

        # Gradients
        dW2 = (self.A2.T.dot(output_errors)) / m  # (hidden_size+1, output_size)
        dW1 = (self.A1.T.dot(hidden_errors)) / m  # (input_size+1, hidden_size)

        # Update
        self.W2 -= self.lr * dW2
        self.W1 -= self.lr * dW1

    def compute_loss(self, y, y_pred):
        # y: (n_samples,) or (n_samples,1); y_pred: (n_samples,1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = - (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return float(np.mean(loss))

    def train_batch(self, X, y, epochs=1000, verbose=False):
        loss_history = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y)
            loss = self.compute_loss(y, y_pred)
            loss_history.append(loss)
            if verbose and (epoch % 100 == 0):
                print(f"Epoch {epoch}, Loss: {loss}")
        return loss_history

    def predict(self, X):
        return self.forward(X)
