import numpy as np
import pickle
import pandas as pd
def one_hot_encode(seq):
    mapping = {'A': [1, 0, 0, 0],
               'T': [0, 1, 0, 0],
               'G': [0, 0, 1, 0],
               'C': [0, 0, 0, 1]}
    return np.array([mapping.get(base, [0, 0, 0, 0]) for base in seq]).flatten()
def relu(x):
    return np.maximum(0, x)
def relu_deriv(x):
    return (x > 0).astype(float)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)
class DNA_NeuralNetwork:
    def __init__(self, input_size, hidden1=16, hidden2=8, output_size=1, dropout_rate=0.2):
        self.W1 = np.random.randn(input_size, hidden1) * 0.01
        self.b1 = np.zeros((1, hidden1))
        self.W2 = np.random.randn(hidden1, hidden2) * 0.01
        self.b2 = np.zeros((1, hidden2))
        self.W3 = np.random.randn(hidden2, output_size) * 0.01
        self.b3 = np.zeros((1, output_size))
        self.dropout_rate = dropout_rate
    def forward(self, X):
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = relu(self.Z1)
        self.dropout1 = (np.random.rand(*self.A1.shape) > self.dropout_rate).astype(float)
        self.A1 *= self.dropout1
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = relu(self.Z2)
        self.dropout2 = (np.random.rand(*self.A2.shape) > self.dropout_rate).astype(float)
        self.A2 *= self.dropout2
        self.Z3 = self.A2 @ self.W3 + self.b3
        self.A3 = sigmoid(self.Z3)
        return self.A3
    def compute_loss(self, Y_hat, Y):
        m = Y.shape[0]
        return -np.mean(Y * np.log(Y_hat + 1e-8) + (1 - Y) * np.log(1 - Y_hat + 1e-8))
    def backward(self, X, Y):
        m = Y.shape[0]
        dZ3 = self.A3 - Y
        dW3 = self.A2.T @ dZ3 / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        dA2 = dZ3 @ self.W3.T
        dA2 *= self.dropout2
        dZ2 = dA2 * relu_deriv(self.Z2)
        dW2 = self.A1.T @ dZ2 / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = dZ2 @ self.W2.T
        dA1 *= self.dropout1
        dZ1 = dA1 * relu_deriv(self.Z1)
        dW1 = X.T @ dZ1 / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        lr = 0.1
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
    def train(self, X, Y, epochs=1000):
        for epoch in range(epochs):
            Y_hat = self.forward(X)
            loss = self.compute_loss(Y_hat, Y)
            self.backward(X, Y)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
    def predict(self, X):
        Y_hat = self.forward(X)
        return (Y_hat > 0.5).astype(int)
    def predict_proba(self, X):
        return self.forward(X)
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            self.__dict__ = pickle.load(f)
sequences_df = pd.read_csv('synthetic_dna_dataset.csv')
if 'Sequence' not in sequences_df.columns:
    raise ValueError("CSV file must have a column named 'Sequence'.")
X = np.array([one_hot_encode(seq) for seq in sequences_df['Sequence']])
Y = np.random.randint(0, 2, size=len(X)).reshape(-1, 1)
print(f"Training on {X.shape[0]} sequences, each of encoded length {X.shape[1]}")
model = DNA_NeuralNetwork(input_size=X.shape[1])
model.train(X, Y, epochs=1000)
# Save the trained model
model.save("dna_model.pkl")
print("✅ Model saved as dna_model.pkl")
test_seq = input("\nEnter test DNA sequence (e.g., ATGCATGCAT): ").upper()
expected_length = X.shape[1] // 4
if len(test_seq) != expected_length:
    raise ValueError(f"Test sequence must be {expected_length} bases long.")
test_input = one_hot_encode(test_seq).reshape(1, -1)
probability = model.predict_proba(test_input)[0][0]
prediction = 1 if probability > 0.5 else 0
print(f"Prediction for '{test_seq}':\n{'Harmful' if prediction else 'Benign'}")
print(f"Probability: {probability * 100:.2f}%")
