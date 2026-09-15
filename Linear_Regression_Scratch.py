import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

class Regression:
    def __init__(self, epochs, eta):
        self.epochs = epochs
        self.eta = eta
        
    def initialize_weights(self, n_features):
        'Initializing w per n and b'
        limit = 1/ np.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, n_features)
        self.b = 0.0
        
    def fit(self, X, y):
        'Training the model'
        self.training_error = []
        self.initialize_weights(n_features = X.shape[1])
        
        'Gradient Descent per Epochs'
        for i in range(self.epochs):
            y_pred = X.dot(self.w) + self.b
            'Calculate loss'
            mse = np.mean(0.5 * (y - y_pred) ** 2)
            self.training_error.append(mse)
            
            grad_w = -(y - y_pred).dot(X) / X.shape[0]
            grad_b = np.mean(-(y - y_pred))
            
            self.w = self.w - self.eta * grad_w
            self.b = self.b - self.eta * grad_b
            if i % 100 == 0:
                print(f"Epoch {i} | MSE: {mse:.4f} | w: {self.w} | b: {self.b:.4f}")

            
    def predict(self, X):
        return X.dot(self.w) + self.b
    
    
def generate_data():
    np.random.seed(10)
    X = np.random.normal(loc = 170, scale = 10, size = 1000).reshape(-1, 1)
    noise = np.random.normal(loc = 0, scale = 2, size = 1000)
    y = (X.flatten() - 170) * .5 + noise
    
    return X, y

X, y = generate_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True, random_state = 30, train_size = .8)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Regression(epochs = 1000, eta = 0.01)
model.fit(X_train, y_train)

print('Finalized Weights & Biases:\n')
print(f'Weight: {model.w}')
print(f'Bias: {model.b}')

y_pred = model.predict(X_test)

print('\nAccuracy Scores:')
print(f' R2 Score: {r2_score(y_test, y_pred):.4f}')
print(f'MSE Score: {mean_squared_error(y_test, y_pred):.4f}')
print(f'MAE Score: {mean_absolute_error(y_test, y_pred):.4f}')
