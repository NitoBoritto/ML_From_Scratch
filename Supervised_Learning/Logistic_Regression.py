import numpy as np 
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

class LogisticRegression:
    def __init__(self, epochs = 1000, eta = 0.01, threshold = 0.5):
        self.epochs = epochs
        self.eta = eta
        self.threshold = threshold
        
    def __linear(self, X):
        return X.dot(self.w) + self.b
        
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1/(1 + np.exp(-z))
        
    def initialize_weights(self, n_features):
        'Initializing w per n and b'
        limit = 1/ np.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, n_features)
        self.b = 0.0
        
    def fit(self, X, y):
        'Training the model'
        self.training_error = []
        self.initialize_weights(n_features = X.shape[1])
        
        'Binary Cross Entropy per Epochs'
        for i in range(self.epochs):
            z = self.__linear(X)
            y_pred = self._sigmoid(z)
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            'Calculate loss'
            bce = -np.mean((y * np.log(y_pred)) + ((1 - y) * np.log(1 - y_pred)))
            self.training_error.append(bce)
            
            grad_w = -(y - y_pred).dot(X) / X.shape[0]
            grad_b = np.mean(-(y - y_pred))
            
            self.w -= self.eta * grad_w
            self.b -= self.eta * grad_b
            if i % 100 == 0:
                print(f"Epoch {i} | Cross Entropy: {bce:.4f} | w: {self.w} | b: {self.b:.4f}")
            
    def predict_proba(self, X):
        z = self.__linear(X)
        return self._sigmoid(z)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= self.threshold).astype(int)
    


def main():
    X, y = make_classification(n_samples = 10000, n_featurejs = 3, n_informative = 2,
                            n_redundant = 1, n_classes = 2, random_state = 30)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 30, train_size = .8)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(epochs = 1000, eta = 0.01)
    model.fit(X_train, y_train)

    print('\nFinalized Weights & Biases:')
    print(f'Weight: {model.w}')
    print(f'Bias: {model.b}')

    y_pred = model.predict(X_test)

    print('\nEvaluation:')
    print(f' Classification Report:\n{classification_report(y_test, y_pred)}\n')
    print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
    
if __name__ == "__main__":
    main()