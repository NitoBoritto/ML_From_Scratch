import numpy as np
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

class DecisionNode:
    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None):
        'Initializing decision node parameters'
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf_node(self):
        'Checking if the node is a leaf'
        return self.value is not None

class DecisionTree:
    def __init__(self, min_samples_split = 2, max_depth = 150, n_features = None, criterion = "gini"):
        'Initializing decision tree parameters'
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.criterion = criterion
        self.root = None


    def fit(self, X, y):
        'Training the decision tree'
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)


    def _grow_tree(self, X, y, depth=0):
            'Growing the decision tree recursively'
            n_samples, n_feats = X.shape
            n_labels = len(np.unique(y))
            
            # Check Stopping Criteria
            if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
                leaf_value = self._most_common_label(y)
                
                return DecisionNode(value=leaf_value)
            
            feature_i = np.random.choice(n_feats, self.n_features, replace=False)
            best_feature, best_threshold = self._best_split(X, y, feature_i)
            
            if best_feature is None:
                leaf_value = self._most_common_label(y)

                return DecisionNode(value=leaf_value)

            # Create Child Nodes
            left_i, right_i = self._split(X[:, best_feature], best_threshold)
            left = self._grow_tree(X[left_i, :], y[left_i], depth + 1)
            right = self._grow_tree(X[right_i, :], y[right_i], depth + 1)
            
            return DecisionNode(best_feature, best_threshold, left, right)

    
    def _best_split(self, X, y, feature_i):
        'Finding the best feature and threshold split'
        best_gain = -1
        split_i, split_threshold = None, None
        
        for feature in feature_i:
            X_col = X[:, feature]
            thresholds = np.unique(X_col)
            
            for thresh in thresholds:
                # Calculate Information Gain
                gain = self._information_gain(y, X_col, thresh)

                if gain > best_gain:
                    best_gain = gain
                    split_i = feature
                    split_threshold = thresh

        return split_i, split_threshold


    def _information_gain(self, y, X_col, threshold):
        'Calculating information gain for a split'
        # Parent Entropy
        parent_entorpy = self._calculate_impurity(y)
        
        # Create Children
        left_i, right_i = self._split(X_col, threshold)
        
        if len(left_i) == 0 or len(right_i) == 0:
            return -1
        
        # Calculate Children Weighted Entropy
        n = len(y)
        n_l, n_r = len(left_i), len(right_i)
        e_l, e_r = self._calculate_impurity(y[left_i]), self._calculate_impurity(y[right_i])
        
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        
        # Calculate IG
        information_gain = parent_entorpy - child_entropy
        return information_gain


    def _split(self, X_col, split_thresh):
        'Splitting data into left and right branches'
        left_i = np.argwhere(X_col < split_thresh).flatten()
        right_i = np.argwhere(X_col > split_thresh).flatten()
        
        return left_i, right_i


    def _entropy(self, y):
        'Calculating entropy impurity'
        hist = np.bincount(y)
        ps = hist / len(y)
        
        return -np.sum([p * np.log(p) for p in ps if p > 0])


    def _gini(self, y):
        'Calculating Gini impurity'
        hist = np.bincount(y)
        ps = hist / len(y)
        
        return 1.0 - np.sum(ps ** 2)


    def _calculate_impurity(self, y):
        'Selecting the impurity criterion'
        if self.criterion == "gini":
            return self._gini(y)
        
        elif self.criterion == "entropy":
            return self._entropy(y)
        
        else:
            raise ValueError(f"Invalid Criterion: {self.criterion}")


    def _most_common_label(self, y):
        'Finding the most common class label'
        counter = Counter(y)
        
        return counter.most_common(1)[0][0]


    def predict(self, X):
        'Predicting class labels'
        return np.array([self._traverse_tree(x, self.root) for x in X])


    def _traverse_tree (self, x, node):
        'Traversing the tree to make a prediction'
        if node.is_leaf_node():
            return node.value
        
        elif x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        
        else:
            return self._traverse_tree(x, node.right)


def main():
    X, y = make_classification(n_samples = 10000, n_features = 3, n_informative = 2,
                            n_redundant = 1, n_classes = 2, random_state = 30)

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, random_state = 30, train_size = .8)

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = DecisionTree(min_samples_split = 2, max_depth = 150, criterion = "entropy")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print('\nEvaluation:')
    print(f' Classification Report:\n{classification_report(y_test, y_pred)}\n')
    print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')
    
if __name__ == "__main__":
    main()