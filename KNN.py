## Scratch implementation of the KNN algorithm

# Steps:
# Calculate distance between training sample and the testing sample
# look for the nearest neighbour popping up most frequently
# find out most common class label
# then classify!

# d=((x2-x1)^2+(y2-y1)^2)^1/2

import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train] #calculate distance
        k_idx = np.argsort(distances)[: self.k] 
        k_neighbor_labels = [self.y_train[i] for i in k_idx] #extract labels
        most_common = Counter(k_neighbor_labels).most_common(1)#return common labels
        return most_common[0][0]

if __name__ == "__main__":

    def acc(y_true, y_pred):
        acc = np.sum(y_true == y_pred) / len(y_true)
        return acc

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    k = 3 #testing with three neighbours

    clf = KNN(k=k)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    print("KNN Accuracy -> ", acc(y_test, predictions))
