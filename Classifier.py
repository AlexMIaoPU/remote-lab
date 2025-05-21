from sklearn.neighbors import KNeighborsClassifier


class KNNClassifier:
    """1-NN Classifier for classifying colour pixels into classes."""
    def __init__(self, data_path=""):
        """
        Initialize the KNN classifier with the given data path.
        Expects to read training data and classes from a CSV file.
        """
        data = []
        classes = []
        knn = KNeighborsClassifier(n_neighbors=1)

        knn.fit(data, classes)
        self._classifier = knn

    def classify(self, data):
        prediction = self._classifier.predict(data)

        return prediction