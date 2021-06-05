from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
import numpy as np


class Data:
    def __init__(self):
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load_data(self, train_val_test_split=(60, 20, 20)):
        X, t = Data.fetch_data()

        X, t = Data.shuffle_data(X, t)

        X = Data.flatten_data(X)

        t = Data.one_hot_encode(t)

        X_train, t_train, X_val, t_val, X_test, t_test = Data.split_data(
            X, t, train_val_test_split
        )

        X_train, X_val, X_test = Data.standardize_data(X_train, X_val, X_test)

        self.train_data = {"X": X_train, "t": t_train}
        self.val_data = {"X": X_val, "t": t_val}
        self.test_data = {"X": X_test, "t": t_test}

    @staticmethod
    def fetch_data(data_set='mnist_784'):
        mnist = fetch_openml(data_set, as_frame=False)
        X = mnist["data"].astype("float64")
        t = mnist["target"].astype("int")
        return (X, t)

    @staticmethod
    def shuffle_data(X, t):
        random_state = check_random_state(1)
        permutation = random_state.permutation(X.shape[0])
        X = X[permutation]
        t = t[permutation]
        return (X, t)

    @staticmethod
    def flatten_data(X_data):
        # flatten the image into a vector
        X_data = X_data.reshape((X_data.shape[0], -1))
        # add a column of '1' to X_data
        X_data = np.column_stack((X_data, [1] * X_data.shape[0]))
        return X_data

    @staticmethod
    def one_hot_encode(t_data, classes_num=10):
        # one-hot encode the classes
        t_data = np.identity(classes_num)[t_data]
        return t_data

    @staticmethod
    def split_data(X, t, train_val_test_split):
        val_size, test_size = Data.get_val_test_size(train_val_test_split)

        X_train, X_test, t_train, t_test = train_test_split(
            X, t, test_size=test_size
        )

        X_train, X_val, t_train, t_val = train_test_split(
            X_train, t_train, test_size=val_size
        )

        return (X_train, t_train, X_val, t_val, X_test, t_test)

    @staticmethod
    def standardize_data(X_train, X_val, X_test):
        # The next lines standardize the images
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        return (X_train, X_val, X_test)

    @staticmethod
    def get_val_test_size(train_val_test_split):
        return (
            train_val_test_split[1] / (train_val_test_split[0] + train_val_test_split[1]),
            train_val_test_split[2] / sum(train_val_test_split)
        )
