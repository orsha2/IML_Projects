import matplotlib.pyplot as plt
import numpy as np
import hickle
import os


class Model:
    EPSILON = 1e-10

    def __init__(self, shape=(10, 785), learning_rate=0.001, desired_val_acc_diff=0.01):

        self.learning_rate = learning_rate
        self.iterations_num = 0

        self.desired_val_acc_diff = desired_val_acc_diff

        self.weights_mat = np.random.random(shape)

        self.history = {"train_loss": [], "train_accuracy": [],
                        "val_loss": [], "val_accuracy": []}

    def train(self, train_data, val_data):
        print("Model Training Started!\n")

        while self.get_val_acc_diff() > self.desired_val_acc_diff:
            self.run_iteration(train_data)
            self.iterations_num += 1

            t_loss, t_accuracy = self.evaluate(train_data)
            v_loss, v_accuracy = self.evaluate(val_data)

            self.update_history(t_loss, t_accuracy, v_loss, v_accuracy)

            print("#Iteration {}: train (loss, acc) = ({}, {:.2f}%), val (loss, acc) = ({}, {:.2f}%)\n".format(
                self.iterations_num, t_loss, t_accuracy, v_loss, v_accuracy))

        print("Model Training Finished!\n")

    def run_iteration(self, train_data):
        X = train_data["X"]
        t = train_data["t"]

        y = self.classify(X)

        gradient = X.T @ (y - t)
        self.weights_mat = self.weights_mat - self.learning_rate * gradient.T

    def classify(self, X_data):
        y = X_data @ (self.weights_mat).T

        y = np.exp(y - np.max(y, axis=1)[:, None])

        y = y / np.sum(y, axis=1)[:, None]

        return y

    def get_val_acc_diff(self, iter_num_to_evaluate=3):
        if(len(self.history["val_accuracy"]) < iter_num_to_evaluate):
            return 1

        evaluated_acc = np.array(self.history["val_accuracy"][-iter_num_to_evaluate:])

        return np.sum((evaluated_acc - np.mean(evaluated_acc))**2)

    def evaluate(self, data):
        # get loss for data
        y = self.classify(data["X"])

        loss = Model.calc_cross_entropy_loss(y, data["t"])

        # calc accuracy
        predicated_labels = np.argmax(y, axis=1)
        true_labels = np.argmax(data["t"], axis=1)
        accuracy = np.sum(predicated_labels == true_labels) / len(true_labels) * 100

        return loss, accuracy

    @staticmethod
    def calc_cross_entropy_loss(y, t):
        return -np.sum(t * np.log(np.maximum(y, Model.EPSILON)))

    def update_history(self, t_loss, t_accuracy, v_loss, v_accuracy):
        self.history["train_loss"].append(t_loss)
        self.history["train_accuracy"].append(t_accuracy)
        self.history["val_loss"].append(v_loss)
        self.history["val_accuracy"].append(v_accuracy)

    def plot_history(self, save_path=os.getcwd()):
        iterations_range = range(self.iterations_num)

        plt.figure(figsize=(10, 10))
        plt.subplot(1, 2, 1)

        plt.plot(iterations_range, self.history['train_loss'], label='Training Loss')
        plt.plot(iterations_range, self.history['val_loss'], label='Val Loss')

        plt.xlabel('Iteration Number')
        plt.ylabel('Training Loss')
        plt.title('Train and Val Loss')
        plt.grid()
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(iterations_range, self.history['train_accuracy'], label='Training Accuracy')
        plt.plot(iterations_range, self.history['val_accuracy'], label='Val Accuracy')

        plt.xlabel('Iteration Number')
        plt.ylabel('Training Accuracy')
        plt.title('Train and Val Accuracy')
        plt.grid()
        plt.legend()

        plt.show(block=False)

        if save_path is not None:
            plt.savefig(os.path.join(save_path, "Training_and_Validation.png"))

    def print_results(self, data):
        print('Model Results:')
        print('\titerations_num = {} '.format(self.iterations_num))
        print('\ttrain (loss, accuracy) = ({:0.3f}, {:0.3f}%)'.format(
            *self.evaluate(data.train_data)))
        print('\tval   (loss, accuracy) = ({:0.3f}, {:0.3f}%)'.format(
            *self.evaluate(data.val_data)))
        print('\ttest  (loss, accuracy) = ({:0.3f}, {:0.3f}%)'.format(
            *self.evaluate(data.test_data)))

    def save_model(self, save_path=os.getcwd()):
        if (save_path is None):
            return
        os.makedirs(save_path, exist_ok=True)
        self_list = [self.weights_mat, self.iterations_num, self.history]
        hickle.dump(self_list, os.path.join(save_path, "model_data.hkl"))

    @ staticmethod
    def load_model(load_path=os.getcwd()):
        new_model = Model()
        new_model.weights_mat, new_model.iterations_num, new_model.history = hickle.load(
            os.path.join(load_path, "model_data.hkl"))

        return new_model
