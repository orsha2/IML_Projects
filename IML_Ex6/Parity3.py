import numpy as np
import matplotlib.pyplot as plt


class Xor3:

    def __init__(self, size_of_hidden_layer=3, eta=2, running_number=100, iteration_number=2000):
        self.x = None
        self.weight_x_z = np.random.normal(0, 1, (size_of_hidden_layer, 4))
        self.weight_z_y = np.random.normal(0, 1, size_of_hidden_layer + 1)
        self.z = None
        self.y = None
        self.t = None
        self.mse = []
        self.size_of_hidden_layer = size_of_hidden_layer
        self.eta = eta
        self.running_number = running_number
        self.iteration_number = iteration_number

    def run_iteration(self):
        self.load_database()
        for run_number in range(self.running_number):
            for iteration_number in range(self.iteration_number):

                print(self.size_of_hidden_layer, 'cells', 'run_number', run_number, 'iteration_number',
                      iteration_number)

                self.forward_scan()

                self.mse.append(self.calc_mse())

                delta_output, delta_inner = self.backward_scan()

                self.weight_z_y = self.batch_gradient_descent(self.weight_z_y, delta_output, self.z)
                self.weight_x_z = self.batch_gradient_descent(self.weight_x_z, delta_inner, self.x.T)

    # TODO - improve function
    def backward_scan(self):
        delta_out = self.y * (1 - self.y) * (self.y - self.t)
        y_inner = self.z[1:, :]
        delta_hidden = (y_inner * (1 - y_inner)) * self.weight_z_y[1:].reshape((-1, 1)) * delta_out
        return delta_out, delta_hidden

    def load_database(self):
        self.x = np.array([np.array([1, x, k, z]) for x in [0, 1] for k in [0, 1] for z in [0, 1]])
        self.t = []
        for row in self.x:
            xor = np.logical_xor(np.logical_xor(row[0], row[1]), row[2])
            self.t.append(int(xor))

    def forward_scan(self):
        alpha_i = self.weight_x_z.dot(self.x.T)
        self.z = np.vstack((np.ones(8), self.logistic_sigmoid(alpha_i)))
        self.y = self.logistic_sigmoid(self.weight_z_y.dot(self.z))

    def calc_mse(self):
        return (1 / 8) * np.sum(np.square(self.y - self.t))

    def batch_gradient_descent(self, w_previous, delta, zi):
        return w_previous - self.eta * delta.dot(zi.T)

    def plot(self):
        plt.figure(figsize=(10, 10))
        plt.plot(self.mse, label=str(self.size_of_hidden_layer)+'cells')
        plt.xlabel('Iteration')
        plt.ylabel('MSE')
        plt.title('MSE (Hidden layer include '+str(self.size_of_hidden_layer)+' cells )')
        plt.show(block=False)
        plt.savefig(str(self.size_of_hidden_layer)+'cells.png')

    @staticmethod
    def logistic_sigmoid(x):
        return 1 / (1 + np.exp(-x))
