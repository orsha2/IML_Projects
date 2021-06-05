
"""

Michael Zhitomirsky 321962714
Or Shahar           208712471

"""

from Data import Data
from Model import Model


TRAIN_VAL_TEST_SPLIT = (60, 20, 20)


def main():

    data = Data()
    data.load_data(TRAIN_VAL_TEST_SPLIT)

    model = Model()

    model.train(data.train_data, data.val_data)
    model.plot_history()
    model.print_results(data)

    model.save_model()


if __name__ == "__main__":
    main()
