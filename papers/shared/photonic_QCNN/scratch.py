import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from .bars_and_stripes import generate_bars_and_stripes


def get_bas(random_state):
    np.random.seed(random_state)
    bas_x_all, bas_y_all = generate_bars_and_stripes(600, 4, 4, 0.5)
    bas_x_all = bas_x_all.squeeze(1)

    bas_x_train, bas_x_test, bas_y_train, bas_y_test = train_test_split(
        bas_x_all, bas_y_all, test_size=200, train_size=400, random_state=random_state
    )

    bas_x_train = np.reshape(bas_x_train, (400, 4, 4))
    bas_x_test = np.reshape(bas_x_test, (200, 4, 4))

    bas_y_train = bas_y_train == 1
    bas_y_test = bas_y_test == 1
    bas_y_train = bas_y_train.astype(float)
    bas_y_test = bas_y_test.astype(float)

    return bas_x_train, bas_x_test, bas_y_train, bas_y_test


def get_custom_bas(random_state):
    custom_bas_x, custom_bas_y = generate_bars_and_stripes(600, 4, 4, 0)

    mask = custom_bas_x == 1
    rng = np.random.default_rng(seed=random_state)
    noise = rng.normal(0, 0.5, size=custom_bas_x.shape)
    noisy_bas_x = custom_bas_x + noise * mask

    c_bas_x_train, c_bas_x_test, c_bas_y_train, c_bas_y_test = train_test_split(
        noisy_bas_x,
        custom_bas_y,
        test_size=200,
        train_size=400,
        random_state=random_state,
    )
    c_bas_x_train = np.reshape(c_bas_x_train, (400, 4, 4))
    c_bas_x_test = np.reshape(c_bas_x_test, (200, 4, 4))

    c_bas_y_train = c_bas_y_train == 1
    c_bas_y_test = c_bas_y_test == 1
    c_bas_y_train = c_bas_y_train.astype(float)
    c_bas_y_test = c_bas_y_test.astype(float)

    return c_bas_x_train, c_bas_x_test, c_bas_y_train, c_bas_y_test


def get_mnist(random_state, class_list=(0, 1)):
    mnist_x, mnist_y = load_digits(return_X_y=True)

    mask = np.isin(mnist_y, class_list)
    mnist_x = mnist_x[mask]
    mnist_y = mnist_y[mask]

    mnist_x_train, mnist_x_test, mnist_y_train, mnist_y_test = train_test_split(
        mnist_x, mnist_y, test_size=200, random_state=random_state
    )

    mnist_x_train = mnist_x_train.reshape(-1, 8, 8)
    mnist_x_test = mnist_x_test.reshape(-1, 8, 8)

    return mnist_x_train, mnist_x_test, mnist_y_train, mnist_y_test
