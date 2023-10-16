from helpers import *
from implementations import *

y = np.genfromtxt("y_train_processed.csv", delimiter=" ", skip_header=0, usecols=0)
x = np.genfromtxt("x_train_processed.csv", delimiter=" ", skip_header=0)

split = int(np.floor(3 * y.shape[0] / 4))
x_train = x[:split, :]
y_train = y[:split]
x_test = x[split:, :]
y_test = y[split:]

initial_w = np.zeros(x_train.shape[1])

loss, w = logreg_gd(y_train, x_train, initial_w, 500, 0.01)

print("Train loss: ", loss, "Test loss: ", logreg_loss(y_test, x_test, w))

np.savetxt(
    "submission.csv",
    np.dot(x_test, w),
    header="_MICHD",
)
