from helpers import *
from implementations import *

y_train = np.genfromtxt(
    "y_train_processed.csv", delimiter=" ", skip_header=0, usecols=0
)
x_train = np.genfromtxt("x_train_processed.csv", delimiter=" ", skip_header=0)
x_test = np.genfromtxt("x_test_processed.csv", delimiter=" ", skip_header=0)
ids = np.genfromtxt("test_ids.csv", delimiter=",")

x_test = x_test.T
for col in x_test:
    avg = 0
    card = 0
    for v in col:
        if not np.isnan(v):
            card += 1
            avg += v
    if card != 0:
        col = np.where(np.isnan(col), avg / card, col)
x_test = x_test.T

initial_w = np.zeros(x_train.shape[1])


def sigmoid(z):
    return np.exp(z) / (1 + np.exp(z))


def train_logistic_regression(X, y, learning_rate, num_iterations):
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)

    for epoch in range(num_iterations):
        model = np.dot(X, weights)
        predictions = sigmoid(model)
        gradient = (1 / num_samples) * np.dot(X.T, (predictions - y))
        weights -= learning_rate * gradient
        if (epoch % 100 == 0) or (epoch == num_iterations - 1):
            print(f"Epoch {epoch}/{num_iterations - 1}")

    return weights


def predict(X, weights):
    model = np.dot(X, weights)
    predictions = sigmoid(model)
    return (predictions >= 0.5).astype(int)


# Train the logistic regression model
learning_rate = 0.01
num_iterations = 1000
weights = train_logistic_regression(x_train, y_train, learning_rate, num_iterations)

# Make predictions using the trained model
y_pred = predict(x_test, weights)
y_pred = np.where(y_pred == 0, -1, y_pred)

create_csv_submission(ids, y_pred, "y_pred.csv")
