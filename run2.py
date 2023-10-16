import numpy as np
import csv

"""
x = np.eye(568238, 2)
for i in range(568238):
    x[i, 0] = i
print(x)
np.savetxt(
    "submissionTEST.csv",
    x,
    delimiter=",",
    header="Id, Prediction",
)"""
# Exception: Malformed header of the prediction CSV file. Missing : Prediction
"""
with open("submi", "w") as csvfile:
    fieldnames = ["Id", "Prediction"]
    writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
    writer.writeheader()
    for r1 in range(1, 568239):
        writer.writerow({"Id": int(r1), "Prediction": 0})"""
# Exception: All expected Ids were not present in the submission. Please cross check to ensure that the submitted file contains a row each for all the data points in the test set.


def load_data(path_dataset, sub_sample=True, add_outlier=False, train=True):
    """Load data and convert it to the metric system."""
    print("importing " + path_dataset)
    data = np.genfromtxt(path_dataset, delimiter=",", dtype=str, skip_header=1)
    print("import done")
    ids = data[:, 0]
    labels = data[:, 1]
    if train == True:
        labels[labels == "s"] = 1
        labels[labels == "b"] = -1
        labels = np.asarray(labels, dtype=float)
        print("changes done")
    data = np.delete(data, [0, 1], 1)
    data = np.asarray(data, dtype=np.float)
    print(path_dataset + "is now loaded.")
    return data, labels, ids


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


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


def create_csv_submission(ids, y_pred, name):
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


# Load the training data and labels
X_train, _, _ = load_data("dataset/x_train.csv", train=True)
_, y_train, _ = load_data("dataset/y_train.csv", train=True)

# Train the logistic regression model
learning_rate = 0.01
num_iterations = 1000
weights = train_logistic_regression(X_train, y_train, learning_rate, num_iterations)

# Load the test data
X_test, _, ids = load_data("x_test.csv", train=False)

# Make predictions using the trained model
y_pred = predict(X_test, weights)

# Create a CSV submission file with the predictions
create_csv_submission(ids, y_pred, "submission2.csv")
