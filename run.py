from helpers import *
from implementations import *

_, input_data, ids = load_csv_data("./dataset/x_train.csv", sub_sample=False)
y, _, _ = load_csv_data("./dataset/y_train.csv", sub_sample=False)

print(y, "\n", input_data)

gamma = 0.1
initial_w = np.zeros(input_data.shape[1])
max_iters = 500

w, l = mean_squared_error_gd(y, input_data, initial_w, max_iters, gamma)

create_csv_submission(ids, np.dot(input_data, w), "submission.csv")