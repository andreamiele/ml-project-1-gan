import numpy as np

x = np.eye(568238, 2)
for i in range(568238):
    x[i, 0] = i
print(x)
np.savetxt(
    "submissionTEST.csv",
    x,
    delimiter=",",
    header="Id, Prediction",
)
# Exception: Malformed header of the prediction CSV file. Missing : Prediction
