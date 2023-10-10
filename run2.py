import numpy as np
import csv

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

with open("submi", "w") as csvfile:
    fieldnames = ["Id", "Prediction"]
    writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
    writer.writeheader()
    for r1 in range(1, 568239):
        writer.writerow({"Id": int(r1), "Prediction": 0})
# Exception: All expected Ids were not present in the submission. Please cross check to ensure that the submitted file contains a row each for all the data points in the test set.
