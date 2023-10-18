import pandas as pd

# creating a data frame
from io import StringIO


"""import csv

text = open("2015.csv", "r")

# join() method combines all contents of
# csvfile.csv and formed as a string
text = "".join([i for i in text])

# search and replace the contents
text = text.replace("b'0", "")
text = text.replace("b'", "")
text = text.replace("'", "")

# output.csv is the output file opened in write mode
x = open("output.csv", "w")

# all the replaced text is written in the output.csv file
x.writelines(text)
x.close()
"""

df = pd.read_csv("output.csv")
df2 = pd.read_csv("dataset/x_test.csv")
df = df.fillna("")
print(df2.shape)
print(df2)

c1 = list(df.columns)
c2 = list(df2.columns)
difference = list(set(c1) - set(c2))
print(difference)
x_2015 = df.drop(columns=difference)
print(x_2015.shape)


c1 = list(x_2015.columns)
c2 = list(df2.columns)
difference = list(set(c2) - set(c1))
print(difference)
x_test = df2.drop(columns=difference)


print("x_2015 shape: " + str(x_2015.shape))
print("x_test shape: " + str(x_test.shape))

x_2015 = x_2015.astype(object)
x_test = x_test.astype(object)
x = pd.merge(x_2015, x_test)

print(x)
