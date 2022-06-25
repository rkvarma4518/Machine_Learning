

import pandas as pd
from sklearn.linear_model import LinearRegression

"""### *Load Dataset*"""

dataset = pd.read_csv('data.csv')

"""### *Finding & Removing NA values from our Features X*"""

dataset.columns[dataset.isna().any()]

dataset.hours = dataset.hours.fillna(dataset.hours.mean())

"""### *Segregate Dataset into Input X & Output Y*"""

X = dataset.iloc[:, :-1].values
#print(X)

Y = dataset.iloc[:, -1].values
#print(Y)

"""### *Training Dataset using Linear Regression*"""

model = LinearRegression()
model.fit(X,Y)

"""### *Predicted Price for Land sq.Feet of custom values*"""

hr = float(input("How many hours did you study \n"))
age = float(input("What is your age \n"))
internet = float(input("How many hours you use internet \n"))

inputs=[[hr,age,internet]]
output = float(model.predict(inputs))

marks = str(round(output, 2))
print("You will get approximately",marks,"%")
