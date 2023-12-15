import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("polyreg/Position_Salaries.csv")

x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,train_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x,y)

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 2)

x_poly=poly.fit_transform(x)

reg_2 = LinearRegression()

reg_2.fit(x_poly,y)

plt.scatter(x,y,color="red")

plt.plot(x,reg.predict(x),color="blue")

plt.show()

plt.scatter(x,y,color="red")

plt.plot(x,reg_2.predict(x_poly),color="blue")

plt.show()

reg.predict([[6.5]])

reg_2.predict(poly.fit_transform([[6.5]]))

