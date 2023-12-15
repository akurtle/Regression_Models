import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("SVR/Position_Salaries.csv")

x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

y = y.reshape(len(y),1)



from sklearn.preprocessing import StandardScaler

sc= StandardScaler()

sc_y = StandardScaler()

x = sc.fit_transform(x)


y = sc_y.fit_transform(y)


from sklearn.svm import SVR

reg = SVR(kernel = "rbf")


reg.fit(x,y)

sc_y.inverse_transform(reg.predict(sc.transform([[6.5]])).reshape(-1,1))

plt.scatter(sc.inverse_transform(x),sc_y.inverse_transform(y),color="red")

plt.plot(sc.inverse_transform(x),sc_y.inverse_transform(reg.predict(x).reshape(-1,1)),color="blue")

plt.show()

X_grid= np.arange(min(sc.inverse_transform(x)),max(sc.inverse_transform(x)),0.1)

X_grid= X_grid.reshape((len(X_grid),1))

plt.scatter(sc.inverse_transform(x),sc_y.inverse_transform(y),color="red")

plt.plot(X_grid,sc_y.inverse_transform(reg.predict(sc.transform(X_grid)).reshape(-1,1)),color="blue")

plt.show()


