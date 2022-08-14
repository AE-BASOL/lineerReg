# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
df = pd.read_csv("Salary_Data.csv")
print(df.sample(3))
print(df.isna().sum())
print("\n", df.describe())

plt.scatter(df["YearsExperience"], df["Salary"]) #verinin grafikteki saçılmasını gösterir
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.show()

x = df['YearsExperience']
y = df['Salary']

# %%
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=1)
x_train = x_train.sort_index()
y_train = y_train.sort_index()
print(x_train.shape, y_train.shape)

# %%
from sklearn.linear_model import LinearRegression

X_train = pd.DataFrame(x_train)
X_test = pd.DataFrame(x_test)
Y_train = pd.DataFrame(y_train)
Y_test = pd.DataFrame(y_test)
# print("\n",X_train,"\n",Y_test)

lr = LinearRegression()
lr.fit(X_train, Y_train)

Y_pred = lr.predict(X_test)
print("\n", Y_pred, "\n", Y_test)

# %%
from sklearn.metrics import r2_score, mean_squared_error

r2s = r2_score(Y_test,Y_pred)
mse = mean_squared_error(Y_test,Y_pred)


print("\nmean squared error is: ", mse)

print("\ncofficient is:", lr.coef_)
print("\nthe intercept is: ", lr.intercept_)

# %%
m = [i for i in range (1, len(Y_test)+1)]
plt.plot(m, Y_test, color='r', linestyle='-')
plt.plot(m, Y_pred, color='b', linestyle='-')
plt.xlabel('Salary')
plt.ylabel("index")
plt.title("prediction")
plt.show()

print("\nr2score is: ", r2s)