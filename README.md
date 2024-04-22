# EX 07: Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## DATE:

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:

1. Import the required libraries.

2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4. Apply label encoder to the non-numerical column inoreder to convert into numerical values.
5. Determine training and test data set.
6. Apply decision tree regression on to the dataframe.
7. Get the values of Mean square error, r2 and data prediction.

## Program:
```python
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: VENKATANATHAN P R
RegisterNumber:  212223240173
*/

import pandas as pd
data=pd.read_csv("Salary.csv")
print("\nThe first five data of the Salary.csv:")
print(data.head())
print("\nThe DataFrame:")
print(data.info())
print("\nCount the number of NaN values:")
print(data.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
print("\nThe first five data for Position:")
print(data.head())

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
print("\nThe Mean Squared Error:")
print(mse)

r2=metrics.r2_score(y_test,y_pred)
print("\nCoefficient of Determination:")
print(r2)

print("\nPrediction Value:")
print(dt.predict([[5,6]]))

```

## Output:

### The first five data of the Salary.csv:

![alt text](<Screenshot 2024-04-22 201208.png>)

### The DataFrame:

![alt text](<Screenshot 2024-04-22 201215.png>)

### Count the number of NaN values:

![alt text](<Screenshot 2024-04-22 201221.png>)

### The first five data for Position:

![alt text](<Screenshot 2024-04-22 201227.png>)

### The Mean Squared Error:

![alt text](<Screenshot 2024-04-22 201242.png>)

### Coefficieny of Determination:

![alt text](<Screenshot 2024-04-22 201247.png>)

### Prediction Value:

![alt text](<Screenshot 2024-04-22 201822.png>)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
