
# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for sigmoid, loss, gradient and predict and perform operations. 

## Program:

/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sanjeev D
RegisterNumber: 212223040185
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
dataset
dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes



dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes

dataset

X=dataset.iloc[:, :-1].values
Y=dataset.iloc[:, -1].values

Y

theta=np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
    
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)

def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
    
y_pred = predict(theta,X)

accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:", accuracy)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)


## Output:
### Read the file and display
![WhatsApp Image 2024-05-09 at 01 00 12_ac7fdb76](https://github.com/Sanjuwu21/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146498969/486cc14e-096d-4810-98c6-38d5ee4b3f2a)


### Categorizing columns
![WhatsApp Image 2024-05-09 at 01 00 21_10851fee](https://github.com/Sanjuwu21/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146498969/d32218a8-b769-46f9-afa0-73b46cf8f430)


### Labelling columns and displaying dataset
![WhatsApp Image 2024-05-09 at 01 00 27_fe268639](https://github.com/Sanjuwu21/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146498969/293d4b34-a5f4-49af-b0a0-8a7aeef44058)


### Display dependent variable
![WhatsApp Image 2024-05-09 at 01 00 30_6d921f2c](https://github.com/Sanjuwu21/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146498969/716926f6-f697-45ec-968b-09da209474fa)

### Printing accuracy
![WhatsApp Image 2024-05-09 at 01 00 35_af66a55d](https://github.com/Sanjuwu21/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146498969/807ef2f9-f0e3-4982-a80a-1b11c4940fa8)


### Printing Y
![WhatsApp Image 2024-05-09 at 01 00 38_6457e281](https://github.com/Sanjuwu21/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146498969/95ef67d6-564e-4e06-b50e-88ce9c4b275b)



### Printing y_prednew
![WhatsApp Image 2024-05-09 at 01 00 43_9344818e](https://github.com/Sanjuwu21/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/146498969/21285292-d327-4d15-a1dd-11fb4aa97a13)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
