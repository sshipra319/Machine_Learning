# -*- coding: utf-8 -*-
"""
@author: Shipra
"""
from sklearn.datasets import load_iris
import numpy as np

def linear_regression(x_train, y_train, x_test, y_test):
    
    #Calculate Beta = (x(transpose). x)(inverse) (x(transpose). y)
    
    left = np.dot(np.transpose(x_train), x_train)
    left_inverse = np.linalg.inv(left)
    right = np.dot(np.transpose(x_train), y_train)
    beta = np.matmul(left_inverse, right) 
    
    # Using calculated beta value will predict the test dataset
    
    predict = np.matmul(x_test, beta)
    predict = abs(np.rint(predict)).astype(int)
    print("\n Actual Labels - ", y_test)
    print("Predicted Labels - ", predict)
    
    #Accuracy of trained model
    
    count = 0
    for i in range (len(predict)):
        if(predict[i] == y_test[i]):
            count += 1
    accuracy = count/len(predict)
    print("Accuracy = ", accuracy)
    return accuracy


def cross_valid (k, x_iris, y_iris):
    indices = np.random.permutation(len(x_iris))   #For shuffling data
    n = len(iris.data)
    len_k = n // k
    accuracy_list = []

   #Splitting the data in training and testing data for cross-validation

    for i in range(k):
        start = i * len_k
        end = ((i + 1) * len_k)
        x_test_iris = x_iris[indices[start:end]]
        y_test_iris = y_iris[indices[start:end]]
        x_train_iris = x_iris[indices[[x for x in indices if x not in indices[start:end]]]]
        y_train_iris = y_iris[indices[[x for x in indices if x not in indices[start:end]]]]
        accuracy = linear_regression(x_train_iris, y_train_iris, x_test_iris, y_test_iris)
        accuracy_list.append(accuracy)
        
    avg_accuracy = sum(accuracy_list)/len(accuracy_list)
    print ("\n Average accuracy for", k, "fold cross-validation : ", avg_accuracy)



if __name__ == '__main__':
    iris = load_iris()
    k_fold = int(input("Please input k for the k-fold cross validation for linear regression: "))
    cross_valid(k_fold, iris.data, iris.target)


