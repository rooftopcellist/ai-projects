'''
Bayesian Classification
Author: Christian M. Adams
--------------------------
1. Handle Data: Load the data from CSV file and split it into training and test datasets.
2. Summarize Data: summarize the properties in the training dataset so that we can calculate probabilities and make predictions.
3. Make a Prediction: Use the summaries of the dataset to generate a single prediction.
4. Make Predictions: Generate predictions given a test dataset and a summarized training dataset.
5. Evaluate Accuracy: Evaluate the accuracy of predictions made for a test dataset as the percentage correct out of all predictions made.
6. Tie it Together: Use all of the code elements to present a complete and standalone implementation of the Naive Bayes algorithm.



Compute discriminant for Class
 a. calculate mean vector
 b. make class covariance matrix for Class
 c. take the inverse of the covariance matrix
 d. compute determinant of covariance matrix


 arguments:
 1. apriori probability
 2. point x
 3. mean of dataset, or pass the whole dataset
 4.


POST ANALYSIS
split data into test set
2. run test, prepare statistical report
3. graph out boundary line - compute boundary equation by setting g1(x) = g2(x)



'''

import csv
import random
import math
import numpy as np
from numpy.linalg import inv
import time
import matplotlib.pyplot as plt

def readData(path):
    dataset = []

    with open(path, 'r') as f:
        for line in f:
            new_line = line.rstrip('\r\n')
            new_line = new_line.split(',')
            item = [float(new_line[0]), float(new_line[1])]
            dataset.append(item)
    return dataset


def splitData(dataset):
    training_data = []
    testing_data = []
    for item in dataset:
        if random.randint(1,3) == 3:
            testing_data.append(item)
        else: training_data.append(item)
    return training_data, testing_data



def probabilities(datasetA, datasetB, datasetC):
    total_len = len(datasetA) + len(datasetB) + len(datasetB)
    probA = len(datasetA) / float(total_len)
    probB = len(datasetB) / float(total_len)
    probC = len(datasetC) / float(total_len)
    return probA, probB, probC



def compute_discriminant(dataset, point, priori):

    #Making Covariance Matrix for Class
    x_total = 0
    y_total = 0
    for item in dataset:
        x_total += item[0]
        y_total += item[1]
    x_avg = x_total/len(dataset)
    y_avg = y_total/len(dataset)
    mean_vector = np.array([[x_avg],[y_avg]])
    # print "Mean Vector: ",mean_vector
    matrix_array = []
    for item in dataset:
        n_point = [item[0] - x_avg, item[1] - y_avg]
        np_point = np.array([[n_point[0]], [n_point[1]]])
        trans_point = np_point.transpose()
        covariance_matrix = np_point * trans_point
        matrix_array.append(covariance_matrix)

    #Sums all covariance matrices
    c_matrix_sum = np.array([[0.,0.],[0.,0.]])
    for item in matrix_array:
        c_matrix_sum += item

    #divide by the number of points in dataset
    c_matrix_sum = (1./len(dataset)) * c_matrix_sum
    # print "Covariance Matrix:\n", c_matrix_sum

    #Take inverse of covariance matrix
    inv_co_matrix = inv(c_matrix_sum)
    # print "\nInverse of Covariance Matrix\n", inv_co_matrix

    #calculate determinant of covariance matrix (ad-bc) (not of inverse matrix)
    determinant = c_matrix_sum[0][0]*c_matrix_sum[1][1] - c_matrix_sum[0][1]*c_matrix_sum[1][0]
    # print "Determinant: ", determinant
    # print "Priori Probability: ", priori

    new_point = np.array([[point[0]], [point[1]]])
    #this is the transposition of (x - u)
    mean_diff = new_point - mean_vector
    mean_diff_transposed = mean_diff.transpose()
    temp = np.dot(mean_diff_transposed, inv_co_matrix)
    temp_product = np.dot(temp, mean_diff)[0][0]

    return (-1/2) * (temp_product) - (1/2) * math.log(determinant) + math.log(priori)

def classify(datasetA, datasetB, datasetC, point):
    probA, probB, probC = probabilities(datasetA, datasetB, datasetC)
    dscm_A = compute_discriminant(datasetA, point, probA)
    dscm_B = compute_discriminant(datasetB, point, probB)
    dscm_C = compute_discriminant(datasetC, point, probC)
    if dscm_A == max(dscm_A, dscm_B, dscm_C):
        return 'classA'
    elif dscm_B == max(dscm_A, dscm_B, dscm_C):
        return 'classB'
    elif dscm_C == max(dscm_A, dscm_B, dscm_C):
        return 'classC'

#computes points in boundary line between two given data sets
def graph_boundary(dsA, dsB, probA, probB):
    boundary_line = []
    minimum_x = min(min(dsA[0]), min(dsB[0])) - 4
    minimum_y = min(min(dsA[1]), min(dsB[1])) - 4
    maximum_x = max(max(dsA[0]), max(dsB[0])) + 4
    maximum_y = max(max(dsA[1]), max(dsB[1])) + 4

    for i in range(int(40*minimum_x), int(40*maximum_x)):
        for j in range(int(40*minimum_y), int(40*maximum_y)):
            point = [i/40., j/40.]
            discriminant_A = compute_discriminant(dsA, point, probA)
            discriminant_B = compute_discriminant(dsB, point, probB)
            if abs(discriminant_A - discriminant_B) < .2:
                boundary_line.append(point)

    return boundary_line


def graph_region(dsA, dsB, dsC):
    region_A = []
    region_B = []
    region_C = []
    minimum_x = min(min(dsA[0]), min(dsB[0]), min(dsC[0])) - 4
    minimum_y = min(min(dsA[1]), min(dsB[1]), min(dsC[0])) - 4
    maximum_x = max(max(dsA[0]), max(dsB[0]), min(dsC[0])) + 4
    maximum_y = max(max(dsA[1]), max(dsB[1]), min(dsC[0])) + 4

    for i in range(int(5*minimum_x), int(5*maximum_x)):
        for j in range(int(5*minimum_y), int(5*maximum_y)):
            point = [i/5., j/5.]
            classification = classify(dsA, dsB, dsC, point)
            if classification == 'classA':
                region_A.append(point)
            if classification == 'classB':
                region_B.append(point)
            if classification == 'classC':
                region_C.append(point)

    return region_A, region_B, region_C



def main():
    # Starts Timer
    start_time = time.time()
    datasetA = readData('dataA.csv')
    datasetB = readData('dataB.csv')
    datasetC = readData('dataC.csv')

    #Hash out to use full data set and skip tests
    datasetA, testsetA = splitData(datasetA)
    datasetB, testsetB = splitData(datasetB)
    datasetC, testsetC = splitData(datasetC)


    probA, probB, probC = probabilities(datasetA, datasetB, datasetC)




#Calls boundary line function to graph boundary lines
    # boundary_line_ab = graph_boundary(datasetA, datasetB, probA, probB)
    # boundary_line_bc = graph_boundary(datasetB, datasetC, probB, probC)
    # boundary_line_ac = graph_boundary(datasetA, datasetC, probA, probC)

    print "Discriminant Computation for A: ", compute_discriminant(datasetA, [2.01132, 1.613742], probA)
    print "Discriminant Computation for B: ", compute_discriminant(datasetB, [2.01132, 1.613742], probB)
    print "Discriminant Computation for C: ", compute_discriminant(datasetC, [2.01132, 1.613742], probC)

    print "Classification of point [2.01132, 1.613742]: ", classify(datasetA, datasetB, datasetC, [2.01132, 1.613742])

    print("--- %s seconds ---" % (time.time() - start_time))

#TESTING
    #classifies test set based on training sets
    total_A = 0
    total_B = 0
    total_C = 0
    for item in testsetA:
        if classify(datasetA, datasetB, datasetC, item) == 'classA':
            total_A += 1
    for item in testsetB:
        if classify(datasetA, datasetB, datasetC, item) == 'classB':
            total_B += 1
    for item in testsetC:
        if classify(datasetA, datasetB, datasetC, item) == 'classC':
            total_C += 1

    print "Class A:\n  Percent Correct", float(total_A) / len(testsetA) * 1.
    print "Class B:\n  Percent Correct", float(total_B) / len(testsetB) * 1.
    print "Class C:\n  Percent Correct", float(total_C) / len(testsetC) * 1.


    print "-------------------------------------------------------------"


    for item in datasetA:
        plt.scatter(item[0], item[1], color = 'blue')
    for item in datasetB:
        plt.scatter(item[0], item[1], color = 'red')
    for item in datasetC:
        plt.scatter(item[0], item[1], color = 'green')

# #Shades in class prediction regions on graph
#     regions = graph_region(datasetA, datasetB, datasetC)
#     for item in regions[0]:
#         plt.scatter(item[0], item[1], color = 'blue', alpha = .1)
#     for item in regions[1]:
#         plt.scatter(item[0], item[1], color = 'red', alpha = .1)
#     for item in regions[2]:
#         plt.scatter(item[0], item[1], color = 'green', alpha = .1)


# #Graphs boundary line from a-->b, b-->c, and a-->c in black, yellow, and orange respectively
#     for item in boundary_line_ab:
#         plt.scatter(item[0], item[1], color = 'black', marker = '.')
#     for item in boundary_line_bc:
#         plt.scatter(item[0], item[1], color = 'purple', marker = '.')
#     for item in boundary_line_ac:
#         plt.scatter(item[0], item[1], color = 'orange', marker = '.')

    plt.show()

    print("--- %s seconds ---" % (time.time() - start_time))





main()
