'''
k-NN Classification Algorithm
Author: Christian Adams

Given a set of pre-classified data, this algorithm


1. Read in data
2. Classify Data Points artificially
3. Add new datapoint
4. using Euclidian Geometry find nearest NN
5. Assign classication

'''

from random import randint
import math

#Reads in article using each word as input
def readData(path):
    x_coor = []
    y_coor = []
    with open(path, 'r') as f:
        for line in f:
            new_line = line.split(",")
            x_coor.append(float(new_line[0]))
            y_coor.append(float(new_line[1]))
    return x_coor, y_coor
readData('data.csv')


#classifies data as part of A or B
def classify():
    dataset = []
    x,y = readData('data.csv')
    for i in range(len(x)):
        if (x[i] <= .4):
            dataset.append([x[i],y[i],'A'])
        if (x[i] >= .6):
            dataset.append([x[i],y[i],'B'])
        else:
            if randint(0,1) == 0:
                dataset.append([x[i],y[i],'A'])
            else:
                dataset.append([x[i],y[i],'B'])
#show dataset cleanly w/ classifications
    # for line in dataset:
    #     print line
    return dataset


def generate_data():
    x = randint(0,1000000000)/1000000000.
    y = randint(0,1000000000)/1000000000.
    return x, y

def eclidian_diff(old_point, new_point):
	x_diff = abs(old_point[0] - new_point[0])
	y_diff = abs(old_point[1] - new_point[1])
	return math.sqrt(x_diff*x_diff + y_diff*y_diff)


# def find_nn(dataset, new_point, k):
#     temp_dataset = dataset
#     temp_dist = 10000
#     lowest_points = []
#     low_threshold = 0
#     temp_point = None
#     while k > 0:
#         for point in dataset:
#             t_point = point[0],point[1]
#             e_diff = eclidian_diff(t_point, new_point)
#             if e_diff <= temp_dist:
#                 temp_dist = e_diff
#                 temp_point = point
#                 low_threshold = e_diff
#             else:
#                 temp_dataset.append(point)
#         lowest_points.append(temp_point)
#         k -= 1
#     # print temp_dist
#     # print point
#     print lowest_points
#     return point


def find_nn(dataset, new_point):
    temp_dist = 10000
    for point in dataset:
        t_point = point[0],point[1]
        e_diff = eclidian_diff(t_point, new_point)
        if e_diff <= temp_dist:
            temp_dist = e_diff
            low_threshold = e_diff
            low_point = point
    return low_point


def find_knn(dataset, new_point, k):
    lowest_points = []
    temp_dataset = dataset
    while k > 0:
        current_point = find_nn(temp_dataset, new_point)
        lowest_points.append(current_point)
        temp_dataset.remove(current_point)
        k -= 1
    return lowest_points

def classify_point(lowest_points):
    classifications = {}
    count = 0
    classifications['A'] = 0
    classifications['B'] = 0
    for point in lowest_points:
        classifications[point[2]] += 1
    if classifications['A'] > classifications['B']:
        return 'A'
    else:
        return 'B'


def main():
    k = 1
    dataset = classify()
    new_point = generate_data()
    #k should be an odd value
    lowest_points = find_knn(dataset, new_point, 5)
    print lowest_points
    classify_point(lowest_points)

main()
