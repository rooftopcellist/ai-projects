'''
k-means Clustering
Author: Christian M. Adams


Implementation:
1.	Initialize
  a.	Acquire data (put into 2d array)
  b.	Choose # of clusters
  c.	Create cluster center starting points (random)
  d.	Make initial cluster assignments
2.	While (before time runs out or means become fixed):
  a.	Calculate k-mean, update cluster center location
  b.	Update cluster assignments
  c.	Find distances from each cluster point to each of the data points.
        The data point gets allocated to whichever centroid has the smaller distance.
  d.	Calculate k-mean by finding the average of all of the distance for all data points allocated to the cluster center, then update the centroid.
Performance - can be improved by finding the optimal # of clusters and their starting locations.


'''


from random import randint
import math
import matplotlib.pyplot as plt

ENGLISH = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
                   'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
                   's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def readData(path):
    dataset = []

    with open(path, 'r') as f:
        for line in f:
            new_line = line.split(",")
            dataset.append([float(new_line[0]), float(new_line[1])])
    return dataset
dataset = readData('data.csv')

class Cluster:

    def __init__(self, point, class_id):
        self.point = point
        self.class_id = class_id

    def set_point(self, point):
        self.point = point
        self.old_point = None

    def get_point(self):
        return self.point

    def get_class(self):
        return self.class_id


def create_clusters(k):
    clusters = []
    for i in range(k):
        x = randint(0,1000000000)/1000000000.
        y = randint(0,1000000000)/1000000000.
        point = x,y
        clusters.append(Cluster(point, ENGLISH[i]))
    return clusters


def eclidian_diff(old_point, new_point):
	x_diff = abs(old_point[0] - new_point[0])
	y_diff = abs(old_point[1] - new_point[1])
	return math.sqrt(x_diff*x_diff + y_diff*y_diff)

def cluster_assignment(dataset, clusters):
    classified_array = []
    index = 0
    for point in dataset:
        lowest_point = 100000000
        lowest_cluster = None
        for cluster in clusters:
            temp_dist = eclidian_diff(point, cluster.point)
            temp_cluster = cluster
            if temp_dist < lowest_point:
                lowest_point = temp_dist
                lowest_cluster = temp_cluster
        classified_array.append([point, lowest_cluster.class_id])
        index += 1
    return classified_array

def compute_mean(classified_dataset, cluster):
    total_sum = 0.
    total = len(classified_dataset)
    x_sum = 0
    y_sum = 0

    for point in classified_dataset:
        if point[1] == cluster.class_id:
            x_sum += point[0][0]
            y_sum += point[0][1]

    x_avg = x_sum/total
    y_avg = y_sum/total
    cluster.old_point = cluster.point
    new_centroid = x_avg, y_avg
    cluster.point = new_centroid

    return cluster.point

def update_clusters(classified_dataset, clusters):
    tries = 0
    clusters_not_set = True
    write_to_csv(classified_dataset, "cluster")



    while (tries < 1000) and clusters_not_set:
        cluster_history = []
        classification_history = []
        for cluster in clusters:
            centroid_point = compute_mean(classified_dataset, cluster)
            classified_dataset = cluster_assignment(dataset, clusters)
            classification_history.append(classified_dataset)
            cluster_history.append(cluster.point)
            if cluster.point == cluster.old_point:
                clusters_not_set = False
                break

        tries += 1
        write_to_csv(classified_dataset, "cluster")
        # print cluster_history
        # print classification_history
        # plot_points(classified_dataset, cluster)
    print "\nTries: ",tries
    return classification_history, cluster_history



#Write to CSV for later analysis in Excel
def write_to_csv(array, cluster_id):
    with open('file-'+cluster_id+'.csv','a') as file:
        file.write('________________________________\n')
        for i in range(27):
            for value in array[i]:
                value = str(value).strip('[]')
                new_value = value + ','
                file.write(new_value)
            file.write('\n')

def plot_points(points, cluster_history):

    count = 0
    # print cluster_history
    #
    # plt.scatter(cluster_history[0][0], cluster_history[0][1], color = 'blue', alpha = 1)
    # plt.scatter(cluster_history[1][0], cluster_history[1][1], color = 'red', alpha = 1)

    for point in points:
        if point[1][0] == 'a':
            plt.scatter(point[0][0], point[0][1], color = 'blue', alpha = .2)
        else:
            plt.scatter(point[0][0], point[0][1], color = 'red', alpha = .2)

    plt.xlabel("x-axis")
    plt.ylabel("y-axis")
    plt.title("K-means Data")
    plt.show()


def main():
    with open('file-cluster.csv','wb') as file:
        file.write(' ')
        file.close()
    clusters = create_clusters(2)
    classified_dataset = cluster_assignment(dataset, clusters)
    classified_dataset, cluster_history = update_clusters(classified_dataset, clusters)
    print cluster_history
    for i in range(len(classified_dataset)):
        plot_points(classified_dataset[i], cluster_history[i])

main()
