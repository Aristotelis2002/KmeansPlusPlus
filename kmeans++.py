from math import sqrt
import sys
import random
import functools
import numpy as np
import matplotlib.pyplot as plt


def readFile(localPathToFile):
    f = open(localPathToFile, 'r')
    listOfCordinates = []
    for line in f.readlines():
        listOfCordinates.append([float(i) for i in line.split()])
    return listOfCordinates


def plotPoints(points):
    for point in points:
        plt.scatter(point[0], point[1], marker='x', color='red')
    plt.show()


def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n)


def plotRes(clusters, centroids, k):
    colors = plt.cm.rainbow(np.linspace(0, 1, k))
    for i, sublist in enumerate(clusters):
        x_vals = [point[0] for point in sublist]
        y_vals = [point[1] for point in sublist]
        plt.scatter(x_vals, y_vals, color=colors[i], label=f"Set {i+1}")
        plt.scatter(centroids[i][0], centroids[i][1],
                    color=colors[i], marker='^', s=150)
    plt.show()


def euclideanDistance(point1, point2):
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    distance = sqrt(distance)
    return distance


def funcD(pointX, centroids):
    return min(map(lambda mu: sqrt((pointX[0] - mu[0])**2 + (pointX[1] - mu[1])**2), centroids))


def initiateCentroids(points, k):
    centroids = []
    centroids.append(random.choice(points))
    for _ in range(1, k):
        sumD = sum([funcD(y, centroids)**2 for y in points])
        prob = [funcD(x, centroids)**2 / sumD for x in points]
        cumulative_prob = np.cumsum(prob)
        rand_val = np.random.rand()
        for j, p in enumerate(cumulative_prob):
            if rand_val < p:
                centroids.append(points[j])
                break
    return centroids


def calcRSS(clusters):
    rss = 0
    for cluster in clusters:
        muI = list(map(lambda x: x / len(cluster),
                       functools.reduce(lambda x,
                                        y: [x[0] + y[0], x[1] + y[1]], cluster, [0, 0])))
        rss += sum([sqrt((x[0] - muI[0])**2 + (x[1] - muI[1])**2)
                   for x in cluster])
    return rss


def buildClusters(points, centroids, k):
    clusters = [[] for _ in range(k)]
    for point in points:
        closestCentroid = list(map(lambda mu: euclideanDistance(point, mu), centroids)).index(
            min(map(lambda mu: euclideanDistance(point, mu), centroids)))
        clusters[closestCentroid].append(point)
    return clusters


def sumListOfPoints(points):
    return functools.reduce(lambda x, y: [x[0] + y[0], x[1] + y[1]], points, [0, 0])


def updateCentroids(clusters):
    centroids = list(map(lambda cluster: list(
        map(lambda x: x / len(cluster), sumListOfPoints(cluster))), clusters))
    return centroids


def solve(points, k):
    rssOld = 0
    rss = 1
    centroids = initiateCentroids(points, k)
    while (rss != rssOld):
        rssOld = rss
        clusters = buildClusters(points, centroids, k)
        centroids = updateCentroids(clusters)
        rss = calcRSS(clusters)
    return clusters, centroids, rss


points = readFile(sys.argv[1])
k = int(sys.argv[2])
clustersFinal, centroidsFinal, rssFinal = solve(points, k)
if(len(sys.argv) < 4):
    restarts = 0 
else:
    restarts = int(sys.argv[3])
for _ in range(restarts):
    clusters, centroids, rss = solve(points, k)
    if (rss < rssFinal):
        clustersFinal = clusters
        centroidsFinal = centroids
        rssFinal = rss
colors = plt.cm.rainbow(np.linspace(0, 1, k))
plotRes(clustersFinal, centroidsFinal, k)