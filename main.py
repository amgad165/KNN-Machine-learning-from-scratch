from mlxtend.data import loadlocal_mnist
from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import resize
from skimage.feature import hog
import math
import operator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, y_train = loadlocal_mnist(
    images_path='train-images.idx3-ubyte',
    labels_path='train-labels.idx1-ubyte')

X_test, y_test = loadlocal_mnist(
    images_path='t10k-images.idx3-ubyte',
    labels_path='t10k-labels.idx1-ubyte')



# def euclideanDistance(instance1, instance2):
#     distance = 0
#     for x in range(len(instance1)):
#         distance += pow((instance1[x] - instance2[x]), 2)
#     return math.sqrt(distance)

def euclideanDistance(image1, image2):
    distance = 0
    # for i in range(len(image1)):
    #     for j in range(len(image1[0])):
    #         # distance += math.pow((image1[i][j]-image2[i][j]), 2)
    distance = np.sum((image1 - image2) ** 2)
    distance = np.sqrt(distance)
    return distance


def getKNeighbors(trainingSet, testInstance, labelInstance, k):
    distances = []
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x])
        distances.append((labelInstance[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def knn_Predictions(train_images, test_images, y_train, k):
    predictions = []
    for img in range(len(test_images)):
        neighbors = getKNeighbors(train_images, test_images[img], y_train[:len(train_images)], k)
        predictions.append(getResponse(neighbors))
    predictions = np.array(predictions)
    return predictions


# trainSet = [[2, 2, 2], [4, 4, 4], [3, 3, 3], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8]]
# labelInstance= ['a', 'a', 'a', 'b', 'b','a','a']
# testInstance = [2, 2, 4]
# k = 5
# neighbors = getKNeighbors(trainSet, testInstance, labelInstance, k)
# print(neighbors)
# print(getResponse(neighbors))

train_images = []
test_images = []

for i in range(30):
    print(i)
    train_image = X_train[i].reshape(28, 28)
    resized_img = resize(train_image, (128 * 4, 64 * 4))

    fd, hog_train_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True)
    train_images.append(hog_train_image)

    test_image = X_test[i].reshape(28, 28)
    resized_img = resize(test_image, (128 * 4, 64 * 4))
    fd, hog_test_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
                             cells_per_block=(2, 2), visualize=True)
    test_images.append(hog_test_image)

train_images = np.array(train_images)
test_images = np.array(test_images)

kVals = range(1, 30, 2)
accuracies = []
# loop over various values of `k` for the k-Nearest Neighbor classifier
for k in range(1, 30, 2):
    # train the k-Nearest Neighbor classifier with the current value of `k`
    y_predict = knn_Predictions(train_images, test_images, y_train[:len(train_images)], k)
    # evaluate the model and update the accuracies list
    score = accuracy_score(y_test[:len(y_predict)], y_predict)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))
    accuracies.append(score)

# find the value of k that has the largest accuracy
i = int(np.argmax(accuracies))
print("k=%d achieved highest accuracy of %.2f%% on test data" % (kVals[i], accuracies[i] * 100))


