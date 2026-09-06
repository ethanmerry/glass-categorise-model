import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

def splitTrainTest(x, y):
    #Uses student id as seed for random permutation
    np.random.seed(39587266)
    indices = np.random.permutation(len(x))

    #Splits data into 80/20 train test data
    size = int(0.8 * len(x))

    trainIndices = indices[:size]
    testIndices = indices[size:]

    xTrain = x[trainIndices]
    xTest = x[testIndices]

    yTrain = y[trainIndices]
    yTest = y[testIndices]

    return xTrain, xTest, yTrain, yTest

#Calculates euclidean distance between train and test
def calculateDistance(test, train):
    squared_distance = np.sum((test - train) ** 2)
    return np.sqrt(squared_distance)

#Finds most common label
def mostCommon(labels):
    uniqueLabels, labelCount = np.unique(labels, return_counts=True)
    mostCommon = np.argmax(labelCount)
    return uniqueLabels[mostCommon]

def knnPredictSingle(x, y, test, k):
    #Calculate distance between test instance and all train instances
    distances = []
    for trainRow in x:
        distance = calculateDistance(test, trainRow)
        distances.append(distance)

    sorted = np.argsort(distances)

    #Nearest k labels
    nearestIndices = sorted[:k]
    nearestLabels = y[nearestIndices]

    return mostCommon(nearestLabels)

def knnPredict(xTrain, yTrain, xTest, k):
    predictions = []

    for testRow in xTest:
        predicted = knnPredictSingle(xTrain, yTrain, testRow, k)
        predictions.append(predicted)

    return np.array(predictions)

#Initalises nodes for the tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, label=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.label = label

#Calculates gini impurity
def gini(y):
    _, count = np.unique(y, return_counts=True)
    probability = count / len(y)
    return 1.0 - np.sum(probability ** 2)

#Finds threshold with minimum gini impurity
def bestSplit(x, y):
    bestGini = float('inf')
    bestFeature = None
    bestThreshold = None

    samples, features = x.shape

    #Finds feature and threshold with lowest gini 
    for feature in range(0, features):
        thresholds = np.unique(x[:, feature])

        for threshold in thresholds:
            left = x[:, feature] <= threshold
            right = x[:, feature] > threshold

            yLeft = y[left]
            yRight = y[right]

            if len(yLeft) == 0:
                continue

            if len(yRight) == 0:
                continue

            #Calculates gini index
            leftIndex = (len(yLeft) / samples) * gini(yLeft)
            rightIndex = (len(yRight) / samples) * gini(yRight)

            index = (leftIndex + rightIndex)

            if index < bestGini:
                bestGini = index
                bestFeature = feature
                bestThreshold = threshold

    return bestFeature, bestThreshold

#Builds tree recursively by splitting dataset on best threshold
def buildTree(x, y, maxDepth, depth=0):    
    labelsNum = len(np.unique(y))
    if labelsNum == 1:
        return Node(label=y[0])

    if depth >= maxDepth:
        return Node(label=mostCommon(y))

    feature, threshold = bestSplit(x, y)

    if feature is None:
        return Node(label=mostCommon(y))

    #Splits data based on threshold
    left = x[:, feature] <= threshold
    right = x[:, feature] > threshold

    leftChild = buildTree(x[left], y[left], maxDepth, depth + 1)
    rightChild = buildTree(x[right], y[right], maxDepth, depth + 1)

    return Node(feature, threshold, leftChild, rightChild)


#Recursively traverse the tree to find predicted values
def dtPredictSingle(node, test):
    if node.label is not None:
        return node.label
    
    if test[node.feature] <= node.threshold:
        return dtPredictSingle(node.left, test)
    else:
        return dtPredictSingle(node.right, test)


def dtPredict(xTrain, yTrain, xTest, maxDepth):
    tree = buildTree(xTrain, yTrain, maxDepth)

    predictions = []
    for testRow in xTest:
        predicted = dtPredictSingle(tree, testRow)
        predictions.append(predicted)

    return np.array(predictions)

def fit(x, y):
    labels = np.unique(yTrain)

    priors = {}
    means = {}
    variances = {}

    for label in labels:
        xLabel = x[y == label]

        priors[label] = len(xLabel) / len(x)

        #Finds mean and variance of each column
        means[label] = np.mean(xLabel, axis=0)
        variances[label] = np.var(xLabel, axis=0)

    return labels, means, variances, priors

def logLikelihood(x, mean, variance):
    variance = np.maximum(variance, 1e-9)

    normaliser = np.log(2 * np.pi * variance)
    scaledError = (x - mean) ** 2 / variance

    return np.sum(-0.5 * (normaliser + scaledError))

def nbPredictSingle(x, labels, means, variances, priors):
    posteriors = []
    for label in labels:
        logPrior = np.log(priors[label])

        likelihood = logLikelihood(x, means[label], variances[label])

        posterior = logPrior + likelihood
        posteriors.append(posterior)

    return labels[np.argmax(posteriors)]

def nbPredict(xTrain, yTrain, xTest):
    classes, means, variances, priors = fit(xTrain, yTrain)

    predictions = []
    for testRow in xTest:
        predicted = nbPredictSingle(testRow, classes, means, variances, priors)
        predictions.append(predicted)

    return np.array(predictions)

#Calculates time taken, accuracy and produces confusion matrices
def evaluate(test, prediction, start, end, type):
    test = np.array(test).flatten()
    prediction = np.array(prediction).flatten()

    accuracy = (np.sum(test == prediction) / len(test)) * 100

    print(f"Accuracy:  {accuracy:.2f}%")
    print(f"Time Taken: {end - start:.6f}s")

    classes = np.unique(np.concatenate((test, prediction)))

    mapLabel = {}
    for i, label in enumerate(classes):
        mapLabel[label] = i

    confusionMatrix = np.zeros((len(classes), len(classes)), dtype=int)
    for actual, predicted in zip(test, prediction):
        confusionMatrix[mapLabel[actual], mapLabel[predicted]] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(confusionMatrix, cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_title(type)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    for i in range(0,len(classes)):
        for j in range(0,len(classes)):
            ax.text(j, i, confusionMatrix[i, j], ha="center", va="center", color="black")

    plt.tight_layout()
    plt.show()

def knnEvaluate(xTrain, yTrain, xTest, yTest):
    print("K-nearest Neighbour")
    
    for k in range(1,5):
        start = time.time()
        predictions = knnPredict(xTrain, yTrain, xTest, k)
        end = time.time()

        print(f"k={k}")
        evaluate(yTest, predictions, start, end, f"KNN (k={k})")

def dtEvaluate(xTrain, yTrain, xTest, yTest):
    print("\nDecision Trees")
    
    for maxDepth in range(1,5):
        start = time.time()
        predictions = dtPredict(xTrain, yTrain, xTest, maxDepth)
        end = time.time()

        print(f"max depth={maxDepth}")
        evaluate(yTest, predictions, start, end, f"Decision Tree (depth={maxDepth})")

def nbEvaluate(xTrain, yTrain, xTest, yTest):
    print("\nNaive Bayes")
    
    start = time.time()
    predictions = nbPredict(xTrain, yTrain, xTest)
    end = time.time()

    evaluate(yTest, predictions, start, end, "Naive Bayes")

column_names = ['id', 'ri', 'na', 'mg', 'al', 'si', 'k', 'ca', 'ba', 'fe', 'glass_type']

df = pd.read_csv('glass+identification/glass.data', names=column_names)

df = df.drop(columns=['id'])

x = df.drop(columns=['glass_type'])
y = df['glass_type']

x = x.to_numpy(dtype=float)
y = y.to_numpy()

xTrain, xTest, yTrain, yTest = splitTrainTest(x, y)

#implements StandardScaler fitted on training data
mean = xTrain.mean()
std = xTrain.std()

std = np.where(std == 0, 1, std)

xTrain = (xTrain - mean) / std
xTest = (xTest - mean) / std

knnEvaluate(xTrain, yTrain, xTest, yTest)

dtEvaluate(xTrain, yTrain, xTest, yTest)

nbEvaluate(xTrain, yTrain, xTest, yTest)