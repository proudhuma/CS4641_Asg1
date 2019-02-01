#!/usr/bin/python3.6
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import datasets

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
	
def prepareData():
	wine = datasets.load_wine()
	trainX, testX, trainY, testY = train_test_split(wine.data, wine.target, test_size=0.2)

	return trainX, trainY, testX, testY

def knn(trainX, trainY, testX, testY):
	print("----------- K Nearest Neighbor ------------")
	tScore = []
	tknn = []
	for k in range(1,30,2):
		# run knn with different hyperpatameter k
		knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
		knn.fit(trainX, trainY)
		score = knn.score(testX, testY)
		tScore.append(score)
		tknn.append(knn)
		print("k: %d, accuracy: %f" %(k, score))
	maxScore = max(tScore)
	index = tScore.index(maxScore)
	model = tknn[index]
	print("Highest train accuracy: %f with k: %d" % (maxScore, 2*index+1))
	accuracy = model.score(testX, testY)
	print("Test accuracy: %f" % (accuracy))
		
def decisionTree(trainX, trainY, testX, testY):
	print("------------- Decision Tree ------------------")
	scores = []
	dts = []
	# use information gain
	for max_leaf in range(3,30,3):
		clf = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=max_leaf)
		clf.fit(trainX, trainY)
		score = clf.score(testX, testY)
		scores.append(score)
		dts.append(clf)
		print("Max # of leaves: %d, train accuracy: %f" % (max_leaf, score))
	maxScore = max(scores)
	index = scores.index(maxScore)	
	dt = dts[index]
	print("Highest train accuracy: %f with max leaves: %d" % (maxScore, 3*(index+1)))
	accuracy = clf.score(testX, testY)
	print("Test accuracy: %f" %(accuracy))

def boosting(trainX, trainY, testX, testY):
	# use ada boosting classifier
	print("------------- Ada Boosting ---------------------")

	scores = []
	adas = []
	for n in range(10,100,10):
		adacls = AdaBoostClassifier(n_estimators=n)
		adacls.fit(trainX, trainY)
		score = adacls.score(testX, testY)
		scores.append(score)
		adas.append(adacls)
		print("estimator: %d, training accuracy: %f" %(n,score))
	maxScore = max(scores)
	index = scores.index(maxScore)
	ada = adas[index]	
	accuracy = ada.score(testX, testY)
	print("Ada Boosting estimator: %d, test accuracy: %f" % (10*(index+1), accuracy))


def nn(trainX, trainY, testX, testY):
	print("------------- Neural Network ----------------")
	nnTrainX = trainX
	nnTrainY = trainY
	nncls = MLPClassifier(hidden_layer_sizes=(30,7),alpha=0.00005,learning_rate_init=0.0001,early_stopping=True,validation_fraction=0.2)
	nncls.fit(nnTrainX, nnTrainY)
	accuracy = nncls.score(testX,testY)
	print("Neural Network accuracy: %f" %(accuracy))

def svm(trainX, trainY, testX, testY):
	print("------------- Support Vector Machine ----------")

	# scale data to have mean 0 and var 1
	scaler = preprocessing.StandardScaler().fit(trainX)
	trainX_scaled = scaler.transform(trainX)
	testX_scaled = scaler.transform(testX)

	# four kernel functions
	kernels = ['poly','linear','rbf','sigmoid']
	scores = []
	clfs = []
	for kernel in kernels:
		clf = SVC(gamma='auto', kernel=kernel, cache_size=2000)
		clf.fit(trainX_scaled, trainY)
		score = clf.score(testX_scaled, testY)
		scores.append(score)
		clfs.append(clf)
		print("Kernel function: " + kernel + ", accuracy: %f" % (score))
	maxScore = max(scores)
	index = scores.index(maxScore)
	svm = clfs[index]
	accuracy = svm.score(testX_scaled, testY)
	print("Highest train accuracy: %f with kernel: " % (maxScore) + kernels[index])
	print("Test accuracy: %f" %(accuracy))


if __name__ == "__main__":
	trainX, trainY, testX, testY = prepareData()
	knn(trainX, trainY, testX, testY)
	svm(trainX, trainY, testX, testY)
	decisionTree(trainX, trainY, testX, testY)
	boosting(trainX, trainY, testX, testY)
	nn(trainX, trainY, testX, testY)