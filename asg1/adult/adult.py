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

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def loadData(filename):
	# load adult data and drop missing information
	data = pd.read_csv(filename, names=['age','workclass','fnlwgt','education',
		'education_num','marital_status','occupation','relationship','race','sex',
		'capital_gain','capital_loss','hours_per_week','native_country','result'],
		engine="python", skipinitialspace=True, na_values=['?'])
	data = data.dropna()

	# encode the labels to numbers
	eData = data.copy()
	for column in eData.columns:
		if eData.dtypes[column] == 'object':
			le = preprocessing.LabelEncoder()
			le.fit(eData[column])
			eData[column] = le.transform(eData[column])
	return eData
	
def prepareData():
	trainset = loadData("adult.data")
	testset = loadData("adult.test")

	# seperate data for cross validation
	trainX, validationX, trainY, validationY = train_test_split(trainset[['age','workclass','fnlwgt','education',
		'education_num','marital_status','occupation','relationship','race','sex',
		'capital_gain','capital_loss','hours_per_week','native_country']], trainset['result'], test_size=0.2)
	testX = testset[['age','workclass','fnlwgt','education',
		'education_num','marital_status','occupation','relationship','race','sex',
		'capital_gain','capital_loss','hours_per_week','native_country']]
	testY = testset["result"]

	return trainX, validationX, trainY, validationY, testX, testY

def knn(trainX, validationX, trainY, validationY, testX, testY):
	print("----------- K Nearest Neighbor ------------")
	tScore = []
	tknn = []
	for k in range(1,30,2):
		# run knn with different hyperpatameter k
		knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
		knn.fit(trainX, trainY)
		score = knn.score(validationX, validationY)
		tScore.append(score)
		tknn.append(knn)
		print("k: %d, accuracy: %f" %(k, score))
	maxScore = max(tScore)
	index = tScore.index(maxScore)
	model = tknn[index]
	print("Highest train accuracy: %f with k: %d" % (maxScore, 2*index+1))
	accuracy = model.score(testX, testY)
	print("Test accuracy: %f" % (accuracy))
		
def decisionTree(trainX, validationX, trainY, validationY, testX, testY):
	print("------------- Decision Tree ------------------")
	scores = []
	dts = []
	# use information gain
	for max_leaf in range(3,30,3):
		clf = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=max_leaf)
		clf.fit(trainX, trainY)
		score = clf.score(validationX, validationY)
		scores.append(score)
		dts.append(clf)
		print("Max # of leaves: %d, train accuracy: %f" % (max_leaf, score))
	maxScore = max(scores)
	index = scores.index(maxScore)	
	dt = dts[index]
	print("Highest train accuracy: %f with max leaves: %d" % (maxScore, 3*(index+1)))
	accuracy = clf.score(testX, testY)
	print("Test accuracy: %f" %(accuracy))

def boosting(trainX, validationX, trainY, validationY, testX, testY):
	# use ada boosting classifier
	print("------------- Ada Boosting ---------------------")

	scores = []
	adas = []
	for n in range(10,100,10):
		adacls = AdaBoostClassifier(n_estimators=n)
		adacls.fit(trainX, trainY)
		score = adacls.score(validationX, validationY)
		scores.append(score)
		adas.append(adacls)
		print("estimator: %d, training accuracy: %f" %(n,score))
	maxScore = max(scores)
	index = scores.index(maxScore)
	ada = adas[index]	
	accuracy = ada.score(testX, testY)
	print("Ada Boosting estimator: %d, test accuracy: %f" % (10*(index+1), accuracy))


def nn(trainX, validationX, trainY, validationY, testX, testY):
	print("------------- Neural Network ----------------")
	nnTrainX = trainX.append(validationX)
	nnTrainY = trainY.append(validationY)
	nncls = MLPClassifier(hidden_layer_sizes=(20,5),alpha=0.00005,learning_rate_init=0.0001,early_stopping=True,validation_fraction=0.2)
	nncls.fit(nnTrainX, nnTrainY)
	accuracy = nncls.score(testX,testY)
	print("Neural Network accuracy: %f" %(accuracy))

def svm(trainX, validationX, trainY, validationY, testX, testY):
	print("------------- Support Vector Machine ----------")

	# scale data to have mean 0 and var 1
	scaler = preprocessing.StandardScaler().fit(trainX)
	trainX_scaled = scaler.transform(trainX)
	validationX_scaled = scaler.transform(validationX)
	testX_scaled = scaler.transform(testX)

	# four kernel functions
	kernels = ['poly','linear','rbf','sigmoid']
	scores = []
	clfs = []
	for kernel in kernels:
		clf = SVC(gamma='auto', kernel=kernel, cache_size=2000)
		clf.fit(trainX_scaled, trainY)
		score = clf.score(validationX_scaled, validationY)
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
	trainX, validationX, trainY, validationY, testX, testY = prepareData()
	knn(trainX, validationX, trainY, validationY, testX, testY)
	svm(trainX, validationX, trainY, validationY, testX, testY)
	decisionTree(trainX, validationX, trainY, validationY, testX, testY)
	boosting(trainX, validationX, trainY, validationY, testX, testY)
	nn(trainX, validationX, trainY, validationY, testX, testY)