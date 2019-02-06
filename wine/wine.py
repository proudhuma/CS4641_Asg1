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
	
def prepareData(cross):
	wine = datasets.load_wine()
	trainX, testX, trainY, testY = train_test_split(wine.data, wine.target, test_size=cross)

	return trainX, trainY, testX, testY

def knn(trainX, trainY, testX, testY):
	print("----------- K Nearest Neighbor ------------")
	tScore = []
	tknn = []
	trainAcc = []
	for k in range(1,30,2):
		# run knn with different hyperpatameter k
		knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
		knn.fit(trainX, trainY)
		train_acc = knn.score(trainX, trainY)
		score = knn.score(testX, testY)
		tScore.append(score)
		trainAcc.append(train_acc)
		tknn.append(knn)
		print("k: %d, accuracy: %f" %(k, score))
	maxScore = max(tScore)
	index = tScore.index(maxScore)
	model = tknn[index]
	print("Highest train accuracy: %f with k: %d" % (maxScore, 2*index+1))
	accuracy = model.score(testX, testY)
	print("Test accuracy: %f" % (accuracy))
	x = range(1,30,2)
	plt.figure(figsize=(8,4))
	axes = plt.gca()
	# axes.set_ylim([0.0,1.0])
	val, = plt.plot(x,tScore,"b-",linewidth=1)
	tra, = plt.plot(x,trainAcc,"r-",linewidth=1)
	val.set_label("Validation")
	tra.set_label("Train")
	plt.xlabel("K")
	plt.ylabel("Accuracy") 
	plt.title("KNN with Different K")
	plt.legend()
	plt.savefig("knn_k.jpg")
	return model

		
def decisionTree(trainX, trainY, testX, testY):
	print("------------- Decision Tree ------------------")
	scores = []
	dts = []
	trainAcc = []
	# use information gain
	for max_leaf in range(3,30,3):
		clf = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=max_leaf)
		clf.fit(trainX, trainY)
		train_acc = clf.score(trainX, trainY)
		trainAcc.append(train_acc)
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
	x = range(3,30,3)
	plt.figure(figsize=(8,4))
	axes = plt.gca()
	# axes.set_ylim([0.0,1.0])
	val, = plt.plot(x,scores,"b-",linewidth=1)
	tra, = plt.plot(x,trainAcc,"r-",linewidth=1)
	val.set_label("Validation")
	tra.set_label("Train")
	plt.xlabel("Max Leaf")
	plt.ylabel("Accuracy") 
	plt.title("Descision Tree with Different Max Leaf")
	plt.legend()
	plt.savefig("dt_maxleaf.jpg")
	return clf

def boosting(trainX, trainY, testX, testY):
	# use ada boosting classifier
	print("------------- Ada Boosting ---------------------")
	scores = []
	trainAcc = []
	adas = []
	for n in range(10,100,10):
		adacls = AdaBoostClassifier(n_estimators=n)
		adacls.fit(trainX, trainY)
		train_acc = adacls.score(trainX, trainY)
		trainAcc.append(train_acc)
		score = adacls.score(testX, testY)
		scores.append(score)
		adas.append(adacls)
		print("estimator: %d, training accuracy: %f" %(n,score))
	maxScore = max(scores)
	index = scores.index(maxScore)
	ada = adas[index]	
	accuracy = ada.score(testX, testY)
	print("Ada Boosting estimator: %d, test accuracy: %f" % (10*(index+1), accuracy))
	x = range(10,100,10)
	plt.figure(figsize=(8,4))
	axes = plt.gca()
	# axes.set_ylim([0.0,1.0])
	val, = plt.plot(x,scores,"b-",linewidth=1)
	tra, = plt.plot(x,trainAcc,"r-",linewidth=1)
	val.set_label("Validation")
	tra.set_label("Train")
	plt.xlabel("Number of Estimator")
	plt.ylabel("Accuracy") 
	plt.title("Ada Boosting with Different Number of Estimator")
	plt.legend()
	plt.savefig("ada_num_estimator.jpg")
	return ada


def nn(trainX, trainY, testX, testY):
	print("------------- Neural Network ----------------")
	nnTrainX = trainX
	nnTrainY = trainY
	scores = []
	for x in range(5,27,2):
		nncls = MLPClassifier(hidden_layer_sizes=(30,x),alpha=0.00005,learning_rate_init=0.0001,early_stopping=True,validation_fraction=0.2)
		nncls.fit(nnTrainX, nnTrainY)
		accuracy = nncls.score(testX,testY)
		scores.append(accuracy)
	maxScore = max(scores)
	index = scores.index(maxScore)
	print("Neural Network accuracy: %f with %d layer" %(accuracy, index*2+5))
	x = range(5,27,2)
	plt.figure(figsize=(8,4))
	axes = plt.gca()
	axes.set_xlim([0,25])
	val, = plt.plot(x,scores,"b-",linewidth=1)
	val.set_label("Test")
	plt.xlabel("Number of Layer")
	plt.ylabel("Accuracy") 
	plt.title("Neural Network with Different Number of Layer")
	plt.legend()
	plt.savefig("nn_num_layer.jpg")
	return nncls

def svm(trainX, trainY, testX, testY):
	print("------------- Support Vector Machine ----------")

	# scale data to have mean 0 and var 1
	scaler = preprocessing.StandardScaler().fit(trainX)
	trainX_scaled = scaler.transform(trainX)
	testX_scaled = scaler.transform(testX)

	# four kernel functions
	kernels = ['poly','linear','rbf','sigmoid']
	scores = []
	trainAcc = []
	clfs = []
	for kernel in kernels:
		clf = SVC(gamma='auto', kernel=kernel, cache_size=2000)
		clf.fit(trainX_scaled, trainY)
		train_acc = clf.score(trainX_scaled, trainY)
		trainAcc.append(train_acc)
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
	plt.figure(figsize=(8,4))
	axes = plt.gca()
	# axes.set_ylim([0.0,1.0])
	x=[1,2,3,4]
	val, = plt.plot(x,scores,"b-",linewidth=1)
	tra, = plt.plot(x,trainAcc,"r-",linewidth=1)
	plt.xticks(x, kernels)
	val.set_label("Validation")
	tra.set_label("Train")
	plt.xlabel("Kernel")
	plt.ylabel("Accuracy") 
	plt.title("Support Vector Machine with Different Kernels")
	plt.legend()
	plt.savefig("svm_kernel.jpg")
	return svm

if __name__ == "__main__":
	# use 20% train data as validation data
	trainX, trainY, testX, testY = prepareData(0.2)
	knn = knn(trainX, trainY, testX, testY)
	svm = svm(trainX, trainY, testX, testY)
	dt = decisionTree(trainX, trainY, testX, testY)
	boosting = boosting(trainX, trainY, testX, testY)
	nn = nn(trainX, trainY, testX, testY)
