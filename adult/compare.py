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
	
def prepareData(cross):
	trainset = loadData("adult.data")
	testset = loadData("adult.test")

	# seperate data for cross validation
	trainX, validationX, trainY, validationY = train_test_split(trainset[['age','workclass','fnlwgt','education',
		'education_num','marital_status','occupation','relationship','race','sex',
		'capital_gain','capital_loss','hours_per_week','native_country']], trainset['result'], test_size=cross)
	testX = testset[['age','workclass','fnlwgt','education',
		'education_num','marital_status','occupation','relationship','race','sex',
		'capital_gain','capital_loss','hours_per_week','native_country']]
	testY = testset["result"]

	return trainX, validationX, trainY, validationY, testX, testY

if __name__ == "__main__":
	percentage = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
	knn_score = []
	dt_score = []
	ada_score = []
	svm_score = []
	nn_score = []
	for p in percentage:
		trainX, validationX, trainY, validationY, testX, testY = prepareData(p)
		# knn
		knn = KNeighborsClassifier(n_neighbors=19, n_jobs=-1)
		knn.fit(trainX, trainY)
		knn_score.append(knn.score(testX, testY))
		# decision tree
		dt = DecisionTreeClassifier(criterion='entropy', max_leaf_nodes=21)
		dt.fit(trainX, trainY)
		dt_score.append(dt.score(testX, testY))
		# boosting
		ada = AdaBoostClassifier(n_estimators=50)
		ada.fit(trainX, trainY)
		ada_score.append(ada.score(testX, testY))
		# svm
		scaler = preprocessing.StandardScaler().fit(trainX)
		trainX_scaled = scaler.transform(trainX)
		validationX_scaled = scaler.transform(validationX)
		testX_scaled = scaler.transform(testX)
		svm = SVC(gamma='auto', kernel='rbf', cache_size=2000)
		svm.fit(trainX_scaled, trainY)
		svm_score.append(svm.score(testX, testY))
		#nn
		nn = MLPClassifier(hidden_layer_sizes=(30,19),alpha=0.00005,learning_rate_init=0.0001,early_stopping=True,validation_fraction=p)
		nnTrainX = trainX.append(validationX)
		nnTrainY = trainY.append(validationY)
		nn.fit(nnTrainX, nnTrainY)
		nn_score.append(nn.score(testX, testY))
	plt.figure(figsize=(8,4))
	axes = plt.gca()
	# axes.set_ylim([0,1])
	plt.plot(percentage,knn_score,"b-",linewidth=1, label='KNN')
	plt.plot(percentage,dt_score,"r-",linewidth=1, label='Decision Tree')
	plt.plot(percentage,ada_score,"g-",linewidth=1, label='Ada')
	plt.plot(percentage,svm_score,"y-",linewidth=1, label='SVM')
	plt.plot(percentage,nn_score,"p-",linewidth=1, label='NN')
	plt.xlabel("Cross Validation Percentage")
	plt.ylabel("Accuracy") 
	plt.title("Accuracy with Different Cross Validation Percentage")
	plt.legend()
	plt.savefig("compare.jpg")