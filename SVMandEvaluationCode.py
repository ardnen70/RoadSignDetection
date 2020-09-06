import cv2         
import numpy as np
from matplotlib import pyplot as plt
import math as m
import operator
import glob
import pickle
import pprint
from sklearn import svm
from sklearn.metrics import confusion_matrix



def getConfusionAndPrintPercentage(results,y_true):
	img_area = ['PRIORITY_ROAD','PASS_EITHER_SIDE','SPEED_LIMIT','GIVE_WAY','STOP','PEDESTRIAN_CROSSING','NO_PARKING','OTHER']
	print img_area
	print "Corresponding correct predictions in %:" 
	#y_true = y_true + 1
	maxindex = np.argmax(results,axis = 1) + 1
	maxvals = np.amax(results,axis = 1)
	for i in range(len(maxvals)):
		if (maxvals[i] < 0):
			maxindex[i] = 0
			y_true[i] = 0
	confusion = confusion_matrix(y_true, maxindex)
	percentages = (np.diag(confusion.astype(float))/1901) * 100
	print percentages[1:12] 
	print confusion[1:12,1:12]


featuresTrain = np.load('featuresTrain.npy')
featuresTest = np.load('featuresTest.npy')
trainDecisions = np.load('featuresTrainDecisions.npy')
testDecisions = np.load('featuresTestDecisions.npy')

featuresTrain = np.asarray(featuresTrain)
featuresTest = np.asarray(featuresTest)
trainDecisions = np.asarray(trainDecisions[0])
ll = len(trainDecisions);
testDecisions = np.asarray(testDecisions[0])


st1 = ['PRIORITY_ROAD','PASS_EITHER_SIDE','SPEED_LIMIT','GIVE_WAY','STOP','PEDESTRIAN_CROSSING','NO_PARKING','OTHER']
decisionTrainMat = np.empty((ll,11))
for i in range(len(st1)):
	for j in range(ll):
		if (int(trainDecisions[j]) == i+1):
			decisionTrainMat[j,i] = 1
		else:
			decisionTrainMat[j,i] = -1
decisionTrainMat = np.transpose(decisionTrainMat)
lenn = len(testDecisions)
results = np.zeros((lenn,11))
savebjs = np.zeros(8)
for j in range(len(st1)):
	Clf = svm.LinearSVC(C = 5) #assigning different coefficients to different classifiers
	Clf.fit(featuresTrain,decisionTrainMat[j,:])
	weights = Clf.coef_ #get weights
	np.save('weights'+str(j+1), weights)
	bj = Clf.intercept_[0]
	savebjs[j] = bj
	w_mag = np.sqrt(np.sum(weights * weights))
	dj = (1/w_mag) * (np.dot(featuresTest,np.transpose(weights)) + bj)
	results[0:lenn,j] = dj[0:lenn,0]

getConfusionAndPrintPercentage(results,testDecisions)
np.save('BJS', savebjs)