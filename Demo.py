import cv2         
import numpy as np
from matplotlib import pyplot as plt
import math as m
import image_plot_utilities
import operator
import glob
import image_plot_utilities
import random

w1 = np.load('weights1.npy')
w2 = np.load('weights2.npy')
w3 = np.load('weights3.npy')
w4 = np.load('weights4.npy')
w5 = np.load('weights5.npy')
w6 = np.load('weights6.npy')
w7 = np.load('weights7.npy')
w8 = np.load('weights8.npy')
ll = len(w1)
weights = (w1,w2,w3,w4,w5,w6,w7,w8)
BJS = np.load('BJS.npy')

def FilterHSV(Oimg,HSVim,Channel):
	'''Filters the HSV image based on the channel: Red = 1, Blue = 2 and Yellow = 3
	as signs always comprise of colored components with these three colors and returns 
	the original image with only the surviving pixels'''
	if (Channel == 1):
		# define range of red color in HSV
		lower = np.array([150,55,1])
		upper = np.array([180,255,175])
	if (Channel == 2):
		# define range of blue color in HSV
		lower = np.array([110,130,50])
		upper = np.array([120,215,255])
	if (Channel == 3):
		# define range of yellow color in HSV
		lower = np.array([1,100,100])
		upper = np.array([60,225,255])
	# Threshold the HSV image to get only blue colors
	mask = cv2.inRange(HSVim, lower, upper)

	# Bitwise-AND mask and original image
	HSV_filter = cv2.bitwise_and(Oimg, Oimg, mask= mask)
	return HSV_filter

def ApplyFilterAndBinarize(image):
	''' Converts the returned image from FilterHSV to grayscale,
		Then applies thresholding based on Otsu's method.'''
	F_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	th,F_img = cv2.threshold(F_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	kernel = np.ones((5,5),np.uint8)
	#after_open = cv2.morphologyEx(F_img,cv2.MORPH_OPEN,kernel)
	img3 = cv2.morphologyEx(F_img,cv2.MORPH_CLOSE,kernel)
	final_img = cv2.GaussianBlur(img3,(9,9),0)
	th,final_img = cv2.threshold(final_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return final_img

def find_contours(img):
	contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	return contours,hierarchy

def find_maxContour(contours,im_thresh):
	cols,rows = im_thresh.shape
	img_area = 200 #float(cols * rows)/96
	cnt_no = []
	for i in range(len(contours)-1):
		cnt = contours[i]
		area2 = cv2.contourArea(cnt)
		if (area2 > img_area):
			cnt_no.append(i)
	return cnt_no

def computeThreeDHistogram(img1,t):
	'''Computes a 3-D histogram based on given image and t values'''
	histogram = cv2.calcHist([img1],[0,1,2],None,[t,t,t],[0,256,0, 256,0,256])
	final_hist = np.reshape(histogram, (1,np.product(histogram.shape)))[0]
	return final_hist	

def getSegmentedBlocks(img,bw,bh,t):
	'''Extracts the blocks as given by the parameters from the image and calls the 
	   Histogram function to calculate the 3-d histogram for each block and concat-
	   anates them to produce a feature vector'''
	sh = img.shape
	ttt = t*t*t
	W, H = sh[1] ,sh[0]
	dw, dh = float(W)/(bw+1), float(H)/(bh + 1)
	feature = np.empty((bw*bh,ttt))
	count = 0
	for i in range (bh):
		for j in range (bw):
			a,b,c,d = i*dh,i*dh+2*dh,j*dw,j*dw + 2*dw
			image_block = img[a:b,c:d]
			ret_hist = computeThreeDHistogram(image_block,t) #t = 3 for x3 and x4
			feature[count,0:ttt] = ret_hist
			count = count + 1
	final_hist = np.reshape(feature, (1,np.product(feature.shape)))[0]
	return final_hist

def getBordertoBoxfeatures(edges):
	imgg = edges.copy()
	kernel = np.ones((3,3),np.uint8)
	bg = cv2.morphologyEx(imgg, cv2.MORPH_CLOSE,kernel);
	ss1, ss2 = bg.shape[0], bg.shape[1]
	D1 = np.empty((1,ss1*2))
	count = 0
	for i in range(2):
		for j in range(20):
			if (i == 0):
				arre = np.nonzero(bg[2*j,:])
			else:
				arre = np.nonzero(bg[:,2*j])
			arr = arre[0]
			ll = len(arr)
			if (ll < 1):
				first = 40
				last = 40
			else:
				first = arr[0]
				last = arr[ll-1]
			m = i*20
			D1[0,j+m] = first
			if (last == 40):
				D1[0,j+m+40] = last
			else:
				D1[0,j+m+40] = 40 - last
	flip1 = D1[0,0:20]
	flip2 = D1[0,60:80]
	flip3,flip4 = flip1[::-1],flip2[::-1]
	D1[0,0:20] = flip3
	D1[0,60:80] = flip4
	x = np.arange(0, 80, 1);
	y = D1[0,:]
	plt.bar(x,y)
	plt.show
	return D1

def classifyDetection(feature_vect):
	st1 = ['PRIORITY_ROAD','PASS_EITHER_SIDE','SPEED_LIMIT','GIVE_WAY','STOP','PEDESTRIAN_CROSSING','NO_PARKING','OTHER']
	result = np.zeros(8)
	for i in range(len(st1)):
		wt = weights[i]
		bj = BJS[i]
		w_mag = np.sqrt(np.sum(wt * wt))
		dj = (1/w_mag) * (np.dot(feature_vect,np.transpose(wt)) + bj)
		result[i] = dj
	maxindex = np.argmax(result,axis = 0)
	if (result[maxindex] < 0):
		text = 'NOT A SIGN'
		return text
	return st1[maxindex]

def getShapeFeatures(resizedImg):
	img = resizedImg.copy()
	gray = cv2.cvtColor(resizedImg,cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray,50,150,apertureSize = 3)
	hu = cv2.HuMoments(cv2.moments(edges)).flatten()
	Btb = getBordertoBoxfeatures(edges)
	shapeF = np.empty((1,87))
	shapeF[0,0:80] = Btb
	shapeF[0,80:87] = hu[:]
	return shapeF

def getAllFeatures(cropImg):
	resizedImg = cv2.resize(cropImg, (40,40))
	a,b,t = 10,10,4
	lenvect = a*b*t*t*t
	colorFeatures = getSegmentedBlocks(resizedImg,10,10,4) #from 40x40 image we obtain 16 10x10 blocks and compute hist in each
	shapeFeatures = getShapeFeatures(resizedImg)
	finalFeatures = np.empty((1,87+lenvect))
	finalFeatures[0,0:87] = shapeFeatures
	finalFeatures[0,87:lenvect+87] = colorFeatures[:]
	return finalFeatures
	
def getBoundingBox(thresh_img,orig):
	origCopy = orig.copy()
	contours, hierarchy = find_contours(thresh_img)
	cnt_no = find_maxContour(contours,thresh_img)
	for i in cnt_no:
		cnt = contours[i]
		area = cv2.contourArea(cnt)
		rect = cv2.minAreaRect(cnt)
		box = cv2.cv.BoxPoints(rect)
		box = np.int0(box)
		x,y,w,h = cv2.boundingRect(cnt)
		pt1, pt2 = (box[1,0],box[1,1]),(box[3,0],box[3,1])
		if ((np.absolute(pt2[1] - pt1[1]) > 15) and (np.absolute(pt2[0] - pt1[0]) > 15)):
			a = 5
			cv2.rectangle(orig,(x,y),(x+w,y+h),(0,255,0),2)
			cropImg = origCopy[y-a:y+h+a,x-a:x+w+a,:]
			feature_vect = getAllFeatures(cropImg) #pass this to the classifier and get what it is..
			classVal = classifyDetection(feature_vect)
			cv2.putText(orig,classVal,(x+15,y+15),cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, [0,155,255])
	return orig
	
def getAllFilters(im1,HSVim):
	orig = im1.copy()
	s = HSVim.shape[0:2]
	final_img = np.zeros((s), dtype=np.uint8)
	for i in range(3):
		HSV = FilterHSV(im1,HSVim,i+1)
		fil = ApplyFilterAndBinarize(HSV)
		im = getBoundingBox(fil,orig) #MAYBE FILTER BEFORE GETTING BOUNDING BOXES
		orig = im
		final_img = final_img + fil
	#final_img = cv2.GaussianBlur(final_img,(25,25),0)
	#th,final_img = cv2.threshold(final_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return final_img, orig

im1 = cv2.imread('t3.JPG')
HSVim = cv2.cvtColor(im1, cv2.COLOR_BGR2HSV)
HSV_final, boundingBox = getAllFilters(im1,HSVim)
#image_plot_utilities.show_with_pixel_values(im1[:,:,::-1])
#print HSV_final.shape
#image_plot_utilities.plot_pics([im1[:,:,::-1], HSV_final], 2, ['BoundingBox', 'Threshold'] )
cv2.imshow('BoundingBox', boundingBox)
cv2.waitKey(0)
