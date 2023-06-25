from __future__ import print_function
import numpy as np
import pywt
import cv2

def w2d(imArray, mode='haar', level=1):
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)   
    imArray /= 255;
    # compute coefficients 
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0;  

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)
    return imArray_H

class ImageProcess:
	def __init__(self,img):
		self.nor_img = img
		self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	def proceess_images(self):
		ret,binary = cv2.threshold(self.img,160,255,cv2.THRESH_BINARY)# 160 - threshold, 255 - value to assign, THRESH_BINARY_INV - Inverse binary
		kernel = np.ones((5,5),np.float32)/9  		#averaging filter
		dst = cv2.filter2D(binary,-1,kernel)		# -1 : depth of the destination image
		kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
		erosion = cv2.erode(dst,kernel2,iterations = 1)
		dilation = cv2.dilate(erosion,kernel2,iterations = 1)
		edges = cv2.Canny(dilation,100,200)			#edge detection
		contours,hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)	# Size detection
		total_ar=0
		for cnt in contours:
			x,y,w,h = cv2.boundingRect(cnt)
			aspect_ratio = float(w)/h
			if(aspect_ratio<1):
				aspect_ratio=1/aspect_ratio
			#print( round(aspect_ratio,2),self.get_classificaton(aspect_ratio))
			total_ar+=aspect_ratio
		avg_ar=total_ar/len(contours)
		#print ("Average Aspect Ratio=",round(avg_ar,2),self.get_classificaton(avg_ar))
		wavelet = w2d(self.nor_img)
		self.img_list = [self.img , binary , dst, erosion ,dilation , edges, wavelet]
		self.title = ["Original image","Binary image","Filtered image","Eroded image","Dialated image","Edge detect", "wavelet"]
		return self.img_list, self.title
