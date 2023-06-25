from __future__ import print_function
import tensorflow as tf
import numpy as np
import pandas as pd
import os

from frcnn_mtl import *
from image_process import ImageProcess

import tkinter as tki
from PIL import Image
from PIL import ImageTk
import threading
import imutils
import cv2
import time

class MyThread(threading.Thread):
	def __init__(self):
		threading.Thread.__init__(self)
		self.stop = threading.Event()

	def terminate(self):
		self.stop.set()
		
class PhotoBoothApp:
	def __init__(self, vs):
		# store the video stream object and output path, then initialize the most recently read frame, 
		# thread for reading frames, and the thread stop event
		self.vs = vs
		self.frame = None
		self.thread = None
		self.stopEvent = None		
		self.yo_thread = MyThread()
		
		self.root = tki.Tk()        # initialize the root window and image panel	
		self.root.geometry("1300x660")

		w = tki.Label(self.root, text="Mango(Image) Collect").place(y =10, anchor="center", x =650)

		image = cv2.imread("load.jpg")
		image = Image.fromarray(image)
		image = ImageTk.PhotoImage(image)
		self.panel = tki.Label(image=image, padx= 10, pady =10)
		self.panel.place( y =270, anchor="w", x=20)
		self.panel1 = tki.Label(image=image, padx=10, pady=10 )
		self.panel1.place(y = 270, anchor = "w", x = 670)
		self.panel.configure(image=image)
		self.panel.image = image
		self.panel1.configure(image=image)
		self.panel1.image = image

		start_time = time.time()
		btn = tki.Button(self.root, text="Take",command=self.run_process,padx=10,pady=10 )
		btn.place( y =540,anchor="center", x =650)
		timey  =(time.time() - start_time)

		self.instruct =  "Click Take Button..."
		self.text_label = tki.Label(self.root, text=self.instruct, padx=10,pady=10,)
		self.text_label.place(y =580, anchor="center", x= 650)

		tit = ["Grade I %", "Grade II %", "Grade III %", "Defect %", "Grade Percent", "Quality", "Size"]
		g = 1300//7
		for i in range(7):
			self.e = tki.Label(self.root,text = tit[i])
			self.e.place(y= 610 , x = (i*g)+15 , anchor = 'w')
		
		self.val_obj = []
		for i in range(7):
			self.e = tki.Label(self.root,fg="red",text = "-")
			self.e.place(y= 630 , x = (i*g)+20 , anchor = 'w')
			self.val_obj.append(self.e)

		self.stopEvent = threading.Event()
		self.thread = threading.Thread(target=self.videoLoop)
		self.thread.start()

	def videoLoop(self):
		# This try/except statement is a pretty ugly hack to get around
		# a RunTime error that Tkinter throws due to threading 
		# If possible, look for a possible solution around this
		try:
			# keep looping over frames until we are instructed to stop
			while not self.stopEvent.is_set():
				# grab the frame from the video stream and resize it to
				self.frame = self.vs.read()
				self.frame = imutils.resize(self.frame, width=300)

		except RuntimeError:
			print("[INFO] caught a RuntimeError")

	def run_process(self):
		img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
		image = Image.fromarray(img)
		image = ImageTk.PhotoImage(image)

		self.panel.configure(image=image)
		self.panel.image = image
		self.panel1.configure(image=image)
		self.panel1.image = image	

		df,alrt = take_picture(img)
		val = df.iloc[0]
		val = val.to_numpy()
		if alrt:
			self.instruct = "Predicted Successfully!..."
			color = "blue"
		else:
			self.instruct = "Please, Show the Valid Image... "
			color = "red"
		self.text_label.configure(text = self.instruct)
		self.text_label.text = self.instruct
		self.text_label.configure(fg = color)
		self.text_label.fg = color

		for i in range(7):
			obj = self.val_obj[i]
			obj.configure(text = val[i])
			obj.text = val[i]
			obj.configure(fg = color)
			obj.fg = color
	
	def onClose(self):
		# set the stop event, cleanup the camera, and allow the rest of
		# the quit process to continue
		print("[INFO] closing...")
		self.stopEvent.set()
		self.vs.stop()
		self.root.quit()

class Open_Results():
	def __init__(self, df):
		self.root = tki.Tk() 	
		self.root.geometry("1300x150")
		self.instruct =  "Predicted Successfully..."
		self.text_label = tki.Label(self.root, text=self.instruct, padx=10,pady=10,)
		self.text_label.place(y =15, anchor="center", x= 650)
		val = df.iloc[0]
		val = val.to_numpy()
		tit = ["Grade I %", "Grade II %", "Grade III %", "Defect %", "Grade Percent", "Quality", "Size"]
		g = 1300//7
		for i in range(7):
			self.e = tki.Label(self.root,text = tit[i])
			self.e.place(y=50 , x = (i*g)+15 , anchor = 'w')
		
		self.val_obj = []
		for i in range(7):
			self.e = tki.Label(self.root,fg="blue",text = val[i])
			self.e.place(y= 110 , x = (i*g)+20 , anchor = 'w')
			self.val_obj.append(self.e)

def Live_Image_Capture():
    from imutils.video import VideoStream
 
    print("[INFO] warming up camera...")
    vs = VideoStream(0).start()
    time.sleep(2.0)

    # start the app
    pba = PhotoBoothApp(vs)
    pba.root.mainloop()

print("Start ...")

model_mtl,model_frcnn,model_lr = create_models()
HEIGHT = 256
WIDTH = 128
DIMENSION =3
def image_to_array(image):
    image=Image.fromarray(image)
    #image = Image.open(crop_img)
    image = image.resize((HEIGHT,WIDTH))
    #image = image.convert("L")
    return np.array(image)    # return values as numpy array format

def cropped_image(image): 
	(boxes, scores, classes, num) = model_frcnn.run(image)
	box_array, scores_ = get_box(boxes, scores, image)
	if scores_:
		left, right, top, bottom  = map(lambda x :int(x),box_array)
		#rec_img = cv2.rectangle(image,(left,top),(right,bottom),(255,255,0),2)
		crop = image[top:bottom, left:right]	
		return crop
	
def image_analysis(image):
	(boxes, scores, classes, num) = model_frcnn.run(image)
	box_array, scores_ = get_box(boxes, scores, image)
	print(box_array, scores_)
	if scores_ > 0.95:
		left, right, top, bottom  = map(lambda x :int(x),box_array)
		rec_img = cv2.rectangle(image,(left,top),(right,bottom),(255,255,0),2)
		crop = image[top:bottom, left:right]
		ImageProcess_obj = ImageProcess(crop)
		img_list, img_title = ImageProcess_obj.proceess_images()	
		img_list.append(rec_img)
		img_title.append("Object Detect")
		img_list.append(crop)
		img_title.append("Cropped Img")
		x, y =get_size(crop, 82)
		weight, size = predict_size([x,y])
		quality, ripeness = model_mtl.run(crop)
		img, quality_, ripeness_ = draw_boxes_scores(box_array, scores_, ripeness, quality, image)
		plot_img(3,3,img_list,img_title)
	else:
		quality_, ripeness_, size = 0,0,0
	return quality_, ripeness_, size

def load_model():
    import joblib
    return joblib.load('new_model.pkl')

def Image_Predict_Func():
	find_path = r"../Dataset/Grading_dataset/Extra_Class/Other"
	images_to_find = os.listdir(find_path)
	n =1
	details = []
	data = []
	for img_file in images_to_find[:n]:
		path = os.path.join(find_path, img_file)
		image = cv2.imread(path)
		quality, ripeness, size = image_analysis(image)
		if (quality==0 and ripeness== 0 and size ==0):
			df = pd.DataFrame([ ['-','-','-','-','-','-','-'] ], columns = ['Grade I %','Grade II %','Grade III %','Defect %',"Grade Percent","Quality",  "Size"])
			print(df)
			return 
		details.append([size])
		crop_img = cropped_image(image)
		data.append(image_to_array(crop_img) /255)
	to_find_grade_img = np.array(data).reshape(len(data), -1)        	    
	savedModel = load_model()
	h2 =savedModel.predict_proba(to_find_grade_img)
	df = pd.DataFrame(h2 * 100, columns =['Grade I','Grade II','Grade III','Defect'])
	grd = df.idxmax(axis=1)
	val = df.max(axis=1)
	df = pd.concat([df,val,grd,pd.DataFrame(details)],axis=1)
	df.columns = ['Grade I %','Grade II %','Grade III %','Defect %',"Grade Percent","Quality",  "Size"]
	print(df)

	cr = Open_Results()
	cr.mainloop()

def take_picture(image):
	details = []
	data = []
	quality, ripeness, size = image_analysis(image)
	if (quality==0 and ripeness== 0 and size ==0):
		df = pd.DataFrame([ ['-','-','-','-','-','-','-'] ], columns = ['Grade I %','Grade II %','Grade III %','Defect %',"Grade Percent","Quality",  "Size"])
		print(df)
		return df, False
	details.append([size])
	crop_img = cropped_image(image)
	data.append(image_to_array(crop_img) /255)
	to_find_grade_img = np.array(data).reshape(len(data), -1)        	    
	savedModel = load_model()
	h2 =savedModel.predict_proba(to_find_grade_img)
	df = pd.DataFrame(h2 * 100, columns =['Grade I','Grade II','Grade III','Defect'])
	grd = df.idxmax(axis=1)
	val = df.max(axis=1)
	df = pd.concat([df,val,grd,pd.DataFrame(details)],axis=1)
	df.columns = ['Grade I %','Grade II %','Grade III %','Defect %',"Grade Percent","Quality",  "Size"]
	print(df)
	return df, True

v =  int(input("Welcome to Mango Grading Application \nEnter 1 for Image prediction model \nEnter 2 for Video Capturing\n"))

if (v==1):
	Image_Predict_Func()
elif (v==2):
	Live_Image_Capture()
else:
	print("Please Enter Correct value")
