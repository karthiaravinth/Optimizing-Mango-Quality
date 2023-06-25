from __future__ import print_function
import tensorflow as tf
import numpy as np
import cv2
import logging
import matplotlib.pyplot as plt

class Import_Frcnn():
	def __init__(self, location):
		self.graph_frcnn = tf.Graph()
		self.sess = tf.compat.v1.Session(graph=self.graph_frcnn)
		with self.graph_frcnn.as_default():
			self.od_graph_def = tf.compat.v1.GraphDef()	
			with tf.io.gfile.GFile(location, 'rb') as fid:
				serialized_graph = fid.read()
				self.od_graph_def.ParseFromString(serialized_graph)
				tf.import_graph_def(self.od_graph_def, name='')
		try:
			self.image_tensor = self.graph_frcnn.get_tensor_by_name('image_tensor:0')
			self.detection_boxes = self.graph_frcnn.get_tensor_by_name('detection_boxes:0')
			self.detection_scores = self.graph_frcnn.get_tensor_by_name('detection_scores:0')
			self.detection_classes = self.graph_frcnn.get_tensor_by_name('detection_classes:0')
			self.num_detections = self.graph_frcnn.get_tensor_by_name('num_detections:0')
			print("Model FRCNN ready")
		except:
			logging.warning("FRCNN Model loading Error...")

	def run(self, frame):
		image_np = np.expand_dims(frame, axis = 0)
		return self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],feed_dict={self.image_tensor: image_np})

class Import_MTL():
	def __init__(self, location):
		self.graph_mtl = load_graph(location)
		self.sess = tf.compat.v1.Session(graph = self.graph_mtl)
		self.y_pred_quality = self.graph_mtl.get_tensor_by_name("prefix/y_pred_quality:0")
		self.y_pred_ripeness = self.graph_mtl.get_tensor_by_name("prefix/y_pred_ripeness:0")
		self.x = self.graph_mtl.get_tensor_by_name("prefix/x:0") 
		print("Model MTL ready")

	def run(self, frame):
		image_rgb = cv2.resize(frame, (50,50))
		image_rgb = np.expand_dims(image_rgb, axis = 0)
		return self.sess.run([self.y_pred_quality, self.y_pred_ripeness], feed_dict={self.x: image_rgb})

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
    with tf.io.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def normalize_size(x):
    #Values obtained from train.py output
    mean = np.asarray([[76.06636364, 119.57220779]])
    std = np.asarray([[5.95719927, 8.19216614]])
    x_normalized = (x - mean) / std
    return x_normalized

def convert_sizes(size):
	size = int(size)
	if size >= 400:
		return "Large"
	elif size <= 399 and size >= 200:
		return "medium"
	elif size < 199:
		return "small"

def predict_size(x_input):
    # input: [width, length, thickness]
    # output: [size, size_classification]
    x_input = normalize_size(x_input)
    graph = load_graph("frozen_models/LR_frozen_model.pb")

    #input and output node
    x = graph.get_tensor_by_name('prefix/x:0')
    y = graph.get_tensor_by_name('prefix/Wx_b/Add:0')

    with tf.compat.v1.Session(graph = graph) as sess:
        y_output = sess.run(y, feed_dict={x: x_input})

    return y_output, convert_sizes(y_output)

def get_size(image_frame, calibrated_pxm):
	#This function returns Y dimension, X dimension, Area, midx, midy, 
	gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
	cv2.imwrite("gray.png", gray)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)
	cv2.imwrite("gaussianblur.png", gray)
	# perform edge detection, then perform a dilation + erosion to close gaps in between object edges
	canny = cv2.Canny(gray, 50, 100)
	edged = cv2.dilate(canny, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)
	#Find contours
	(cnts,_) = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	#print(cnts,_)
	areaArray = []
	for i, c in enumerate(cnts):
		area = cv2.contourArea(c)
		areaArray.append(area)
	sorteddata = sorted(zip(areaArray,cnts), key=lambda x: x[0], reverse=True)
	c = sorteddata[0][1] 
	x,y,w,h = cv2.boundingRect(c)
	dA,dB = w,h
	X = dA / calibrated_pxm
	Y = dB / calibrated_pxm
	X = X * 25.4
	Y = Y * 25.4
	if X > Y:
		Y_ = X
		X_ = Y
	else:
		Y_ = Y
		X_ = X
	return X_, Y_

def get_box(boxes, scores, image):
	boxes = np.squeeze(boxes)
	height, width = image.shape[:2]
	box = None
	score = None
	ymin, xmin, ymax, xmax = boxes[0]
	box = [xmin * width, xmax * width, ymin * height, ymax * height]
	score = scores.item(0)
	return box, score

def draw_boxes_scores(box_array, score_array, ripe_array, quality_array, frame):
	ripeness_dict = {0: 'Green', 1: 'Semi-Ripe', 2: 'Ripe'}
	quality_dict = {0: 'Good', 1: 'Defect'}
	cv2.rectangle(frame, (int(box_array[0]), int(box_array[2])), (int(box_array[1]), int(box_array[3])),(0,255,0),3)
	#cv2.putText(frame, "Detection:  {0:.2f}".format(score_array), (10, 16), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0,0,0))
	#cv2.putText(frame, "Quality:  {}".format(quality_dict[int(np.argmax(quality_array, axis=1))]), (10, 32), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0,0,0))
	#cv2.putText(frame, "Ripeness:  {}".format(ripeness_dict[int(np.argmax(ripe_array, axis=1))]), (10, 48), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0,0,0))
	return frame, quality_dict[int(np.argmax(quality_array, axis=1))], ripeness_dict[int(np.argmax(ripe_array, axis=1))]

def create_models():
	#predict up to 3 items only
	model_mtl = Import_MTL("frozen_models/MTL_frozen_model.pb")
	model_frcnn = Import_Frcnn('frozen_models/frozen_inference_graph.pb')
	model_lr = load_graph('frozen_models/LR_frozen_model.pb')
	return model_mtl,model_frcnn,model_lr 


def plot_img(imgs_row, imgs_col ,img_list, img_title):
	#imgs_row ,imgs_col =2,3
	n = imgs_row * imgs_col
	for no in range(n):
		plt.subplot(imgs_row,imgs_col,no+1)
		plt.imshow(img_list[no],'gray')
		plt.title(img_title[no])
	plt.show()