import numpy as np
import math
import cv2
import os
import json
import copy
#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from ...utils.box import BoundBox


from .tracker import Tracker

# get our tracking module object
global tracker
tracker = Tracker()

global frame
frame = 0

global out_dir
global make_training_data
make_training_data = True
good_path = False
while not good_path:
	out_dir = input("Please enter a directory to store your output for bounding boxes -- if not getting data and using the traker, enter [s\S] to skip ")
	if os.path.isdir(out_dir):
		good_path = True
	elif out_dir == "S" or out_dir == "s":
		good_path = True
		# global skip_gen
		make_training_data = False
		print("skipping, using the tracker")
	else:
		print("Invalid path")

from ...cython_utils.cy_yolo2_findboxes import box_constructor

def expit(x):
	return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def findboxes(self, net_out):
	# meta
	meta = self.meta
	boxes = list()
	boxes=box_constructor(meta,net_out)
	return boxes

def postprocess(self, net_out, im, save = True):
	"""
	Takes net output, draw net_out, save to disk
	"""
	global frame
	frame += 1
	print('frame: ', frame)
	boxes = self.findboxes(net_out)

	# meta
	meta = self.meta
	threshold = meta['thresh']
	colors = meta['colors']
	labels = meta['labels']
	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im
	h, w, _ = imgcv.shape
	resultsForJSON = []

	# for saving largest bounding box of girl
	max_area = 0
	max_image = None         # track image that contains largest bounding box
	lesser_images_for_frame = []     # hold processed bounding box images for each frame

	# copy of image for bounding box isolation

	for i, b in enumerate(boxes):
		boxResults = self.process_box(b, h, w, threshold)
		if boxResults is None:
			continue
		left, right, top, bot, mess, max_indx, confidence = boxResults
		thick = int((h + w) // 300)
		if self.FLAGS.json:
			resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
			continue

		# create bounding box
		cv2.rectangle(imgcv,
			(left, top), (right, bot),
			colors[max_indx], thick)
		# don't need YOLO object classification
		# cv2.putText(imgcv, mess, (left, top - 12),
		# 	0, 1e-3 * h, colors[max_indx],thick//3)

		# get copy of image for each box made
		imgcv_copy = copy.copy(imgcv)
		# process it as blackened
		imgcv_copy = blacken_image(imgcv_copy, h, w, left, top, right, bot)
		imgcv_copy = resize_image(imgcv_copy)

		# we're not making data we're using the tracker
		if not make_training_data:
			pred = tracker.get_prediction(imgcv_copy)
			if pred == 1:
				# save a red circle over the predicted bounding box that yolo is using
				cv2.circle(imgcv, (left, top), 20, (0, 0, 250), -1)

		y = top
		x = left
		x2 = right
		y2 = bot
		roi = im[y:y2, x:x2]

		area = roi.shape[0] * roi.shape[1]
		if area > max_area:
			max_image = imgcv_copy
			max_area = area

		# hold all imgcv_copy objects
		lesser_images_for_frame.append(imgcv_copy)

	# global make_training_data
	if make_training_data:
		# save all of these images
		global out_dir
		# save largest first
		cv2.imwrite((out_dir + "\\%d%dz.jpg" % (frame, 1)), max_image)
		# save others after in sequence
		for idx, img in enumerate(lesser_images_for_frame):
			if img is not max_image:
				cv2.imwrite((out_dir + "\\%d%d.jpg" % (frame, idx+2)), img)

	if not save: return imgcv

	outfolder = os.path.join(self.FLAGS.imgdir, 'out')
	img_name = os.path.join(outfolder, os.path.basename(im))
	if self.FLAGS.json:
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"
		with open(textFile, 'w') as f:
			f.write(textJSON)
		return

	cv2.imwrite(img_name, imgcv)


def blacken_image(image, h, w, left, top, right, bot):
	"""blackens the image surrounding the rectangle of the bounding box"""
	black = (0, 0, 0)
	fill  = -1
	cv2.rectangle(image, (0, 0), (w, top), black, fill)
	cv2.rectangle(image, (0, top), (left, h), black, fill)
	cv2.rectangle(image, (left, bot), (w, h), black, fill)
	cv2.rectangle(image, (right, top), (w, bot), black, fill)
	return image

def resize_image(image):
	""" Resizes the image to 250 x 250 for the CNN """
	size = 250
	image = cv2.resize(image, (size, size), 1, 1, interpolation=cv2.INTER_AREA)
	return image