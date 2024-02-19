
from dlclive import DLCLive, Processor
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib
import random
from sklearn import metrics
import pandas as pd
import math
# import deeplabcut
import os
from datetime import date


# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Arial"

# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"


# Suppress annoying warnings!
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

###########################################################################

class dlclive_commutator_eval_auto():

	def __init__(self, dlc_model_path, video_path=None, dlc_display=False, manual_data_path="", model_name=""):

		self.video = None
		self.dlc_live = None
		self.dlc_proc = None

		self.dlc_model_path = dlc_model_path
		self.video_path = video_path
		self.dlc_display = dlc_display
		self.manual_data_path = manual_data_path
		self.model_name = model_name

		self.frames_toeval = 0
		self.manual_data = 0
		self.video_fps = 0
		self.video_total_frames = 0

	def init_dlc_processor(self):
		self.dlc_proc = Processor()
		print("DLC Processor Initialized.")

	def init_dlclive_object(self):
		self.dlc_live = DLCLive(self.dlc_model_path, processor=self.dlc_proc, display=self.dlc_display)
		print("DLC Live Object Initialized.")

	def import_video(self):
		self.video = cv2.VideoCapture(self.video_path)
		print("Video Capture Initialized.")

		self.video_fps = self.video.get(cv2.CAP_PROP_FPS)
		print('Frames per second =', self.video_fps)

		self.video_total_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
		print('Total frames in the video =', self.video_total_frames)

	def import_manual_data(self):
		self.manual_data = pd.read_csv(self.manual_data_path, header=None)
		self.frames_toeval = self.manual_data.shape[0]

		print("Manual data imported ...")
		print("Number of frames to evaluate: ", self.frames_toeval)
		print("")

	def start_evaluating(self):

		# CALL INITIALIZATION FUNCTIONS

		# DEBUG PRINT
		print("\nCalling initialization functions ...\n")

		self.init_dlc_processor()
		self.init_dlclive_object()
		self.import_video()
		self.import_manual_data()

		# DEBUG PRINT
		print("\nInitialization successful ...\n")

		# DEFINE VARIABLES FOR POSE COMPUTATION

		# Counter variable to keep track of frame count
		counter = 0

		# Variables to hold the different classifications
		fp_head = 0
		fn_head = 0
		tp_head = 0
		tn_head = 0
		fp_tail = 0
		fn_tail = 0
		tp_tail = 0
		tn_tail = 0
		actual_head = []
		predicted_head = []
		actual_tail = []
		predicted_tail = []

		invalid_head = 0
		invalid_tail = 0

		invalid_head_frames = []
		invalid_tail_frames = []

		inference_time = []
		frames_inferenced = []

		# Thresholds
		accuracy_threshold = 95
		multiplier = 0.25
		x_threshold = self.video.get(cv2.CAP_PROP_FRAME_WIDTH) * multiplier
		y_threshold = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT) * multiplier
		# x_threshold = 10      # 10
		# y_threshold = 10      # 10

		# Global time variable
		start_time = 0
		end_time = 0
		inference_time = []

		# DEBUG PRINT
		print("\nInitializing Evaluation ...\n")

		# Run inference on first frame to eliminate delay
		print("\nRunning inference on the first frame to eliminate initial inference time lag ...\n")

		# Set the video position to the frame
		self.video.set(cv2.CAP_PROP_POS_FRAMES, 1)

		# Read single frame
		ret, frame = self.video.read()

		# Get inference from an image
		img_pose = self.dlc_live.init_inference(frame)

		print("\nEntering while loop ...\n")

		while self.video.isOpened() and counter < self.frames_toeval:

			# Determine frame to read from the annotated manual data
			frame_id = self.manual_data[4][counter]

			# Set the video position to the frame
			self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

			# Read single frame
			ret, frame = self.video.read()

			# if frame is read correctly ret is True
			if not ret:
				print("Can't receive frame (stream end?). Exiting ...")
				break

			start_time = time.time()

			# Get inference from an image
			img_pose = self.dlc_live.get_pose(frame)

			end_time = time.time()

			# Get inference duration
			time_diff = end_time - start_time
			inference_time.append(time_diff)
			print("Counter:		", counter)
			print("frame id:	", frame_id)
			print("Inference time: %s seconds" % time_diff)
			print("Image Pose:	", img_pose)
			print("")

			auto_head_x = img_pose[0][0]
			auto_head_y = img_pose[0][1]
			auto_tail_x = img_pose[1][0]
			auto_tail_y = img_pose[1][1]
			auto_head_accuracy = img_pose[0][2] * 100
			auto_tail_accuracy = img_pose[1][2] * 100

			manual_head_x = self.manual_data[0][counter]
			manual_head_y = self.manual_data[1][counter]
			manual_tail_x = self.manual_data[2][counter]
			manual_tail_y = self.manual_data[3][counter]

			# TEST
			# Print out the differences between manual and automatic labels
			print("HEAD X DIFFERENCE:		", auto_head_x - manual_head_x)
			print("HEAD Y DIFFERENCE:		", auto_head_y - manual_head_y)
			print("TAIL X DIFFERENCE:		", auto_tail_x - manual_tail_x)
			print("TAIL Y DIFFERENCE:		", auto_tail_y - manual_tail_y)
			print("\n\n")

			# HEAD
			if not math.isnan(auto_head_x) and not math.isnan(auto_head_y):
				if abs(auto_head_x - manual_head_x) > x_threshold or abs(auto_head_y - manual_head_y) > y_threshold:
					if auto_head_accuracy >= accuracy_threshold:
						fp_head = fp_head + 1
						actual_head.append(0)
						predicted_head.append(1)
					else:
						tn_head = tn_head + 1
						actual_head.append(1)
						predicted_head.append(0)
				else:
					if auto_head_accuracy >= accuracy_threshold:
						tp_head = tp_head + 1
						actual_head.append(1)
						predicted_head.append(1)
					else:
						fn_head = fn_head + 1
						actual_head.append(0)
						predicted_head.append(0)
			else:
				invalid_head += 1
				invalid_head_frames.append(frame_id)

			# TAIL
			if not math.isnan(auto_tail_x) and not math.isnan(auto_tail_y):
				if abs(auto_tail_x - manual_tail_x) > x_threshold or abs(auto_tail_y - manual_tail_y) > y_threshold:
					if auto_tail_accuracy >= accuracy_threshold:
						fp_tail = fp_tail + 1
						actual_tail.append(0)
						predicted_tail.append(1)
					else:
						tn_tail = tn_tail + 1
						actual_tail.append(1)
						predicted_tail.append(0)
				else:
					if auto_tail_accuracy >= accuracy_threshold:
						tp_tail = tp_tail + 1
						actual_tail.append(1)
						predicted_tail.append(1)
					else:
						fn_tail = fn_tail + 1
						actual_tail.append(0)
						predicted_tail.append(0)
			else:
				invalid_tail += 1
				invalid_tail_frames.append(frame_id)

			# increment counter
			counter = counter + 1

			# Sleep for 5 seconds to display final results
			# time.sleep(3.0)

		print("  ***   ")
		print("        ")
		print("Final predicted label classes for head:")
		print("1  (true positive): ", tp_head)
		print("2  (true negative): ", tn_head)
		print("3 (false positive): ", fp_head)
		print("4 (false negative): ", fn_head)
		print("        ")
		print("Final predicted label classes for tail:")
		print("1  (true positive): ", tp_tail)
		print("2  (true negative): ", tn_tail)
		print("3 (false positive): ", fp_tail)
		print("4 (false negative): ", fn_tail)
		print("        ")
		print("Actual head: ", actual_head)
		print("Predicted head: ", predicted_head)
		print("        ")
		print("Actual tail: ", actual_tail)
		print("Predicted tail: ", predicted_tail)
		print("        ")
		print("Inference times: ", inference_time)

		#########################################################################
		# Save confusion matrix and inference time data to a csv file

		# Combine the head and tail data into a list
		ht = [['True Positive', 'True Negative', 'False Positive', 'False Negative', 'Invalid'],
			  [tp_head, tn_head, fp_head, fn_head, invalid_head],
			  [tp_tail, tn_tail, fp_tail, fn_tail, invalid_tail]]

		# Convert the list to dataframe
		df_ht = pd.DataFrame(ht).transpose()

		# Set column titles
		columns = ['Class', 'Head', 'Tail']
		df_ht.columns = columns

		# Inference time
		df_it = pd.DataFrame(inference_time)
		df_it.columns = ['Inference Time (seconds)']

		# Evaluation information
		eval_params = [['x-threshold', 'y-threshold', 'accuracy threshold'],
					   [x_threshold, y_threshold, accuracy_threshold]]
		df_eval_params = pd.DataFrame(eval_params).transpose()

		#########################################################################
		# date object of today's date
		today = date.today() 

        # Save data to csv on disk
		ht_filename = MComp + "\Data\EVAL_" + model_name + "_htcm_" + str(today) + ".csv"
		df_ht.to_csv(ht_filename)

		it_filename = MComp + "\Data\EVAL_" + model_name + "_it_" + str(today) + ".csv"
		df_it.to_csv(it_filename)

		eval_params_filename = MComp + "\Data\EVAL_" + model_name + "_params_" + str(today) + ".csv"
		df_eval_params.to_csv(eval_params_filename)

		
###########################################################################

# Setup data paths
# Some accronym definitions: 
# GT = Ground Truth

MComp = "H:\Other computers\My PC\PhD_UOM\General\BSBRL\Projects\Motorized_Commutator\Paper\Model_Comparison"

# Ground truth data path
GT_APA = MComp + "\Ground_Truth\APA_Ground_Truth.csv"
GT_BM = MComp + "\Ground_Truth\BarnesMaze_Ground_Truth.csv"
GT_OF = MComp + "\Ground_Truth\OpenField_Ground_Truth.csv"

# Load the csv data
GT_APA_Data = pd.read_csv(GT_APA, header=None)
GT_BM_Data = pd.read_csv(GT_BM, header=None)
GT_OF_Data = pd.read_csv(GT_OF, header=None)

# DLC Exported Models
# model_path = MComp + "\Exported_Models\DLC_APA_mobilenet"
model_path = MComp + "\Exported_Models\DLC_APA_resnet"

# Behavior videos
# video_topose = MComp + "\Videos\Open_Field.avi"
video_topose = MComp + "\Videos\APA.mp4"

# model_name = "DLC_MOBILENET-APA"
model_name = "DLC_RESNET-APA"

# Define manual data path
manual_data_path = GT_APA

###########################################################################

# Run evaluation
poser = dlclive_commutator_eval_auto(model_path, video_topose, dlc_display=False, manual_data_path=manual_data_path, model_name=model_name)
poser.start_evaluating()
