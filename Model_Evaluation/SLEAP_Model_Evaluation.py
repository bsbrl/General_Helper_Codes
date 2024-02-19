# Import required libraries
import sleap
import pandas as pd
import time
import math
import matplotlib.pyplot as plt
import matplotlib
from sklearn import metrics
import numpy as np
from datetime import date


# Say, "the default sans-serif font is COMIC SANS"
matplotlib.rcParams['font.sans-serif'] = "Arial"

# Then, "ALWAYS use sans-serif fonts"
matplotlib.rcParams['font.family'] = "sans-serif"


# Suppress annoying warnings!
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


# sleap.disable_preallocation()

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

# SLEAP Exported Models
# SLEAP_APA = MComp + "\Exported_Models\SLEAP_APA_240213.zip"
# SLEAP_BM = MComp + "\Exported_Models\SLEAP_BM_240213.zip"
SLEAP_OF = MComp + "\Exported_Models\SLEAP_OF_240214.zip"

# Behavior videos
# APA_Video = sleap.load_video(MComp + "\Videos\APA.mp4")
# BM_Video = sleap.load_video(MComp + "\Videos\Barnes_Maze.mp4")
OF_Video = sleap.load_video(MComp + "\Videos\Open_Field.avi")

###########################################################################

model_name = "SLEAP-OF"

# Run through each bevaior one at a time
manual_data = GT_OF_Data
video = OF_Video

# Initialize the sleap model and video file
predictor = sleap.load_model(SLEAP_OF, batch_size=16)

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
accuracy_threshold = 95   # 95
multiplier = 0.25
x_threshold = video.shape[2] * multiplier
y_threshold = video.shape[1] * multiplier
# x_threshold = 10      # 10
# y_threshold = 10      # 10

# Global time variable
start_time = 0
end_time = 0
inference_time = []

###########################################################################

# Warm up SLEAP inference before loop
# Get inference from an image frame
prediction = predictor.inference_model.predict(video[0], numpy=True)

# Loop through frames in csv
for i in range(manual_data.shape[0]):

  # Determine frame to read from the annotated manual data
  img = video[manual_data[4][i]]

  start_time = time.time()

  # Get inference from an image frame
  prediction = predictor.inference_model.predict(img, numpy=True)

  end_time = time.time()

  frame_id = manual_data[4][i]

  # Get inference duration
  time_diff = end_time - start_time
  inference_time.append(time_diff)
  print("Counter:		", i)
  print("frame id:	", frame_id)
  print("Inference time: %s seconds" % time_diff)
  print("Image Pose:	", prediction)
  print("")

  auto_head_x = prediction['instance_peaks'][0][0][0][0]
  auto_head_y = prediction['instance_peaks'][0][0][0][1]

  auto_tail_x = prediction['instance_peaks'][0][0][1][0]
  auto_tail_y = prediction['instance_peaks'][0][0][1][1]

  auto_head_accuracy = prediction['instance_peak_vals'][0][0][0] * 100
  auto_tail_accuracy = prediction['instance_peak_vals'][0][0][1] * 100

  manual_head_x = manual_data[0][i]
  manual_head_y = manual_data[1][i]
  manual_tail_x = manual_data[2][i]
  manual_tail_y = manual_data[3][i]

  # TEST
  # Print out the differences between manual and automatic labels
  print("HEAD X DIFFERENCE:		", auto_head_x - manual_head_x)
  print("HEAD Y DIFFERENCE:		", auto_head_y - manual_head_y)
  print("TAIL X DIFFERENCE:		", auto_tail_x - manual_tail_x)
  print("TAIL Y DIFFERENCE:		", auto_tail_y - manual_tail_y)
  print("\n\n")

  # """
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

###########################################################################

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
print("Invalid head: ", invalid_head, ",  Frames: ", invalid_head_frames)
print("Invalid tail: ", invalid_tail, ",  Frames: ", invalid_tail_frames)
print("        ")
print("Actual head: ", actual_head)
print("Predicted head: ", predicted_head)
print("        ")
print("Actual tail: ", actual_tail)
print("Predicted tail: ", predicted_tail)
print("        ")
print("Inference times: ", inference_time)

###########################################################################

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

# date object of today's date
today = date.today() 

# Save data to csv on disk
ht_filename = MComp + "\Data\EVAL_" + model_name + "_htcm_" + str(today) + ".csv"
df_ht.to_csv(ht_filename)

it_filename = MComp + "\Data\EVAL_" + model_name + "_it_" + str(today) + ".csv"
df_it.to_csv(it_filename)

eval_params_filename = MComp + "\Data\EVAL_" + model_name + "_params_" + str(today) + ".csv"
df_eval_params.to_csv(eval_params_filename)

