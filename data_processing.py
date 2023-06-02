import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import sys 
sys.path.insert(0, '/scratch/vivianep/event_based_gaze_tracking/visualize.py')
sys.path.insert(0, '/scratch/vivianep/event_based_gaze_tracking/detect_pupil.py')
import visualize
import detect_pupil
import scipy
import argparse
import time
import tkinter as tk
import plotly.graph_objects as go


FRAME_WIDTH = 346
FRAME_HEIGHT = 260

parser = argparse.ArgumentParser(description='Arguments for using the eye visualizer')
parser.add_argument('--subject', type=int, default=4, help='which subject to evaluate')
parser.add_argument('--eye', default=0, choices=[0, 1], help='Which eye to visualize, left (0) or right (1)')
parser.add_argument('--debug', default=False, help='Whether to print debug statements')

opt, unknown = parser.parse_known_args()

subject = opt.subject
eye = opt.eye
debug = opt.debug

# load the dataset for the subject
eye_dataset = visualize.EyeDataset(os.path.join(os.getcwd(), 'eye_data'), subject)
eye_dataset.collect_data(eye)

# access the frame stack of eye_dataset
frame_stack = eye_dataset.frame_stack
# access the event stack of eye_dataset
event_stack = eye_dataset.event_stack

# convert the data stacks to dataframes
frame_df = pd.DataFrame(frame_stack, columns=['row', 'col', 'img_path', 'timestamp'])

event_stack = np.array(event_stack).reshape(-1, 4)
timestamps = event_stack[:, 0]
x = event_stack[:, 1]
y = event_stack[:, 2]
polarity = event_stack[:, 3]

event_df = pd.DataFrame({'timestamp': timestamps, 'x': x, 'y': y, 'polarity': polarity})

# add a column to event_df that indicates the frame number
frame_rate = 25 # fps
event_df['frame_number'] = event_df['timestamp'].apply(lambda x: int(x / (1000000 / frame_rate)))
frame_df['frame_number'] = frame_df['timestamp'].apply(lambda x: int(x / (1000000 / frame_rate)))

# iterate over the frame_df and detect the center of the pupil in each frame
centers = []
for frame_path in frame_df['img_path'].values:
    if debug:
        print("Processing frame {}...".format(frame_path))
    ellipse, _ = detect_pupil.detect_pupil(frame_path)
    if ellipse is None:
        center = None
    else:
        center = ellipse[0]
    centers.append(center)

# add the centers to the frame_df
frame_df['pupil'] = centers

frame_height = FRAME_HEIGHT
frame_width = FRAME_WIDTH

# set pupile centers to None if they are close to the edge of the image
frame_df['pupil'] = frame_df['pupil'].apply(lambda x: None if x is None else (None if x[0] < 0.1 * frame_width or x[0] > 0.9 * frame_width or x[1] < 0.1 * frame_height or x[1] > 0.9 * frame_height else x))

# original number of frames
original_num_frames = len(frame_df)

original_frame_df = frame_df.copy()

frame_df = frame_df.dropna(subset=['pupil'])

# filter out all pupil centers that are not on a black pixel
pupil_centers = frame_df['pupil'].values
frame_paths = frame_df['img_path'].values
frame_numbers = frame_df['frame_number'].values

# check which pupil centers are on a black pixel
for row in frame_df.iterrows():
    frame_path = row[1]['img_path']
    frame_img = cv2.imread(frame_path)
    coords = (row[1]['pupil'][1], row[1]['pupil'][0])
    coords = (int(coords[0]), int(coords[1]))
    pixel_value = frame_img[coords]
    # check if the pixel value is black
    if not (np.all(pixel_value == 0)):
        # if it is not black, drop the row
        frame_df = frame_df.drop(row[0])


# we will now interpolate the pupil centers that are None
# using the spline interpolation method

new_indices = range(original_frame_df.index.min(), original_frame_df.index.max() + 1)
frame_df = frame_df.reindex(new_indices)

frame_df['img_path'] = original_frame_df['img_path'].fillna(original_frame_df['img_path'])
frame_df['timestamp'] = original_frame_df['timestamp'].fillna(original_frame_df['timestamp'])
frame_df['row'] = original_frame_df['row'].fillna(original_frame_df['row'])
frame_df['col'] = original_frame_df['col'].fillna(original_frame_df['col'])
frame_df['frame_number'] = original_frame_df['frame_number'].fillna(original_frame_df['frame_number'])
frame_df[['pupil_x', 'pupil_y']] = pd.DataFrame(frame_df['pupil'].tolist(), index=frame_df.index)
frame_df['pupil_x'] = frame_df['pupil_x'].interpolate(method='spline', order=3)
frame_df['pupil_y'] = frame_df['pupil_y'].interpolate(method='spline', order=3)
frame_df['pupil'] = frame_df[['pupil_x', 'pupil_y']].values.tolist()

print(frame_df)


# Calculate z-scores for x and y coordinates
z_scores_x = np.abs((frame_df['pupil_x'] - frame_df['pupil_x'].mean()) / frame_df['pupil_x'].std())
z_scores_y = np.abs((frame_df['pupil_y'] - frame_df['pupil_y'].mean()) / frame_df['pupil_y'].std())

threshold = 3

# filter out rows with outliers
frame_df = frame_df[(z_scores_x <= threshold) & (z_scores_y <= threshold)]

# reindex the rows and interpolate the pupil centers
new_indices = range(frame_df.index.min(), frame_df.index.max() + 1)
frame_df = frame_df.reindex(new_indices)

frame_df['img_path'] = frame_df['img_path'].interpolate(method='pad')
frame_df['timestamp'] = frame_df['timestamp'].interpolate(method='pad')
frame_df['row'] = frame_df['row'].interpolate(method='pad')
frame_df['col'] = frame_df['col'].interpolate(method='pad')
frame_df['frame_number'] = frame_df['frame_number'].interpolate(method='pad')
frame_df['pupil_x'] = frame_df['pupil_x'].interpolate(method='spline', order=3)
frame_df['pupil_y'] = frame_df['pupil_y'].interpolate(method='spline', order=3)
frame_df['pupil'] = frame_df[['pupil_x', 'pupil_y']].values.tolist()

new_num_frames = len(frame_df)

print("Filtered out {}/{} frames".format(original_num_frames - new_num_frames, original_num_frames))

print("Remaining frames: {}".format(new_num_frames))


# drop all events that are not in the frame_df
original_num_events = len(event_df)
event_df = event_df[event_df['frame_number'].isin(frame_df['frame_number'].values)]
new_num_events = len(event_df)


print("Filtered out {}/{} events".format(original_num_events - new_num_events, original_num_events))

print("Remaining events: {}".format(new_num_events))

# the original events are sampled at around 160us
# but we can only sample at 1ms
# so we need to bin the events into 1ms bins
# and then sample from those bins
# print("Binning events...")

# new_timestamps = []
# new_x = []
# new_y = []
# new_polarity = []
# new_frame_numbers = []
# # take every 10th event
# for i in range(0, len(event_df), 10):
#     new_timestamps.append(event_df.iloc[i]['timestamp'])
#     # for the x and y coordinates, take all of the 10 events in one list
#     x_elem = event_df.iloc[i:i+10]['x'].values
#     y_elem = event_df.iloc[i:i+10]['y'].values
#     polarity_elem = event_df.iloc[i:i+10]['polarity'].values
#     frame_number_elem = event_df.iloc[i:i+10]['frame_number'].values
#     new_x.append(x_elem)
#     new_y.append(y_elem)
#     new_polarity.append(polarity_elem)
#     new_frame_numbers.append(frame_number_elem)

# new_event_df = pd.DataFrame({'timestamp': new_timestamps, 'x': new_x, 'y': new_y, 'polarity': new_polarity, 'frame_number': new_frame_numbers})
# print(new_event_df)
    



print("========== Frames ==========")
print(frame_df)
print("========== Events ==========")
print(event_df)

event_count_df = event_df.groupby('frame_number').count().reset_index()[['frame_number', 'timestamp']]
event_count_df.columns = ['frame_number', 'event_count']
print("========== Event Count ==========")
print(event_count_df)

# we want at least 5000 events for one high density frame
# if necessary, we can add more frames

buffer_size = 5000

# iterate through the rows of event_count_df
# and accumulate the frames necessary to get at least 5000 events

frame_numbers = event_count_df['frame_number'].values
event_counts = event_count_df['event_count'].values

event_buffer = []
high_density_buffers = []
current_event_count = 0

for row in event_count_df.iterrows():
    # get the frame number
    frame_number = row[1]['frame_number']
    # get the event count
    event_count = row[1]['event_count']
    # add the frame number to the event buffer
    event_buffer.append(frame_number)
    # add the event count to the current event count
    current_event_count += event_count
    # check if the current event count is greater than the buffer size
    if current_event_count > buffer_size:
        # add all frames to the high density events
        high_density_buffers.append(event_buffer)
        # reset the event buffer
        event_buffer = []
        # reset the current event count
        current_event_count = 0


# make a 3D plot of the events with the first high densiy event
first_high_density_event = high_density_buffers[0]
first_high_density_event_df = event_df[event_df['frame_number'].isin(first_high_density_event)]

unique_frame_numbers = first_high_density_event_df['frame_number'].unique()

pupil_timestamps = frame_df[frame_df['frame_number'].isin(unique_frame_numbers)]['timestamp'].values
pupil_centers = frame_df[frame_df['frame_number'].isin(unique_frame_numbers)]['pupil'].values
pupil_x = [pupil_center[0] for pupil_center in pupil_centers]
pupil_y = [pupil_center[1] for pupil_center in pupil_centers]


# get the img paths
frame_paths = frame_df[frame_df['frame_number'].isin(unique_frame_numbers)]['img_path'].values

# get the timestamps
frame_timestamps = frame_df[frame_df['frame_number'].isin(unique_frame_numbers)]['timestamp'].values

# get the x and y coordinates
x = first_high_density_event_df['x'].values
y = first_high_density_event_df['y'].values
z = first_high_density_event_df['timestamp'].values

# get the polarity
polarity = first_high_density_event_df['polarity'].values


# plot the events
# flip along the y axis
y = FRAME_HEIGHT - y
# frame_y_coords = FRAME_HEIGHT - frame_y_coords]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(z, x, y, c=polarity, alpha=0.5)
ax.scatter(pupil_timestamps, pupil_x, pupil_y, c='r', s = 100)
ax.set_xlabel('Timestamp')
ax.set_ylabel('X')
ax.set_zlabel('Y')
plt.show()

