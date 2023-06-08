import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt, mpld3
import seaborn as sns
import cv2
import sys 
# import importlib
# sys.path.insert(0, '/scratch/vivianep/event_based_gaze_tracking/visualize.py')
# importlib.reload(visualize)
# sys.path.insert(0, '/scratch/vivianep/event_based_gaze_tracking/detect_pupil.py')
# importlib.reload(detect_pupil)
import visualize
import detect_pupil as dp
import scipy
import argparse
import time
import tkinter as tk
import plotly.graph_objects as go
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import random
from rich.progress import track
from tabulate import tabulate
from rich import print
import bisect

FRAME_WIDTH = 346
FRAME_HEIGHT = 260

parser = argparse.ArgumentParser(description='Arguments for using the eye visualizer')
parser.add_argument('--subject', type=int, default=4, help='which subject to evaluate')
parser.add_argument('--eye', default=0, help='Which eye to visualize, left (0) or right (1)')
parser.add_argument('--debug', default=False, help='Whether to print debug statements')
parser.add_argument('--demo', default=False, help='Whether to plot a single high density frame for demo purposes')
parser.add_argument('--plot', default=False, help='Whether to plot the data')

opt, unknown = parser.parse_known_args()

subject = opt.subject
eye = opt.eye
debug = opt.debug
demo = opt.demo
plot = opt.plot

# load the dataset for the subject
eye_dataset = visualize.EyeDataset(os.path.join(os.getcwd(), 'eye_data'), subject)
experiment_path = os.path.join(os.getcwd(), 'eye_data/user' + str(subject) + '/' + str(eye) + '/')
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

# sort both dataframes by timestamp in descending order
frame_df = frame_df.sort_values(by=['timestamp'], ascending=False)
event_df = event_df.sort_values(by=['timestamp'], ascending=False)

# determine the frame numbers in the frame_df and event_df
frame_rate = 25 # fps
frame_df['frame_number'] = frame_df['timestamp'].apply(lambda x: int(x / (1000000 / frame_rate)))
event_df['frame_number'] = event_df['timestamp'].apply(lambda x: int(x / (1000000 / frame_rate)))
print("Events that are missing frames: ", set(event_df['frame_number'].unique()) - set(frame_df['frame_number'].unique()))
print("\n")

# dropping the rows that are missing frames
print("Dropping rows that are missing frames...\n")
event_df = event_df[event_df['frame_number'].isin(frame_df['frame_number'].unique())]

print("Performing pupil detection...\n")
# iterate over the frame_df and detect the center of the pupil in each frame
centers = []
num_frames = len(frame_df['img_path'].values)

for i, frame_path in track(enumerate(frame_df['img_path'].values), total=num_frames, description="Detecting pupil centers ... "):
    pupil = dp.pupil_detection(frame_path, plot=False, best=False, debug=False)
    if pupil is not None:
        centers.append(pupil)
    else:
        centers.append(None)

none_count = 0
for center in centers:
    if center is None:
        none_count += 1

print("Center detection successfull in {}/{} frames\n".format(len(centers) - none_count, len(centers)))

# add the centers to the frame_df
frame_df['pupil'] = centers

print("Pupil post-processing...\n")

frame_height = FRAME_HEIGHT
frame_width = FRAME_WIDTH

frame_df = dp.pupil_postprocessing(frame_df)

print("Pupil post-processing complete\n")

# get frame numbers of frame_df
frame_numbers = frame_df['frame_number'].unique()
# get frame numbers of event_df
event_frame_numbers = event_df['frame_number'].unique()

# check if they are the same
if np.array_equal(frame_numbers, event_frame_numbers):
    print("Frame numbers match\n")
else:
    print("Frame numbers do not match\n")
    print("Events that are missing frames: ", set(event_df['frame_number'].unique()) - set(frame_df['frame_number'].unique()))
    print("\n")
    print("Frames that are missing events: ", set(frame_df['frame_number'].unique()) - set(event_df['frame_number'].unique()))
    print("\n")
    print("Events with frames in frame_df: ", len(set(frame_df['frame_number'].unique()) & set(event_df['frame_number'].unique())))
    print("\n")

# add the pupil centers to the event_df based on the frame number
# repeat the pupil center if the frame number is the same
print("Adding pupil centers to event_df...\n")
event_df = event_df.merge(frame_df[['frame_number', 'pupil']], on='frame_number', how='left')
print("Pupil centers added to event_df\n")

# TODO: Check what we need from below


# the original events are sampled at around 160us
# but we can only sample at 1ms
# so we need to bin the events into 1ms bins
# and then sample from those bins

# new_event_df = event_binning(event_df)
    

if debug:

    print("[bold yellow]=[/bold yellow]" * 90 + " [bold yellow]FRAMES[/bold yellow] " + "[bold yellow]=[/bold yellow]" * 90)
    print(tabulate(frame_df.head(), headers='keys', tablefmt='psql'))
    print("[bold yellow]=[/bold yellow]" * 50 + " [bold yellow]EVENTS[/bold yellow] " + "[bold yellow]=[/bold yellow]" * 50)
    print(tabulate(event_df.head(), headers='keys', tablefmt='psql'))

# event_count_df = event_df.groupby('frame_number').count().reset_index()[['frame_number', 'timestamp']]
# buffer_size = 5000
# high_density_buffers = dp.high_density_buffers(buffer_size, event_count_df)

if demo:
    # pick a random range of events (5000 events)
    # and plot them with the pupil centers

    random_start = np.random.randint(0, len(event_df) - 5000)
    random_end = random_start + 5000

    # get the events
    random_events = event_df.iloc[random_start:random_end]

    print("Frames belonging to the random events:\n")
    print(frame_df[frame_df['frame_number'].isin(random_events['frame_number'].unique())])

    
    # 3D plot
    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.scatter3D(random_events['x'], random_events['y'], random_events['timestamp'], c=random_events['polarity'], cmap='bwr')
    ax.scatter3D(random_events['pupil'].apply(lambda x: x[0]), random_events['pupil'].apply(lambda x: x[1]), random_events['timestamp'], c='black')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('timestamp')
    plt.show()

