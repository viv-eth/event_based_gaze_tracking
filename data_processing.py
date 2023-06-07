import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt, mpld3
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
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import random


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

# add a column to event_df that indicates the frame number
frame_rate = 25 # fps
event_df['frame_number'] = event_df['timestamp'].apply(lambda x: int(x / (1000000 / frame_rate)))
frame_df['frame_number'] = frame_df['timestamp'].apply(lambda x: int(x / (1000000 / frame_rate)))

print("Performing pupil detection...")
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

print("Pupil detection complete.")

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

print("Filtering out pupil centers that are not on a black pixel...")
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

print("Interpolating pupil centers from remaining data...")

new_indices = range(original_frame_df.index.min(), original_frame_df.index.max() + 1)
frame_df = frame_df.reindex(new_indices)

pupil_nan_count = frame_df['pupil'].isna().sum()

print("Interpolating {} pupil coordinates...".format(pupil_nan_count))

frame_df['img_path'] = original_frame_df['img_path'].fillna(original_frame_df['img_path'])
frame_df['timestamp'] = original_frame_df['timestamp'].fillna(original_frame_df['timestamp'])
frame_df['row'] = original_frame_df['row'].fillna(original_frame_df['row'])
frame_df['col'] = original_frame_df['col'].fillna(original_frame_df['col'])
frame_df['frame_number'] = original_frame_df['frame_number'].fillna(original_frame_df['frame_number'])
frame_df[['pupil_x', 'pupil_y']] = pd.DataFrame(frame_df['pupil'].tolist(), index=frame_df.index)
frame_df['pupil_x'] = frame_df['pupil_x'].interpolate(method='spline', order=3)
frame_df['pupil_y'] = frame_df['pupil_y'].interpolate(method='spline', order=3)
frame_df['pupil'] = frame_df[['pupil_x', 'pupil_y']].values.tolist()

print("Filtering out outliers from the frame data...")
# Calculate z-scores for x and y coordinates
z_scores_x = np.abs((frame_df['pupil_x'] - frame_df['pupil_x'].mean()) / frame_df['pupil_x'].std())
z_scores_y = np.abs((frame_df['pupil_y'] - frame_df['pupil_y'].mean()) / frame_df['pupil_y'].std())

threshold = 3

# filter out rows with outliers
outlier_count = len(frame_df[(z_scores_x > threshold) | (z_scores_y > threshold)])
print("Filtering out {} outliers...".format(outlier_count))
frame_df = frame_df[(z_scores_x <= threshold) & (z_scores_y <= threshold)]

print("Interpolating the outliers...")
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

print("Frame data processing complete.")


# drop all events that are not in the frame_df

event_df = event_df[event_df['frame_number'].isin(frame_df['frame_number'].values)]


# store the frame_df and event_df to npy files in the experiment directory
frame_numpy = frame_df.to_numpy()
event_numpy = event_df.to_numpy()

np.save(os.path.join(experiment_path, 'frame_data.npy'), frame_numpy)
np.save(os.path.join(experiment_path, 'event_data.npy'), event_numpy)

# event_df_indices = event_df.index.values
# original_event_df = event_df.copy() 

# print("Removing outliers from the event data...")

# # Calculate z-scores for x and y coordinates
# z_scores_x = np.abs((event_df['x'] - event_df['x'].mean()) / event_df['x'].std())
# z_scores_y = np.abs((event_df['y'] - event_df['y'].mean()) / event_df['y'].std())

# threshold = 3

# # filter out rows with outliers
# outlier_count = len(event_df[(z_scores_x > threshold) | (z_scores_y > threshold)])
# print("Filtering out {} outliers...".format(outlier_count))
# event_df = event_df[(z_scores_x <= threshold) & (z_scores_y <= threshold)]

# print("Interpolating the outliers...")
# # reindex the rows and interpolate the pupil centers
# new_indices = range(event_df_indices.min(), event_df_indices.max() + 1)
# event_df = event_df.reindex(new_indices)

# event_df['timestamp'] = original_event_df['timestamp'].fillna(original_event_df['timestamp'])
# event_df['x'] = event_df['x'].interpolate(method='spline', order=3)
# event_df['y'] = event_df['y'].interpolate(method='spline', order=3)
# event_df['polarity'] = original_event_df['polarity'].fillna(original_event_df['polarity'])
# event_df['frame_number'] = original_event_df['frame_number'].fillna(original_event_df['frame_number'])

# print("Event data processing complete.")





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


if demo:
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
    ax.scatter(z, x, y, c=polarity, alpha=0.5, cmap='viridis')
    ax.scatter(pupil_timestamps, pupil_x, pupil_y, c='r', s = 100)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    plt.show()
    # mpld3.save_html(fig, "demo.html")

    # now do the same for the events and pixels using plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=z,
        y=x,
        z=y,
        mode='markers',
        marker=dict(
            size=2,
            color=polarity,                # set color to an array/list of desired values
            colorscale='viridis',   # choose a colorscale
            opacity=0.8
        ),
    ), 
    go.Scatter3d(
        x=pupil_timestamps,
        y=pupil_x,
        z=pupil_y,
        mode='markers',
        marker=dict(
            size=5,
            color='red',                # set color to an array/list of desired values
            opacity=0.8
        ),
        name='Pupil Center'
    )])
    fig.update_layout(scene = dict(
                    xaxis_title='Timestamp',
                    yaxis_title='X',
                    zaxis_title='Y'),
                    width=1000,
                    margin=dict(r=20, b=10, l=10, t=10),
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    ))

    fig.write_html("demo.html")


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pupil_timestamps, pupil_x, pupil_y, c='r', s = 100)
    ax.scatter(z, x, y, c=polarity, alpha=0.5, cmap='viridis')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')

    def rotate(angle):
        ax.view_init(azim=angle)

    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)
    rot_animation.save('rotation.gif', dpi=80, writer='pillow')

    # pick 10 random high density events
    random_high_density_events = random.sample(high_density_buffers, 10)

    # for every random event create a gif
    plot_gif = False
    if plot_gif:
        for i, event in enumerate(random_high_density_events):
            print("Creating gif for event: ", i)
            # get the unique frame numbers
            unique_frame_numbers = event

            # get the pupil centers
            pupil_timestamps = frame_df[frame_df['frame_number'].isin(unique_frame_numbers)]['timestamp'].values
            pupil_centers = frame_df[frame_df['frame_number'].isin(unique_frame_numbers)]['pupil'].values
            pupil_x = [pupil_center[0] for pupil_center in pupil_centers]
            pupil_y = [pupil_center[1] for pupil_center in pupil_centers]

            # get the img paths
            frame_paths = frame_df[frame_df['frame_number'].isin(unique_frame_numbers)]['img_path'].values

            # get the timestamps
            frame_timestamps = frame_df[frame_df['frame_number'].isin(unique_frame_numbers)]['timestamp'].values

            # get the x and y coordinates
            x = event_df[event_df['frame_number'].isin(unique_frame_numbers)]['x'].values
            y = event_df[event_df['frame_number'].isin(unique_frame_numbers)]['y'].values
            z = event_df[event_df['frame_number'].isin(unique_frame_numbers)]['timestamp'].values   
            polarity = event_df[event_df['frame_number'].isin(unique_frame_numbers)]['polarity'].values

            y = FRAME_HEIGHT - y

            # plot the events
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pupil_timestamps, pupil_x, pupil_y, c='r', s = 100)
            ax.scatter(z, x, y, c=polarity, alpha=0.5, cmap='viridis')
            ax.set_xlabel('Timestamp')
            ax.set_ylabel('X')
            ax.set_zlabel('Y')

            def rotate(angle):
                ax.view_init(azim=angle)

            rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)
            rot_animation.save('rotation_{}.gif'.format(i), dpi=80, writer='pillow')

            print("Done creating gif for event: ", i)


if plot:
    # 3D plot of the events with pupil centers
    event_x_coords = event_df['x'].values
    event_y_coords = event_df['y'].values
    event_z_coords = event_df['timestamp'].values
    polarity       = event_df['polarity'].values

    pupil_x_coords = frame_df['pupil_x'].values 
    pupil_y_coords = frame_df['pupil_y'].values  
    pupil_z_coords = frame_df['timestamp'].values.astype(int)

    # flip along the y axis
    event_y_coords = FRAME_HEIGHT - event_y_coords

    print("Drawing 3D plot...")

    fig = plt.figure()
    fig.set_size_inches(30, 20)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(event_z_coords, event_x_coords, event_y_coords, c=polarity, alpha=0.5)
    ax.scatter(pupil_z_coords, pupil_x_coords, pupil_y_coords, c='r', s = 100)
    ax.set_xlabel('Timestamp')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    # mpld3.save_html(fig, "3d_plot.html")
    plt.savefig("3d_plot.png")
