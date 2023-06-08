import pandas as pd
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import scipy
from scipy import stats 
import seaborn as sns

def pupil_detection_refinement(frame_path: str, plot: bool, best: bool, debug: bool):
    """
    Refines the pupil detection by applying edge detection and ellipse fitting to the image.

    Parameters
    ----------
    frame_path : str
        Path to the frame image
    plot : bool
        If True, plots the image with the detected ellipses
    best : bool
        If True, returns the ellipse that fits the pupil best
    debug : bool
        If True, prints debug statements

    Returns
    -------
    ellipse : tuple
        The ellipse that fits the pupil best
    """

    if debug:
        print("Starting pupil detection refinement")
    ellipse = None
    ellipses = []

    rgb_img = cv2.imread(frame_path)
    frame_height = rgb_img.shape[0]
    frame_width = rgb_img.shape[1]

    # Convert to grayscale
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to grayscale image
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    gray_img = clahe.apply(gray_img)

    if plot:
        plt.imshow(gray_img, cmap='gray')
        plt.title("New High-contrast grayscale image")
        plt.show()

    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    blurred = cv2.medianBlur(blurred, 5)
    # blurred = np.uint8(blurred)
    # blurred = cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    # blurred = cv2.equalizeHist(blurred)
    # blurred = cv2.fastNlMeansDenoising(blurred, None, 10, 7, 21)

    edges = cv2.Canny(blurred, 20, 60)

    # connect broken edges
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # smooth out gaps in edges
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    if plot:
        plt.imshow(edges, cmap='gray')
        plt.title("Refined Edge detection")
        plt.show()

    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    if plot:
        # plot the image with all contours
        rgb_img = cv2.drawContours(rgb_img, contours, -1, (0, 255, 0), 3)
        plt.imshow(rgb_img)
        plt.title("All contours found after refinement")
        plt.show()

    # Fit ellipse to each contour
    for contour in contours:
        ellipse = cv2.fitEllipse(contour)
        ellipses.append(ellipse)

    if plot:
        for ellipse in ellipses:
            rgb_img = cv2.ellipse(rgb_img, ellipse, (0, 0, 255), 2)
        plt.imshow(rgb_img)
        plt.title("All ellipses found after refinement")
        plt.show()

    # count None values
    none_count = 0
    for ellipse in ellipses:
        if ellipse is None:
            none_count += 1

    num_ellipses = len(ellipses) - none_count

    if debug:
        print("Number of ellipses found after refinement: {}".format(num_ellipses))

    if plot:
        circle_colors = []
        pixel_vals = []
        for ellipse in ellipses:
            if ellipse is not None:
                # check if the pixel value is inside the image
                x_coord = int(ellipse[0][0]) - 1
                y_coord = int(ellipse[0][1]) - 1
                x_range = range(0, frame_width)
                y_range = range(0, frame_height)
                if debug:
                    print("[x, y] = [{}, {}]".format(x_coord, y_coord))
                    print("x_range: ", x_range)
                    print("y_range: ", y_range)
                if x_coord not in x_range or y_coord not in y_range:
                    ellipse = None
                else: 
                    pixel_value = gray_img[int(ellipse[0][1]) - 1, int(ellipse[0][0]) - 1]
                    pixel_vals.append(pixel_value)
                    if debug:
                        print("Ellipse center x coordinate: ", x_coord)
                        print("Ellipse center y coordinate: ", y_coord)
                    cv2.ellipse(rgb_img, ellipse, (255, 255, 0), 2)
                    r = random.randint(0, 255)
                    g = random.randint(0, 255)
                    b = random.randint(0, 255)
                    circle_colors.append((r, g, b))
                    cv2.circle(rgb_img, (int(ellipse[0][0]), int(ellipse[0][1])), 2, (r, g, b), 3)
        # create a legend for the circles with the circle colors
        patches = []
        for color, px in zip(circle_colors, pixel_vals):
            # scale color values to [0, 1] range for matplotlib
            color = (color[0] / 255, color[1] / 255, color[2] / 255)
            patches.append(mpatches.Patch(color=color, label=str(px)))
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.imshow(rgb_img)
        plt.title("All ellipses after refinement with center points")
        plt.show()

    # replace ellipses with None if the pixel value is too high
    for i, ellipse in enumerate(ellipses):
        if ellipse is not None:
            # check if the pixel value is inside the image
            x_coord = int(ellipse[0][0]) - 1
            y_coord = int(ellipse[0][1]) - 1
            x_range = range(0, frame_width)
            y_range = range(0, frame_height)
            if debug:
                print("[x, y] = [{}, {}]".format(x_coord, y_coord))
                print("x_range: ", x_range)
                print("y_range: ", y_range)
            if x_coord not in x_range or y_coord not in y_range:
                ellipses[i] = None
            else: 
                pixel_value = gray_img[int(ellipse[0][1]) - 1, int(ellipse[0][0]) - 1]
                if pixel_value > 20:
                    ellipses[i] = None

    # count None values
    none_count = 0
    for ellipse in ellipses:
        if ellipse is None:
            none_count += 1

    remaining_ellipses = len(ellipses) - none_count

    if debug:
        print("Number of ellipses after refinement with pixel value > 20: {}".format(remaining_ellipses))

    if plot and remaining_ellipses > 0:
        rgb_img = cv2.imread(frame_path)
        # show the remaining ellipses
        for ellipse in ellipses:
            if ellipse is not None:
                cv2.ellipse(rgb_img, ellipse, (0, 0, 255), 2)
        plt.imshow(rgb_img)
        plt.title("Ellipses after refinement with pixel value > 20")
        plt.show()

    # drop all ellipses with None values
    ellipses = [ellipse for ellipse in ellipses if ellipse is not None]

    if remaining_ellipses > 1:
        # pick the ellipse with the smallest area
        areas = []
        for ellipse in ellipses:
            areas.append(ellipse[1][0] * ellipse[1][1] * math.pi)
        min_area = min(areas)
        min_area_index = areas.index(min_area)
        ellipse = ellipses[min_area_index]
        if debug:
            print("Ellipse with smallest area: {}".format(ellipse))
            print("Area: {}".format(min_area))
            print("Center: {}".format(ellipse[0]))

    elif remaining_ellipses == 1:
        # pick the only ellipse
        ellipse = ellipses[0]
        if debug:
            print("Ellipse with smallest area: {}".format(ellipse))
            print("Area: {}".format(ellipse[1][0] * ellipse[1][1] * math.pi))
            print("Center: {}".format(ellipse[0]))

    if best:
        # plot the image with the best ellipse
        rgb_img = cv2.imread(frame_path)
        cv2.ellipse(rgb_img, ellipse, (255, 0, 0), 2)
        plt.imshow(rgb_img)
        plt.title("Best ellipse found after refinement")
        plt.show()

    return ellipse


    

def pupil_detection(frame_path: str, plot: bool, best: bool, debug: bool):
    """
    Detects the pupil in a given frame.

    Parameters
    ----------
    frame_path : str
        Path to the frame to be processed.
    plot : bool
        If True, the results of the pupil detection will be plotted.
    best : bool
        If True, the best ellipse will be plotted.
    debug : bool
        If True, debug information will be printed.

    Returns
    -------
    ellipse : tuple
        The ellipse that best fits the pupil.
    """
    
    ellipse = None
    ellipses = []

    # Load image
    rgb_img = cv2.imread(frame_path)
    frame_height = rgb_img.shape[0] # y axis
    frame_width = rgb_img.shape[1] # x axis

    # Convert to grayscale
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to grayscale image
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8, 8))
    gray_img = clahe.apply(gray_img)

    if plot:
        plt.imshow(gray_img, cmap='gray')
        plt.title("High-contrast grayscale image")
        plt.show()

    # Apply NLM denoising
    gray_img = cv2.fastNlMeansDenoising(gray_img, None, 10, 7, 21)

    # Perform edge detection
    edges = cv2.Canny(gray_img, 20, 90)

    # connect broken edges
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # smooth out gaps in edges
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    if plot:
        plt.imshow(edges, cmap='gray')
        plt.title("Edge detection")
        plt.show()

    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    if plot:
        # plot the image with all contours
        rgb_img = cv2.drawContours(rgb_img, contours, -1, (0, 255, 0), 3)
        plt.imshow(rgb_img)
        plt.title("All contours found")
        plt.show()


    for contour in contours:
        try: 
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(rgb_img, ellipse, (255, 255, 0), 2)
            ellipses.append(ellipse)
        except:
            ellipses.append(None)
            if debug:
                print("Ellipse fitting failed")

    if plot:
        plt.imshow(rgb_img)
        plt.title("All ellipses after contour fitting")
        plt.show()

    # count None values
    none_count = 0
    for ellipse in ellipses:
        if ellipse is None:
            none_count += 1

    num_ellipses = len(ellipses) - none_count

    if debug:
        print("Number of ellipses found: {}".format(num_ellipses))

    if plot:
        circle_colors = []
        pixel_vals = []
        for ellipse in ellipses:
            if ellipse is not None:
                # check if the pixel value is inside the image
                x_coord = int(ellipse[0][0]) - 1
                y_coord = int(ellipse[0][1]) - 1
                x_range = range(0, frame_width)
                y_range = range(0, frame_height)
                if debug:
                    print("[x, y] = [{}, {}]".format(x_coord, y_coord))
                    print("x_range: ", x_range)
                    print("y_range: ", y_range)
                if x_coord not in x_range or y_coord not in y_range:
                    ellipse = None
                else: 
                    pixel_value = gray_img[int(ellipse[0][1]) - 1, int(ellipse[0][0]) - 1]
                    pixel_vals.append(pixel_value)
                    if debug:
                        print("Ellipse center x coordinate: ", x_coord)
                        print("Ellipse center y coordinate: ", y_coord)
                    cv2.ellipse(rgb_img, ellipse, (255, 255, 0), 2)
                    r = random.randint(0, 255)
                    g = random.randint(0, 255)
                    b = random.randint(0, 255)
                    circle_colors.append((r, g, b))
                    cv2.circle(rgb_img, (int(ellipse[0][0]), int(ellipse[0][1])), 2, (r, g, b), 3)
        # create a legend for the circles with the circle colors
        patches = []
        for color, px in zip(circle_colors, pixel_vals):
            # scale color values to [0, 1] range for matplotlib
            color = (color[0] / 255, color[1] / 255, color[2] / 255)
            patches.append(mpatches.Patch(color=color, label=str(px)))
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.imshow(rgb_img)
        plt.title("All ellipses with center points")
        plt.show()

    # replace ellipses with None if the pixel value is too high
    for i, ellipse in enumerate(ellipses):
        if ellipse is not None:
            # check if the pixel value is inside the image
            x_coord = int(ellipse[0][0]) - 1
            y_coord = int(ellipse[0][1]) - 1
            x_range = range(0, frame_width)
            y_range = range(0, frame_height)
            if debug:
                print("[x, y]: [{}, {}]".format(x_coord, y_coord))
                print("x_range: ", x_range)
                print("y_range: ", y_range)
            if x_coord not in x_range or y_coord not in y_range:
                ellipses[i] = None
            else: 
                pixel_value = gray_img[int(ellipse[0][1]) - 1, int(ellipse[0][0]) - 1]
                if pixel_value > 20:
                    ellipses[i] = None

    # count None values
    none_count = 0
    for ellipse in ellipses:
        if ellipse is None:
            none_count += 1

    remaining_ellipses = len(ellipses) - none_count

    if debug:
        print("Number of ellipses with pixel value > 20: {}".format(remaining_ellipses))

    if plot and remaining_ellipses > 0:
        rgb_img = cv2.imread(frame_path)
        # show the remaining ellipses
        for ellipse in ellipses:
            if ellipse is not None:
                cv2.ellipse(rgb_img, ellipse, (255, 0, 255), 2)
        plt.imshow(rgb_img)
        plt.title("Ellipses with pixel value > 20")
        plt.show()

    # drop all ellipses with None values
    ellipses = [ellipse for ellipse in ellipses if ellipse is not None]

    if remaining_ellipses == 0:
        if debug:
            print("Refinement needed... ")
        ellipse = pupil_detection_refinement(frame_path, plot=plot, best=best, debug=debug)


    elif remaining_ellipses > 1:
        # pick the ellipse with the smallest area
        areas = []
        for ellipse in ellipses:
            areas.append(ellipse[1][0] * ellipse[1][1] * math.pi)
        min_area = min(areas)
        min_area_index = areas.index(min_area)
        ellipse = ellipses[min_area_index]
        if debug:
            print("Ellipse with smallest area: {}".format(ellipse))
            print("Area: {}".format(min_area))
            print("Center: {}".format(ellipse[0]))

    elif remaining_ellipses == 1:
        # pick the only ellipse
        ellipse = ellipses[0]
        if debug:
            print("Ellipse with smallest area: {}".format(ellipse))
            print("Area: {}".format(ellipse[1][0] * ellipse[1][1] * math.pi))
            print("Center: {}".format(ellipse[0]))

    if best and remaining_ellipses >= 1:
        # plot the image with the best ellipse
        rgb_img = cv2.imread(frame_path)
        cv2.ellipse(rgb_img, ellipse, (255, 0, 0), 2)
        plt.imshow(rgb_img)
        plt.title("Best ellipse found")
        plt.show()

    return ellipse

def pupil_postprocessing(frame_df: pd.DataFrame) -> pd.DataFrame:
    """
    Postprocess the pupil detection results by removing outliers and smoothing the pupil center coordinates.

    Parameters
    ----------
    frame_df : pd.DataFrame
        DataFrame containing the pupil center coordinates for each frame

    Returns
    -------
    pd.DataFrame
        DataFrame containing the postprocessed pupil center coordinates for each frame
    """
    # interpolate missing values in the pupil column
    print("Interpolating {} missing values in the pupil column...".format(frame_df["pupil"].isna().sum()))
    
    x_coords = frame_df['pupil'].apply(lambda coord: coord[0][0] if coord is not None else None).values
    y_coords = frame_df['pupil'].apply(lambda coord: coord[0][0] if coord is not None else None).values

    # interpolate missing values in the x and y coordinates
    x_coords = pd.Series(x_coords).interpolate(method='spline', order=3).values
    y_coords = pd.Series(y_coords).interpolate(method='spline', order=3).values

    # replace the pupil column with the interpolated values
    frame_df['pupil'] = list(zip(x_coords, y_coords))

    print("There are {} missing values in the pupil column after interpolation.".format(frame_df["pupil"].isna().sum()))

    # remove outliers using z-score in the pupil column
    threshold = 3

    x_coords = frame_df['pupil'].apply(lambda coord: coord[0] if coord is not None else None).values
    y_coords = frame_df['pupil'].apply(lambda coord: coord[1] if coord is not None else None).values

    # calculate z-score for each pupil coordinate
    z_score_x = np.abs(stats.zscore(frame_df['pupil'].apply(lambda coord: coord[0] if coord is not None else None).values))
    z_score_y = np.abs(stats.zscore(frame_df['pupil'].apply(lambda coord: coord[1] if coord is not None else None).values))

    print("Removing {} outliers in the pupil column...".format(len(np.where(z_score_x > threshold)[0]) + len(np.where(z_score_y > threshold)[0])))

    # remove outliers
    pupil_x = frame_df['pupil'].apply(lambda coord: coord[0] if coord is not None else None).values
    pupil_y = frame_df['pupil'].apply(lambda coord: coord[1] if coord is not None else None).values

    # replace outliers with None
    pupil_x[np.where(z_score_x > threshold)] = None
    pupil_y[np.where(z_score_y > threshold)] = None

    # replace the pupil column with the outlier-free values
    frame_df['pupil'] = list(zip(pupil_x, pupil_y))

    # apply interpolation again to fill the missing values
    x_coords = frame_df['pupil'].apply(lambda coord: coord[0] if coord is not None else None).values
    y_coords = frame_df['pupil'].apply(lambda coord: coord[1] if coord is not None else None).values

    # interpolate missing values in the x and y coordinates
    x_coords = pd.Series(x_coords).interpolate(method='spline', order=3).values
    y_coords = pd.Series(y_coords).interpolate(method='spline', order=3).values

    # replace the pupil column with the interpolated values
    frame_df['pupil'] = list(zip(x_coords, y_coords))


    return frame_df


def high_density_buffer(buffer_size: int, event_count_df: pd.DataFrame):
    """
    Create a buffer of frames with high event density.

    Parameters
    ----------
    buffer_size : int
        The size of the buffer in frames
    event_count_df : pd.DataFrame
        DataFrame containing the event count for each frame

    Returns
    -------
    list
        List of lists containing the frame numbers of the high density buffers
    """

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

    return high_density_buffers


def event_binning(event_df: pd.DataFrame):
    """
    Bin the events into samples of 10 events each.

    Parameters
    ----------
    event_df : pd.DataFrame
        DataFrame containing the events

    Returns
    -------
    pd.DataFrame
        DataFrame containing the binned events
    """
    print("Binning events...")    

    new_timestamps = []
    new_x = []
    new_y = []
    new_polarity = []
    new_frame_numbers = []
    # take every 10th event
    for i in range(0, len(event_df), 10):
        new_timestamps.append(event_df.iloc[i]['timestamp'])
        # for the x and y coordinates, take all of the 10 events in one list
        x_elem = event_df.iloc[i:i+10]['x'].values
        y_elem = event_df.iloc[i:i+10]['y'].values
        polarity_elem = event_df.iloc[i:i+10]['polarity'].values
        frame_number_elem = event_df.iloc[i:i+10]['frame_number'].values
        new_x.append(x_elem)
        new_y.append(y_elem)
        new_polarity.append(polarity_elem)
        new_frame_numbers.append(frame_number_elem)

    new_event_df = pd.DataFrame({'timestamp': new_timestamps, 'x': new_x, 'y': new_y, 'polarity': new_polarity, 'frame_number': new_frame_numbers})
    
    return new_event_df
