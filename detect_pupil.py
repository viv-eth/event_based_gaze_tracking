# write a description for below function
"""
    function name: 
        detect_pupil
    input: 
        frame_path (string)
        debug (boolean)
    output: 
        pupil_center (tuple)

    description:
        We load a frame from the dataset and perform several image processing techniques to detect the pupil.
        If no pupil is detected, we add a refinement step to the image processing pipeline.
        If no pupil is detected after the refinement step, we return None.

"""

import cv2
import numpy as np

def detect_pupil(frame_path, debug=False):
    frame_img = cv2.imread(frame_path)
    frame_height = frame_img.shape[0]
    frame_width = frame_img.shape[1]
    # convert to grayscale
    img_gray = cv2.cvtColor(frame_img, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    blurred = cv2.medianBlur(blurred, 5)
    edges = cv2.Canny(blurred, 20, 100)

    # connect the edges
    kernel = np.ones((7, 7), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    # smooth out gaps in the edges
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    

    if debug:
        # plot the image with the edges
        plt.imshow(edges)
        plt.title("Edges")
        plt.show()

    # try to fit an ellipse to the contour
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    cv2.drawContours(frame_img, contours, -1, (0, 255, 0), 3)

    
    if debug:
        # plot the image with the contours
        plt.imshow(frame_img)
        plt.title("Contours")
        plt.show()


    ellipses = []
    # try to fit an ellipse to the contour
    for contour in contours:
        try:
            # fit an ellipse to the contour using Hough transform
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(frame_img, ellipse, (255, 255, 0), 2)
            ellipses.append(ellipse)
        except:
            # set ellipse to None if no ellipse could be fitted
            ellipses.append(None)
            print("No ellipse could be fitted to the contour!!")


    if debug:
        # plot the image with the ellipses
        plt.imshow(frame_img)
        plt.title("All ellipses")
        plt.show()

    if ellipses not in [None, []]:
        # discard all ellipses that are None
        ellipses = [ellipse for ellipse in ellipses if ellipse is not None]
        ellipses = [ellipse for ellipse in ellipses if np.pi * ellipse[1][0] * ellipse[1][1] < 4200 and np.pi * ellipse[1][0] * ellipse[1][1] > 1200]
       
        # determine the aspect ratio of the ellipses
        aspect_ratios = [ellipse[1][0] / ellipse[1][1] for ellipse in ellipses]
        # discard ellipses with an aspect ratio greater than 1.5 and smaller than 0.5
        ellipses = [ellipse for ellipse, aspect_ratio in zip(ellipses, aspect_ratios) if aspect_ratio < 1.5 and aspect_ratio > 0.4]
        aspect_ratios = [ellipse[1][0] / ellipse[1][1] for ellipse in ellipses]

        if debug:
            if ellipses not in [None, []]:
                print("Ellipses found!!")
                # plot the image with the ellipses
                for ellipse in ellipses:
                    cv2.ellipse(frame_img, ellipse, (255, 0, 255), 2)
                plt.imshow(frame_img)
                plt.title("Filtered ellipses")
                plt.show()
            else:
                print("Further refinement needed!!")
            
    else: 
        print("Further refinement needed!!")
        
    if ellipses not in [None, []]:
        # check which ellipse has the aspect ratio closest to 1
        if len(ellipses) > 1:
            ellipse = ellipses[np.argmin(np.abs(np.array(aspect_ratios) - 1))]
        else:
            ellipse = ellipses[0]

        if debug:
            cv2.ellipse(frame_img, ellipse, (255, 0, 0), 2)
            # print parameters of the ellipse
            print("Center: ", ellipse[0])
            print("Major axis length: ", ellipse[1][0])
            print("Minor axis length: ", ellipse[1][1])
            print("Angle: ", ellipse[2])
            print("Area: ", np.pi * ellipse[1][0] * ellipse[1][1])

            # plot the image with the ellipse
            plt.imshow(frame_img)
            plt.title("Best ellipse")
            plt.show()

    else:
        if debug:
            print("Trying more refined approach...")
        # try smoothing the image more
        blurred = cv2.GaussianBlur(blurred, (5, 5), 0)
        blurred = cv2.medianBlur(blurred, 5)
        blurred = cv2.medianBlur(blurred, 5)
        edges = cv2.Canny(blurred, 20, 100)

        # connect the edges
        kernel = np.ones((10, 10), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)

        # smooth out gaps in the edges
        kernel = np.ones((10, 10), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        if debug:
            # plot the image with the edges
            plt.imshow(edges)
            plt.title("Further smoothed edges")
            plt.show()
            

        # try to fit an ellipse to the contour
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        cv2.drawContours(frame_img, contours, -1, (0, 255, 0), 3)

        if debug:
            # plot the image with the contours
            plt.imshow(frame_img)
            plt.title("Smoothed Contours")
            plt.show()

        ellipses = []
        # try to fit an ellipse to the contour
        for contour in contours:
            # fit an ellipse to the contour using Hough transform
            try:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(frame_img, ellipse, (255, 255, 0), 2)
                ellipses.append(ellipse)
            except:
                # append an ellipse whose parameters are all zero if no ellipse could be fitted
                ellipses.append(None)
                print("No ellipse could be fitted to the contour!!")

        # check if any ellipse in ellipses is None  
        ellipse_None = False
        if any (ellipse is None for ellipse in ellipses):
            ellipse_None = True
            print("Ellipse is None!!")
        
        if not ellipse_None:
            if ellipses not in [None, []]:
                if debug:
                    # plot the image with the ellipses
                    plt.imshow(frame_img)
                    plt.title("All ellipses")
                    plt.show()

                # discard all ellipses that are None
                ellipses = [ellipse for ellipse in ellipses if ellipse is not None]
                # discard ellipses with an area greater than 4000 and smaller than 1100 or if they are None
                ellipses = [ellipse for ellipse in ellipses if np.pi * ellipse[1][0] * ellipse[1][1] < 4200 and np.pi * ellipse[1][0] * ellipse[1][1] > 1200]
                # determine the aspect ratio of the ellipses
                aspect_ratios = [ellipse[1][0] / ellipse[1][1] for ellipse in ellipses]
                # discard ellipses with an aspect ratio greater than 1.5 and smaller than 0.5
                ellipses = [ellipse for ellipse, aspect_ratio in zip(ellipses, aspect_ratios) if aspect_ratio < 1.5 and aspect_ratio > 0.4]
                aspect_ratios = [ellipse[1][0] / ellipse[1][1] for ellipse in ellipses]
                # plot the image with the filtered ellipses
                if debug:
                    for ellipse in ellipses:
                        cv2.ellipse(frame_img, ellipse, (255, 0, 255), 2)
                    plt.imshow(frame_img)
                    plt.title("Filtered ellipses")
                    plt.show()

                # find the ellipse with the aspect ratio closest to 1
                if ellipses not in [None, []]:
                    # check which ellipse has the aspect ratio closest to 1
                    if len(ellipses) > 1:
                        ellipse = ellipses[np.argmin(np.abs(np.array(aspect_ratios) - 1))]
                    else:
                        ellipse = ellipses[0]
                    
                    if debug:
                        cv2.ellipse(frame_img, ellipse, (255, 0, 0), 2)
                        # print parameters of the ellipse
                        print("Center: ", ellipse[0])
                        print("Major axis length: ", ellipse[1][0])
                        print("Minor axis length: ", ellipse[1][1])
                        print("Angle: ", ellipse[2])
                        print("Area: ", np.pi * ellipse[1][0] * ellipse[1][1])

                        # plot the image with the ellipse
                        plt.imshow(frame_img)
                        plt.title("Best ellipse")
                        plt.show()
            else:
                print("Could not find any ellipses after further smoothing!!")
        else:
            ellipse = None


    return ellipse, frame_img