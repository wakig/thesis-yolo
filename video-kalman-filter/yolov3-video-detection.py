# -*- coding: utf-8 -*-

"""
Objects Detection on Image with YOLO v3 and OpenCV
File: yolov3-video-detection.py
"""

# Detecting Objects on Image with OpenCV deep learning library
#
# How does YOLO-v3 Algorithm works for this example case:
# STEP1: Reading input video
# STEP2: Loading YOLO v3 Network
# STEP3: Reading frames in the loop
# STEP4: Getting blob from the frame
# STEP4: Implementing Forward Pass
# STEP5: Getting Bounding Boxes
# STEP6: Non-maximum Suppression
# STEP7: Drawing Bounding Boxes with Labels
# STEP8: creating a new video by writing processed frames
#
# Result:
# New video file with Detected Objects, Bounding Boxes and Labels


# Importing needed libraries
import numpy as np
import cv2
import time
from filterpy.kalman import KalmanFilter
print (cv2.__version__)

# Initialize a Kalman filter for object tracking
def init_kalman_filter(state):
    # Define the Kalman filter with 4 state variables (x, y, x_dot, y_dot)
    # and 2 measurement variables (x, y)
    kf = KalmanFilter(dim_x=4, dim_z=2)
    
    # Define the state transition matrix F, which models constant velocity motion
    dt = 1.0  # time step
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    # Define the measurement matrix H, which maps the state to the measurement space
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    
    # Define the measurement noise covariance matrix R
    kf.R = np.array([[0.1, 0],
                     [0, 0.1]])
    
    # Define the process noise covariance matrix Q, which models system noise
    q = 0.1
    kf.Q = np.array([[q, 0, 0, 0],
                     [0, q, 0, 0],
                     [0, 0, q/10, 0],
                     [0, 0, 0, q/10]])
    
    # Initialize the state estimate
    kf.x = state.reshape(4, 1)
    
    return kf

# Define the IoU threshold for object association
IOU_THRESHOLD = 0.5

# Initialize empty list to hold object trackers
trackers = []

"""
==================     STEP1   ===================
Start of: Reading input video
"""
#NOTE:
# Defining 'VideoCapture' object and reading video from a file make sure that the path and file name is correct
video = cv2.VideoCapture('videos/test.mp4')

# Preparing variable for writer that we will use to write processed frames
writer = None

# Preparing variables for spatial dimensions of the frames
h, w = None, None

"""
End of:
Reading input video
"""

# Initialize Kalman Filter
state = np.array([0, 0, 0, 0])
kf = init_kalman_filter(state)
print("Initial: ({}, {})".format(kf.x[0][0], kf.x[1][0]))

"""
==================     STEP2  ===================
Start of: Loading YOLO v3 network
"""

# Loading COCO class labels from file and Opening file
with open('yolo-coco-data/coco.names') as f:
    # Getting labels reading every line
    # and putting them into the list
    labels = [line.strip() for line in f]


# # Check point
# print('List with labels names:')
# print(labels)

# Loading trained YOLO v3 Objects Detector with the help of 'dnn' library from OpenCV
network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                     'yolo-coco-data/yolov3.weights')

# Getting list with names of all layers from YOLO v3 network
layers_names_all = network.getLayerNames()

# # Check point
# print()
# print(layers_names_all)

# Getting only output layers' names that we need from YOLO v3 algorithm
# with function that returns indexes of layers with unconnected outputs
# print(network.getUnconnectedOutLayers())
# print(layers_names_all)
# layers_names_output = \
#     [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]
layers_names_output = \
    [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

# # Check point
# print()
# print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

# Setting minimum probability to eliminate weak predictions
probability_minimum = 0.5

# Setting threshold for filtering weak bounding boxes
# with non-maximum suppression
threshold = 0.3

# Generating colours for representing every detected object
# with function randint(low, high=None, size=None, dtype='l')
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# # Check point
# print(colours.shape)
# print(colours[0])

"""
End of:
Loading YOLO v3 network
"""


"""
==================     STEP3  ===================
Start of: Reading frames in the loop
"""

# Defining variable for counting frames at the end we will show total amount of processed frames
f = 0

# Defining variable for counting total time At the end we will show time spent for processing all frames
t = 0

# Defining loop for catching frames
while True:
    # Capturing frame-by-frame
    ret, frame = video.read()

    # If the frame was not retrieved e.g.: at the end of the video, then we break the loop
    if not ret:
        break

    # Getting spatial dimensions of the frame as we do it only once from the very beginning
    # all other frames have the same dimension
    if w is None or h is None:
        # Slicing from tuple only first two elements
        h, w = frame.shape[:2]
    
    """
    End of: Reading frame in loop
    """

    """
    ==================     STEP4  ===================
    Start of: Getting blob from current frame
    """

    # Getting blob from current frame
    # The 'cv2.dnn.blobFromImage' function returns 4-dimensional blob from current
    # frame after mean subtraction, normalizing, and RB channels swapping
    # Resulted shape has number of frames, number of channels, width and height
    # e.g.:
    # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

    """
    End of: Getting blob from current frame
    """

    """
    ==================     STEP4  ===================
    Start of: Implementing Forward pass
    """

    # Implementing forward pass with our blob and only through output layers
    # Calculating at the same time, needed time for forward pass
    network.setInput(blob)  # setting blob as input to the network
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    # Increasing counters for frames and total time
    f += 1
    t += end - start

    # Showing spent time for single current frame
    print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))
    
    """
    End of:Implementing Forward pass
    """
    
    """
    ==================     STEP5  ===================
    Start of: Getting bounding boxes
    """

    # Preparing lists for detected bounding boxes, obtained confidences and class's number
    bounding_boxes = []
    confidences = []
    classIDs = []

    # Going through all output layers after feed forward pass
    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting 80 classes' probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]
            
            # # Every 'detected_objects' numpy array has first 4 numbers with
            # # bounding box coordinates and rest 80 with probabilities
            #  # for every class
            # print(detected_objects.shape)  # (85,)

            # Eliminating weak predictions with minimum probability
            if confidence_current > probability_minimum:
                # Scaling bounding box coordinates to the initial frame size
                # YOLO data format keeps coordinates for center of bounding box
                # and its current width and height
                # That is why we can just multiply them elementwise
                # to the width and height
                # of the original frame and in this way get coordinates for center
                # of bounding box, its width and height for original frame
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Now, from YOLO data format, we can get top left corner coordinates that are x_min and y_min
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                classIDs.append(class_current)
            	
        """	
        End of: Getting bounding boxes
        """

    """
    ==================     STEP6   ===================
    Start of: Non-maximum suppression
    """

    # Implementing non-maximum suppression of given bounding boxes
    # With this technique we exclude some of bounding boxes if their
    # corresponding confidences are low or there is another
    # bounding box for this region with higher confidence

    # It is needed to make sure that data type of the boxes is 'int'
    # and data type of the confidences is 'float'
    # https://github.com/opencv/opencv/issues/12789
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)
    
    """
    End of: Non-maximum suppression
    """

    # Make KF predictions
    kf.predict()
    # print("Predict: ({}, {})".format(kf.x[0][0], kf.x[1][0]))
    # cv2.circle(frame, (int(kf.x[0][0]), int(kf.x[1][0])),
    #             10,
    #             (0,0,255), cv2.FILLED)
    # kf.predict()
    # cv2.circle(frame, (int(kf.x[0][0]), int(kf.x[1][0])),
    #             10,
    #             (0,0,255), cv2.FILLED)

    """
    ==================     STEP6   ===================
    Start of: Drawing bounding boxes and labels
    """

    cv2.putText(frame, 'Frame {}'.format(f), (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    # cv2.putText(frame, 'Predict: ({}, {})'.format(int(kf.x[0][0]), int(kf.x[1][0])), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    # Checking if there is at least one detected object after non-maximum suppression
    if len(results) > 0:
        # Going through indexes of results
        for i in results.flatten():
            # Getting current bounding box coordinates, its width and height
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            if labels[int(classIDs[i])] == 'person':
                measurement = np.array([x_min + box_width/2, y_min + box_height/2])
                kf.update(measurement)
                print("Update: ({}, {})".format(kf.x[0][0], kf.x[1][0]))
                cv2.circle(frame, (int(kf.x[0][0]), int(kf.x[1][0])),
                10,
                (255,0,0), cv2.FILLED)

                start_x = kf.x[0][0]
                start_y = kf.x[1][0]
                dx = kf.x[2][0]
                dy = kf.x[3][0]
                end_x = start_x + dx
                end_y = start_y + dy
                # m = (end_x - start_x) / (end_y - start_y)
                angle = np.rad2deg(np.arctan2(end_y - start_y, end_x - start_x))
                length = 40 # length of the arrow
                new_x = start_x + length * np.cos(angle * np.pi / 180.0)
                new_y = start_y + length * np.sin(angle * np.pi / 180.0)
                cv2.arrowedLine(frame, (int(start_x), int(start_y)), (int(new_x), int(new_y)), (255,0,0), 3)

                # Here I tried to make arrow length correspond to velocity. But it didn't look good
                # if np.sqrt(dx**2 + dy**2) <= length:
                #     cv2.arrowedLine(frame, (int(start_x), int(start_y)), (int(start_x)+int(dx), int(start_y)+int(dy)), (255,0,0), 3)
                # else:
                #     cv2.arrowedLine(frame, (int(start_x), int(start_y)), (int(new_x), int(new_y)), (255,0,0), 3)

                cv2.putText(frame, 'KF: ({}, {})'.format(int(kf.x[0][0]), int(kf.x[1][0])), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            # Preparing colour for current bounding box and converting from numpy array to list
            colour_box_current = colours[classIDs[i]].tolist()

            # print(type(colour_box_current))  # <class 'list'>
            # print(colour_box_current)  # [172 , 10, 127]

            if labels[int(classIDs[i])] == 'person':

                # Drawing bounding box on the original current frame
                cv2.rectangle(frame, (x_min, y_min),
                            (x_min + box_width, y_min + box_height),
                            (0,255,0), 2)

                # Drawing circle on center of frame
                cv2.circle(frame, (int(x_min + box_width / 2), int(y_min + box_height / 2)),
                            10,
                            (0,255,0), cv2.FILLED)

                # Preparing text with label and confidence for current bounding box
                text_box_current = '{}: {:.4f}'.format(labels[int(classIDs[i])],
                                                    confidences[i])

                # Putting text with label and confidence on the original image
                cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    """
    End of:
    Drawing bounding boxes and labels
    """
    
    
    """
    ==================     STEP7   ===================
    Start of: Writing processed frame into the file
    """

    # Initializing writer
    # we do it only once from the very beginning when we get spatial dimensions of the frames
    if writer is None:
        # Constructing code of the codec to be used in the function VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Writing current processed frame into the video file
        writer = cv2.VideoWriter('videos/result.mp4', fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

    # Write processed current frame to the file
    writer.write(frame)

"""
End of: Writing processed frame into the file
"""

"""
End of: Writing processed frame into the file
"""


# Printing final results
print()
print('Total number of frames', f)
print('Total amount of time {:.5f} seconds'.format(t))
print('FPS:', round((f / t), 1))


# Releasing video reader and writer
video.release()
writer.release()


"""
Some comments

What is a FOURCC?
    FOURCC is short for "four character code" - an identifier for a video codec,
    compression format, colour or pixel format used in media files.
    http://www.fourcc.org


Parameters for cv2.VideoWriter():
    filename - Name of the output video file.
    fourcc - 4-character code of codec used to compress the frames.
    fps	- Frame rate of the created video.
    frameSize - Size of the video frames.
    isColor	- If it True, the encoder will expect and encode colour frames.
"""
