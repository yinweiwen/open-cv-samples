import sys

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

videoReader=cv.VideoCapture('ExampleVideo.avi')

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')

# With webcam get(CV_CAP_PROP_FPS) does not work.
# Let's see for ourselves.

if int(major_ver)  < 3 :
    fps = videoReader.get(cv.cv.CV_CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = videoReader.get(cv.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


ret,frame=videoReader.read()
if not ret:
    sys.exit(1);

# https://www.codegrepper.com/code-examples/python/python+opencv+draw+rectangle
# gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
# Adding Function Attached To Mouse Callback
drawing = False
ix = 0
iy = 0
def draw(event,x,y,flags,params):
    global ix,iy,drawing
    # Left Mouse Button Down Pressed
    if(event==cv.EVENT_LBUTTONDOWN):
        print("a")
        drawing = True
        ix = x
        iy = y
    if(event==cv.EVENT_MOUSEMOVE):
        if(drawing==True):
            print("from {0},{1} to {2},{3}".format(ix,iy,x,y))
            #For Drawing Line
            # cv.line(frame,pt1=(ix,iy),pt2=(x,y),color=(255,255,255),thickness=3)
            # For Drawing Rectangle
            cv.rectangle(frame,pt1=(ix,iy),pt2=(x,y),color=(0,0,255),thickness=3)
    if(event==cv.EVENT_LBUTTONUP):
        cv.imshow("cap",frame)
        drawing = False

cv.imshow("cap",frame)
# Adding Mouse CallBack Event
cv.setMouseCallback("cap",draw)

# todo fix here
# from 603,205 to 734,299

# detect mask 参数
# https://www.ccoderun.ca/programming/doxygen/opencv/classcv_1_1ORB.html#aa4e9a7082ec61ebc108806704fbd7887
mask=np.zeros(frame.shape[:2], dtype=np.uint8)
cv.rectangle(mask,(603,205),(734,299),255,-1)

gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB
kp = orb.detect(gray,mask=mask)
# compute the descriptors with ORB
kp, des = orb.compute(gray, kp)
# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(gray, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()

tracker_type='KCF'
if tracker_type == 'BOOSTING':
    tracker = cv.TrackerBoosting_create()
if tracker_type == 'MIL':
    tracker = cv.TrackerMIL_create()
if tracker_type == 'KCF':
    tracker = cv.TrackerKCF_create()
if tracker_type == 'TLD':
    tracker = cv.TrackerTLD_create()
if tracker_type == 'MEDIANFLOW':
    tracker = cv.TrackerMedianFlow_create()
if tracker_type == 'GOTURN':
    tracker = cv.TrackerGOTURN_create()
if tracker_type == 'MOSSE':
    tracker = cv.TrackerMOSSE_create()
if tracker_type == "CSRT":
    tracker = cv.TrackerCSRT_create()

tracker.init(frame,kp)
while True:
    ok, frame = videoReader.read()
    if not ok:
        break;

    ok,bbox=tracker.update(frame)
    print(bbox)
    # # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else :
        # Tracking failure
        cv.putText(frame, "Tracking failure detected", (100,80), cv.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display tracker type on frame
    cv.putText(frame, tracker_type + " Tracker", (100,20), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

    # Display FPS on frame
    cv.putText(frame, "FPS : " + str(int(fps)), (100,50), cv.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    # Display result
    cv.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv.waitKey(1000) & 0xff
    if k == 27 : break

k = cv.waitKey(0)

videoReader.release()
cv.destroyAllWindows()