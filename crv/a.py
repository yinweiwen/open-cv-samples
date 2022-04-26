import sys

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

videoReader = cv.VideoCapture('ExampleVideo.avi')

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')
if int(major_ver) < 3:
    fps = videoReader.get(cv.cv.CV_CAP_PROP_FPS)
    N = videoReader.get(cv.cv.CV_CAP_PROP_FRAME_COUNT)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else:
    fps = videoReader.get(cv.CAP_PROP_FPS)
    N = videoReader.get(cv.CAP_PROP_FRAME_COUNT)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))

ret, frame = videoReader.read()
if not ret:
    sys.exit(1)

# detect mask 参数
# https://www.ccoderun.ca/programming/doxygen/opencv/classcv_1_1ORB.html#aa4e9a7082ec61ebc108806704fbd7887
mask = np.zeros(frame.shape[:2], dtype=np.uint8)
cv.rectangle(mask, (801,204), (892, 284), 255, -1)

# 点位捕获
# params for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=mask, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(frame)
N = int(N)
print('Total Frame Count:{0}'.format(N))

# numpy matlab 函数对比https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
print(p0.shape[0])
pos = np.zeros((N, p0.shape[0], 2))
pos[0] = p0[:, 0, :]
i = 0
while 1:
    i += 1
    ret, frame = videoReader.read()
    if not ret:
        print('No frames grabbed!')
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
        pos[i] = good_new
    else:
        continue
    # draw the tracks
    # for i, (new, old) in enumerate(zip(good_new, good_old)):
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #     frame = cv.circle(frame, (int(a), int(b)), 2, color[i].tolist(), -1)
    # cv.imshow('frame', frame)
    # k = cv.waitKey(10) & 0xff
    # if k == 27:
    #     break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

Disp2D=pos-pos[np.zeros(N,dtype=np.uint8),:,:]
t=np.arange(1.,N+1.)/fps
print(Disp2D)

fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Time Serial Pixels Change')
ax1.plot(t, Disp2D[:,:,0].squeeze(),linewidth=0.5,label='Disp(X)[px]')
ax2.plot(t, Disp2D[:,:,1].squeeze(),linewidth=0.5,label='Disp(Y)[px]')

print(Disp2D)
plt.show()

Disp2D_ave=np.mean(Disp2D,axis=1)
fig, (ax1, ax2) = plt.subplots(2, sharex=True)
fig.suptitle('Time Serial Pixels MEAN')
ax1.plot(t, Disp2D_ave[:,0],linewidth=0.5,label='Mean.Disp(X)[px]')
ax2.plot(t, Disp2D_ave[:,1],linewidth=0.5,label='Mean.Disp(Y)[px]')
print(Disp2D_ave)
plt.show()

# todo fft

# 像素转位移
#l = drawline; ##matlab
# estimate the scaling factor
line=np.array([[34 ,249],[847,  245]])
dist_px=line[1,1]-line[0,1]
dist_m=2.
px2m = dist_m/dist_px
# apply scale
Disp_m = px2m * Disp2D_ave
plt.title('Vertical Distance(m)')
plt.plot(t, -Disp_m[:,1],linewidth=0.5,label='Vert.Disp[m]')
plt.show()

videoReader.release()
cv.destroyAllWindows()
