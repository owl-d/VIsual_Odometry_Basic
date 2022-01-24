import numpy as np
import cv2 as cv

lk_params = dict(winSize = (15,15),
                 maxLevel = 2,
                 criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

cap = cv.VideoCapture('./01_Basic_Optical_Flow/slow_traffic_small.mp4')

color = np.random.randint(0, 255, (2000, 3))

orb = cv.ORB_create(nfeatures=500)

ret, prev_frame = cap.read()
prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)

prev_keypoints = orb.detect(prev_frame, None)
prev_keypoints, prev_descriptiors = orb.compute(prev_gray, prev_keypoints)
prev_keypoints = np.array((prev_keypoints))

##### Convert ORB keypoints into numpy format (single point precision / clumn keypoint vector) #####
correct_keypoints = []
for i in range(len(prev_keypoints)):
    correct_keypoints.append([[np.float32(prev_keypoints[i].pt[0]), np.float32(prev_keypoints[i].pt[1])]])

np_prev_correct_keypoints = np.array(correct_keypoints)
####################################################################################################

mask = np.zeros_like(prev_frame)

while(1):

    ret, frame = cap.read()
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    current_keypoints, st, err = cv.calcOpticalFlowPyrLK(prev_gray, frame_gray, np_prev_correct_keypoints, None, **lk_params)

    good_new = current_keypoints[st==1]
    good_prev = np_prev_correct_keypoints[st==1]

    ### Feature Retraking : If the feature is below certain number, re-conduct feature extraction & tracking ###
    print('Current Feature Num : ', len(good_new))
    if len(good_new) <= 400:

        print('[Re-Tracking] Current Feature Num : ', len(good_new))

        prev_keypoints = orb.detect(prev_gray, None)
        prev_keypoints, prev_descriptiors = orb.compute(prev_gray, prev_keypoints)
        prev_keypoints = np.array((prev_keypoints))

        ### Convert ORB keypoints into numpy format (single point precision / column keypoint vector) ###
        correct_keypoints = []
        for i in range(len(prev_keypoints)):
            correct_keypoints.append([[np.float32(prev_keypoints[i].pt[o]), np.float32(prev_keypoints[i].pt[1])]])

        np_prev_correct_keypoints = np.array(correct_keypoints)
        #################################################################################################
        current_keypoints, st, err = cv.calcOpticalFlowPyrLK(prev_gray, frame_gray, np_prev_correct_keypoints, None, **lk_params)
        good_new = current_keypoints[st==1]
        good_prev = np_prev_correct_keypoints[st==1]
    ############################################################################################################

    for i, (new, prev) in enumerate(zip(good_new, good_prev)):
        a, b = new.ravel()
        c, d = prev.ravel()
        mask = cv.line(mask, (a,b), (c,d), color[i].tolist(), 2)
        frame = cv.circle(frame, (a,b), 5, color[i].tolist(), -1)

    img = cv.add(frame.copy(), mask)

    cv.imshow('ORB-based Optical Flow + Retracking', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    prev_gray = frame_gray.copy()
    np_prev_correct_keypoints = good_new.reshape(-1, 1, 2)