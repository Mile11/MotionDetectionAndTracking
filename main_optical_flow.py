##############################################
# DIGITAL IMAGE ANALYSIS AND PROCESSING
# PROJECT 2019/2020
# SIMPLE MOTION DETECTION ALGORITHM
##############################################

##############################################
# IMPORTS
##############################################

import cv2 as cv
import numpy as np
import os
import imutils


##############################################
# CORE FUNCTION (OPTICAL FLOW)
##############################################

def motion_tracking(foldPath, extension='', vidFile=True, frameDigits=3, extraPrefix='', minSize=50, tresh_val=0.8, resiz_am=400, nopad=False,
                    withTest=False, gt_foldPath='', gt_extension='', gt_extraPrefix=''):

    # What the function will return (Will not be empty only if withTest=True)
    rets = [] # Accuracy
    rets_relevant = [] # Relevant accuracy

    if not nopad:
        prefix = '0' * (frameDigits - 1)
    else:
        prefix = ''

    i = 1
    j = 1
    file = foldPath + extraPrefix + prefix + str(i) + extension

    if withTest:
        fileTest = gt_foldPath + gt_extraPrefix + prefix + str(i) + gt_extension

    cap = None

    if vidFile:
        # Load the video file
        filename = foldPath
        cap = cv.VideoCapture(filename)
        if not cap.isOpened():
            print("ERROR LOADING THE VIDEO!")
            exit(1)

    first_frame = True

    while (not vidFile and os.path.isfile(file)) or (vidFile and cap.isOpened()):

        # Read the capture frames
        # Load next frame
        if not vidFile:
            frame = cv.imread(file)

            if withTest:
                frame_gt = cv.imread(fileTest)

            i += 1
            if not nopad and i >= 10 ** j:
                j += 1
                prefix = '0' * (frameDigits - j)
            file = foldPath + extraPrefix + prefix + str(i) + extension

            if withTest:
                fileTest = gt_foldPath + gt_extraPrefix + prefix + str(i) + gt_extension

        else:
            ret, frame = cap.read()
            if not ret:
                break

        frame = imutils.resize(frame, resiz_am)

        # Grayscale the frame
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if first_frame:
            prev = frame_gray
            hsv = np.zeros_like(frame)
            hsv[..., 1] = 255
            first_frame = False
            continue

        # Calculate the dense optical flow
        flow = cv.calcOpticalFlowFarneback(prev, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        mag[mag <= tresh_val] = 0
        hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        fgMask = cv.cvtColor(cv.cvtColor(hsv, cv.COLOR_HSV2BGR), cv.COLOR_BGR2GRAY)

        # fgMask = cv.threshold(fgMask, 25, 255, cv.THRESH_BINARY)[1]
        fgMask = cv.dilate(fgMask, None, iterations=3)
        contours = cv.findContours(fgMask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        # Prepare a black image that shares the frame size
        helpBG = np.zeros(fgMask.shape)

        # Go through contours found
        for c in contours:

            # Eliminate ones that are not up to par in terms of size
            if cv.contourArea(c) < minSize:
                continue

            # Create bounding box
            (x, y, w, h) = cv.boundingRect(c)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if withTest:
                cv.rectangle(helpBG, (x, y), (x + w, y + h), 255, -1)

        # If we have ground truth, calculate directly
        if withTest:
            helpBG = helpBG // 255
            helpBG2 = np.zeros(fgMask.shape)

            # Resize the frame to make it more managable on the processing and viewing
            frame_gt = imutils.resize(frame_gt, resiz_am)
            frame_gt = cv.cvtColor(frame_gt, cv.COLOR_BGR2GRAY)

            # Extract the foreground from the ground truth
            contours = cv.findContours(frame_gt, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)

            for c in contours:
                (x, y, w, h) = cv.boundingRect(c)
                cv.rectangle(helpBG2, (x, y), (x + w, y + h), 255, -1)

            helpBG2 = helpBG2 // 255

            # Calculate the difference in surfaces
            diff_surface = np.sum(np.abs(helpBG - helpBG2))

            # Calculate the union
            overlap = helpBG + helpBG2
            overlap[overlap > 0] = 1
            overlap_surface = np.sum(overlap)

            # Calculate the accuracy based on the overlap
            if overlap_surface == 0:
                rets.append(1)
            else:
                ratio = diff_surface / overlap_surface
                rets.append(1 - ratio)
                rets_relevant.append(1 - ratio)

        cv.imshow('Frame', frame)
        cv.imshow('FG Mask2', fgMask)

        if withTest:
            cv.imshow('GT bounding boxes', helpBG2)

        # Stop the algorithm with the 'q' key
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

        prev = frame_gray

    return np.array(rets), np.array(rets_relevant)


if __name__ == '__main__':
    motion_tracking('./LASIESTA/I_SI_01/', extension='.bmp', vidFile=False, extraPrefix='I_SI_01-', minSize=2500, nopad=True)
    #motion_tracking('./LASIESTA/I_SI_02/', extension='.bmp', vidFile=False, extraPrefix='I_SI_02-', minSize=2000, nopad=True)
    #motion_tracking('./LASIESTA/O_CL_01/', extension='.bmp', vidFile=False, extraPrefix='O_CL_01-', minSize=2000, nopad=True)
    #motion_tracking('./LASIESTA/O_CL_02/', extension='.bmp', vidFile=False, extraPrefix='O_CL_02-', minSize=1000, nopad=True)
    #motion_tracking('./LASIESTA/O_RA_01/', extension='.bmp', vidFile=False, extraPrefix='O_RA_01-', minSize=2000, nopad=True)
    #motion_tracking('./LASIESTA/O_RA_02/', extension='.bmp', vidFile=False, extraPrefix='O_RA_02-', minSize=2000, nopad=True)
    #motion_tracking('./LASIESTA/O_MC_01/', extension='.bmp', vidFile=False, extraPrefix='O_MC_01-', minSize=2000, nopad=True)