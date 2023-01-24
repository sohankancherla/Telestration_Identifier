# packages required to be installed: opencv-python, numpy, imutils, beautifulsoup4
from imutils.object_detection import non_max_suppression
import numpy as np
import cv2
import os
from bs4 import BeautifulSoup
import glob

# path to images to test
rootdir = "telestration focus on text only/images/"
list_of_files = sorted(filter(os.path.isfile, glob.glob(rootdir + '*')))

boxesWithOverlap = 0
falsePositiveBoxes = 0
faseNegativeBoxes = 0

def is_overlap(boxa, boxb, thresh=0.9):
    anb = {}
    if boxa['r'] < boxb['l'] or boxa['l'] > boxb['r']:
        return False
    if boxa['t'] > boxb['b'] or boxa['b'] < boxb['t']:
        return False
    anb['l'] = max(boxa['l'], boxb['l'])
    anb['r'] = min(boxa['r'], boxb['r'])
    anb['t'] = max(boxa['t'], boxb['t'])
    anb['b'] = min(boxa['b'], boxb['b'])
    I = (anb['r']-anb['l'])*(anb['b']-anb['t'])
    U = boxa['w']*boxa['h']+boxb['w']*boxb['h'] - I
    IOU = I/U
    if IOU < thresh:
        return False
    return True

for fileName in list_of_files:
    image = cv2.imread(fileName)

    xmlFileName = fileName.rsplit('.', maxsplit=1)[0].rsplit('/', maxsplit=1)[1]

    with open('telestration focus on text only/annotations/' + xmlFileName + '.xml', 'r') as f:
        xmlData = f.read()

    soup = BeautifulSoup(xmlData, 'xml')
    xMins = [int(x.text) for x in soup.find_all('xmin')]
    yMins = [int(x.text) for x in soup.find_all('ymin')]
    xMaxs = [int(x.text) for x in soup.find_all('xmax')]
    yMaxs = [int(x.text) for x in soup.find_all('ymax')]

    print(fileName)

    orig = image.copy()
    (H, W) = image.shape[:2]

    # the image must have dimensions as multiples of 32 for EAST.
    # Grab the closest values (rounding up) from original size
    (newW, newH) = (int(W / 32) * 32, int(H / 32) * 32)

    # scaling factor saved for restoring image size after computation
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # the first is the probabilities and the
    # second will be the bounding boxes
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):

        # use probabilities and draw bounding boxes
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):

            # if our score does not have sufficient probability (0.05), ignore it
            if scoresData[x] < 0.05:
                continue

            # compute the offset factor
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to get size of box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates to list of rects
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # remove overlapping bounding
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    ocrBoxes = []
    gtBoxes = []

    for (startX, startY, endX, endY) in boxes:
        ocrBoxes.append((startX, startY, endX, endY))

    for (xMin, yMin, xMax, yMax) in zip(xMins, yMins, xMaxs, yMaxs):
        gtBoxes.append((xMin, yMin, xMax, yMax))

    # loop over the OCR bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the ratios
        startXScale = int(startX * rW)
        startYScale = int(startY * rH)
        endXScale = int(endX * rW)
        endYScale = int(endY * rH)

        # loop over the GT bounding boxes
        for (xMin, yMin, xMax, yMax) in zip(xMins, yMins, xMaxs, yMaxs):
            boxa = {'r': endXScale, 'b': endYScale, 'l': startXScale, 't': startYScale, 'w': endXScale - startXScale,
                    'h': endYScale - startYScale}

            boxb = {'r': xMax, 'b': yMax, 'l': xMin, 't': yMin, 'w': xMax - xMin, 'h': yMax - yMin}

            if is_overlap(boxa, boxb):
                boxesWithOverlap += 1
                try:
                    gtBoxes.remove((xMin, yMin, xMax, yMax))
                    ocrBoxes.remove((startX, startY, endX, endY))
                except ValueError:
                    nothing = 0

            # # determine if there is any pixel overlap between the two bounding boxes
            # areaOfOverlap = (
            #             max(0, min(xMax, endXScale) - max(xMin, startXScale)) * max(0, min(yMax, endYScale) - max(yMin,
            #                                                                                                       startYScale)))
            # # if some overlap exists
            # if areaOfOverlap > 0:
            #     # calculate the IOU overlap
            #     iouOverlapPercentage = (areaOfOverlap / ((endX - startXScale) * (endY - startYScale) + (xMax - xMin) * (
            #             yMax - yMin) - areaOfOverlap)) * 100
            #
            #     #if iouOverlap is greater than 10%, then it is a true positive
            #     if iouOverlapPercentage > tolerance:
            #         boxesWithOverlap += 1
            #         try:
            #             gtBoxes.remove((xMin, yMin, xMax, yMax))
            #             ocrBoxes.remove((startX, startY, endX, endY))
            #         except ValueError:
            #             nothing = 0

    falsePositiveBoxes += len(ocrBoxes)
    faseNegativeBoxes += len(gtBoxes)

print("Total True Positive: ", boxesWithOverlap)
print("Total False Positive: ", falsePositiveBoxes)
print("Total False Negative: ", faseNegativeBoxes)
