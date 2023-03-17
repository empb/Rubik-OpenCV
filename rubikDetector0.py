#!/usr/bin/env python

# > ./rubikDetector0.py
# > ./rubikDetector0.py --dev=help

import cv2          as cv
from umucv.stream import autoStream
import numpy as np
from rubik_solver import utils
import sys

SCANKEY = 13	# RETURN key code

STRINGCOLORS = [ 'y', 'r', 'g', 'o', 'b', 'w' ]

cube = []		# Color of the faces of the cube
centers = []	# Center colors

for key, frame in autoStream():
	# Length of the frame of the cube and the stickers
	h, w, _ = frame.shape
	cubeside = (min(h, w)//2)#*1
	stickerside = cubeside//3
	# Initial point of the frame of the first sticker
	x1 = w//2 - cubeside//2
	y1 = h//2 - cubeside//2

	# List of coordinates of the upper left cornner of the frames of the sticker
	stickers = []
	for i in range(3):
		for j in range(3):
			cw = x1 + j*stickerside
			ch = y1 + i*stickerside
			stickers.append((cw, ch))

	# Draw all frames of the stickers 
	for q in stickers:
		cv.rectangle(frame, (q[0], q[1]), (q[0]+stickerside, q[1]+stickerside), color=(255, 255, 255), thickness=3)

	# Draw the arrows
	faceact = len(centers)
	if faceact == 1 or faceact == 5:
		cv.arrowedLine(frame, (x1-stickerside//2, y1 + cubeside), (x1-stickerside//2, y1), color=(255, 255, 255), thickness=3)
		cv.arrowedLine(frame, (x1+cubeside+stickerside//2, y1 + cubeside), (x1+cubeside+stickerside//2, y1), color=(255, 255, 255), thickness=3)
	else: 
		if faceact > 0:
			cv.arrowedLine(frame, (x1 + cubeside, y1-stickerside//2), (x1, y1-stickerside//2), color=(255, 255, 255), thickness=3)
			cv.arrowedLine(frame, (x1 + cubeside, y1+cubeside+stickerside//2), (x1, y1+cubeside+stickerside//2), color=(255, 255, 255), thickness=3)

	# Draw the texts
	cv.putText(frame, '[ENTER] TO SCAN', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv.LINE_AA)
	cv.putText(frame, '[ENTER] TO SCAN', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)
	cv.putText(frame, str(faceact) + '/6', (w-55, 30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv.LINE_AA)
	cv.putText(frame, str(faceact) + '/6', (w-55, 30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv.LINE_AA)

	# Calculate the centers of the frames for every sticker
	stkcenters = []
	stickerrad = stickerside//2
	for q in stickers:
		cx = q[0] + stickerrad
		cy = q[1] + stickerrad
		stkcenters.append((cx, cy))

	# Draw the centers and store the colors
	centerrad = stickerside//15
	facecolors = []
	for c in stkcenters:
		col = frame[c[1], c[0], :]
		colT = (int(col[0]), int(col[1]), int(col[2]))
		cv.rectangle(frame, (c[0]-centerrad, c[1]-centerrad), (c[0]+centerrad, c[1]+centerrad), color=colT, thickness=-1)
		facecolors.append(col)

	# When the scan key is pressed, store the colors of the 9 stickers
	# The center color is taken as the reference color for the face
	if key == SCANKEY:
		cube.append(facecolors)
		centers.append(facecolors[4])
		if len(cube) == 6:	# All faces scanned
			cube = np.int64(cube)
			cubescan = []
			for face in cube:
				difcolors = [ abs (face-c) for c in centers]
				sumcolors = [ np.sum(dif, axis=1) for dif in difcolors ]
				colorsdetected  = [ STRINGCOLORS[d] for d in np.argmin(sumcolors, axis=0) ]
				cubescan.append(colorsdetected)
			# The order of the stickers on the scanned cube has to be the same as the order used by the solver. We adapt this order:
			newcube = []
			# UP face stays the same
			newcube.append(cubescan[0])
			# The (scan) order FRONT-RIGHT-BACK-LEFT has to be LEFT-FRONT-RIGHT-BACK
			newcube.append(cubescan[4])
			newcube.append(cubescan[1])
			newcube.append(cubescan[2])
			newcube.append(cubescan[3])
			# The BOTTOM face has to rotate -90 degrees
			bottomface = (np.array(cubescan[5])).reshape(3, 3)
			bottomface[:,[0, 2]] = bottomface[:,[2, 0]]
			bottomface = bottomface.transpose()
			newcube.append((bottomface.flatten()).tolist())
			# Convert de list to string
			cubestr = ""
			for nm in newcube:
				rows = ["".join(r) for r in  nm]
				cubestr += "".join(rows)
			# Solve the cube with the library and print result
			print('Cube scanned: ' + cubestr)
			moves = utils.solve(cubestr, 'Kociemba')
			print('Solution: ' + str(moves))
			# When it solves the cube, the program ends
			sys.exit()

	cv.imshow('output', frame)

cv.destroyAllWindows()
