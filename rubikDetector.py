#!/usr/bin/env python

# > python3.8 rubikDetector.py
# > python3.8 rubikDetector.py --dev=help
# > python3.8 rubikDetector.py --dev=dir:imagenes/test*.png 

import cv2               as cv
from umucv.stream import autoStream
from umucv.util import Help
import numpy             as np
from rubik_solver import utils

###############################################################################
#                           CONSTANTS VALUES
###############################################################################

### SHAPE DETECTION CONSTANTS:
# Epsilon for aprroxPolyDP
EPSILON_POLY = 2
# Initial border dilatation value
INIT_BORDER_DIL = 5
# Maximum distance to the model
MAX_DIST_MODEL = 0.3
# Model image file
MODEL_FILE = 'models/model1.png'
# Mininimun size which sticker is detected (in pixels)
MIN_SIZE_STICKER = 30

### GRID CONSTANTS:
# Size of the sticker and cube in the grid (in pixels)
GRID_SIZE = 50
G_CUBE_S = GRID_SIZE*3
# Grid margin size and grid background color
GRID_MARGIN = 20
GRID_BACK_COL = [226, 211, 163]
GRID_TEXT_COL = [30, 30, 30]

### COLOR CLASSIFICATION CONSTANTS:
# Threshold of when two colours are different. Measured in absolute difference
SIM_COLOR_THRES = 150
# String which represent the six color of the faces of the cube
STRING_COLORS = [ 'y', 'r', 'g', 'o', 'b', 'w' ]

### SHOW CONSTANTS:
# Show more information
SHOW_CONTOURS, SHOW_BOXES = False, True
# Show windows
SHOW_CANNY, SHOW_MODEL, SHOW_GRID = False, False, True

# Initial value of the frames delayed when face is scanned
INIT_FRAMES_DELAY = 45

###############################################################################
#                           AUXILIAR FUNCTIONS
###############################################################################

def sort_points(points):
    """
    @brief Sort the 9 points of the coordinates of a face detected.

    This funcion takes un ondered list of nine points (x, y). Sort the dots in
    relation to the positions of the stickers on one side of the rubik's cube.

    @type points:   List of nine integer duples (x, y)
    @param points:  Coordinates of the points unordered
   
    @rtype:   List of nine integer duples (x, y)
    @return:  Coordinates of the points unordered
    """
    points_sorted = []
    # Points sorted by y coordinate
    uppest = sorted(points, key = lambda p: p[1])
    for i in range(3):
        # Sort the three "uppest points" by x coordinate
        uppest_sorted = sorted(uppest[0:3], key = lambda p: p[0])
        # Add points to points_sorted and remove them from uppest
        points_sorted, uppest = add_points(points_sorted, uppest, uppest_sorted)
    return points_sorted

def add_points(newlist, oldlist, elems):
    """
    @brief Adds "elems" to "newlist" and removes them from "oldlist".
    Lists must be of the same type.

    @param newlist:  List to which the elements are added
    @param oldlist:  List to which the elements are removed
   
    @return:  Updated lists of "newlist" and "oldlist"
    """
    for elem in elems:
        newlist.append(elem)
        oldlist.remove(elem)
    return newlist, oldlist

def isPointInRectangles(point, rectangles):
    """
    Check if "point" is contained inside one of the rectangles given as a list.
    This test is based on a comparison of coordinates. 

    @type point:        Integer duple (x, y)
    @param point:       Coordinates of the point
    @type rectangles:   List with elements of the form [(x0, y0), (x1, y1)]
    @param rectangles:  List of rectangles given by their start and end point
   
    @rtype:     Boolean
    @return:    True if point is contained inside at least in one rectangle
                False otherwise
    """
    for rect in rectangles:
        # Compare the point and rectangle coordinates
        if rect[0][0] <= point[0] <= rect[1][0] and \
           rect[0][1] <= point[1] <= rect[1][1]:
            return True
    return False

def getPointsInRectangle(points, rect):
    """
    @brief Returns the subset of "points" which are inside the given rectangle.
    This test is based on a comparison of coordinates.

    @type points:   List of integer duples (x, y)
    @param points:  List of coordinates of the points
    @type rect:     Rectangle of the form [(x0, y0), (x1, y1)]
    @param rect:    Rectangle given by their start and end point
   
    @rtype:     List of integer duples (x, y)
    @return:    The subset of points which are included in te rectangle
    """
    points_inside = []
    for point in points:
        # Compare the point and rectangle coordinates
        if rect[0][0] <= point[0] <= rect[1][0] and \
           rect[0][1] <= point[1] <= rect[1][1]:
            points_inside.append(point)
    return points_inside

def getReasonablePoints(points, radiuses):
    """
    @brief Find nine close points of "points"

    The lists "points" and "radiuses" defines a list of bounding boxes which 
    center is given in "points" and the radius is given in "radiuses".
    The idea is to find a box which is closely surrounded by another 8 boxes.
    If box is found, it's returned with a big box which surroundes the 9 boxes.

    @type points:       List of integer duples (x, y)
    @param points:      List of coordinates of the center points of the boxes
    @type radiuses:     List of integers
    @param radiuses:    List of the radiuses of the boxes
   
    @rtype:     1. List of integer duples (x, y)
                2. Rectangle of the form [(x0, y0), (x1, y1)]
    @return:    1. The nine points which fits the condition
                2. A bounding box for boxes of the nine points returned
    """
    # It there are less than 9 points
    if len(points) < 9:
        return [], []
    # There are enough points
    for i in range(len(points)):
        p = points[i]
        # Radius in which we are going to look for the points
        big_rad = radiuses[i]*3.5
        big_square = [(int(p[0]-big_rad), int(p[1]-big_rad)), 
                      (int(p[0]+big_rad), int(p[1]+big_rad))]
        # Points inside the big square
        points_inside = getPointsInRectangle(points, big_square)
        # Returns if eight points are arranged in a square around p
        if len(points_inside) == 9:
            return points_inside, big_square
    # If there is no candidate
    return [], []

def simColor(color1, color2):
    """
    @brief Compares if color1 and color2 are simliar.

    The measure used is the sum of the absolute values of the differences.
    This value is compared with SIM_COLOR_THRES constant.

    @type color1:   List of integer
    @param color1:  First color to be compared
    @type color2:   List of integer
    @param color2:  Second color to be compared
   
    @rtype:     Boolean
    @return:    True if both colors are similar
                False otherwise
    """
    # Colors are similar if measure is less than the threshold
    return sum(abs(color1-color2)) < SIM_COLOR_THRES

def drawTextS(img, text, org, fontScale, color, thickness):
    """
    @brief Write text using OpenCV putText funtion.

    Params names and types are the same as OpenCV putText function.
    """
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, fontScale, \
               color, thickness, cv.LINE_AA)

def drawTextC(img, text, org, fontScale, color, thickness):
    """
    @brief Write text with countour using OpenCV putText funtion.

    Params names and types are the same as OpenCV putText function.
    """
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, fontScale, \
               (0, 0, 0), thickness*2, cv.LINE_AA)
    cv.putText(img, text, org, cv.FONT_HERSHEY_SIMPLEX, fontScale, \
               color, thickness, cv.LINE_AA)

def drawGrid(cube, moves):
    """
    @brief Creates and image of the six faces of the cube.

    The faces contain the color of the faces already scanned.
    If cube can be solved, write the moves on the grid too.

    @type cube:   List of the faces of the cube (each face is a list of colors)
    @param cube:  Cube colors already scanned 
    @type moves:  List of strings
    @param moves: Moves to solve the cube
   
    @rtype:     Image (OpenCV format)
    @return:    The image of the cube colors grid
    """
    ## Replace pixels of the cube buy GRID_SIZE x GRID_SIZE images
    cube_grid = []
    # Runs through the faces
    for face in cube:
        face_grid = []
        # For every color, creates the image
        for color in face:
            x = np.zeros([GRID_SIZE, GRID_SIZE, 3], np.uint8)
            x[:,:] = color
            face_grid.append(x)
        # Create the face image combining the nine color grids
        f1 = np.hstack([face_grid[0], face_grid[1], face_grid[2]])
        f2 = np.hstack([face_grid[3], face_grid[4], face_grid[5]])
        f3 = np.hstack([face_grid[6], face_grid[7], face_grid[8]])
        face_grid = np.vstack([f1, f2, f3])
        # Add to the cube grid
        cube_grid.append(face_grid)

    ## Create the result image
    interior = np.zeros([G_CUBE_S*3, G_CUBE_S*4, 3], np.uint8)
    # Background color
    interior[:, :] = GRID_BACK_COL
    # Draw the grid colors on the image
    for i in range(len(cube_grid)):
        if i == 0:
            interior[0:G_CUBE_S, 0:G_CUBE_S] = cube_grid[0]
        elif i == 5:
            interior[G_CUBE_S*2:G_CUBE_S*3, G_CUBE_S*3:G_CUBE_S*4] = cube_grid[5]
        else:
            interior[G_CUBE_S:G_CUBE_S*2, (i-1)*G_CUBE_S:i*G_CUBE_S] = cube_grid[i]
    
    ## Add a border to the image (just for aesthetic reasons)
    grid_frame = cv.copyMakeBorder(interior, GRID_MARGIN, GRID_MARGIN, GRID_MARGIN,
                    GRID_MARGIN, cv.BORDER_CONSTANT, 20, GRID_BACK_COL)

    ## We need to draw the black frame of the cube. 
    # First we get the initial points of each face.
    # Initial point of the first face (with the margin)
    ipoints = [(GRID_MARGIN, GRID_MARGIN)]
    # Initial points of the four middle faces
    for i in range(1,5):
        ipoints.append(((i-1)*G_CUBE_S+GRID_MARGIN, G_CUBE_S+GRID_MARGIN))
    # Initial point of the last face
    ipoints.append((G_CUBE_S*3+GRID_MARGIN, G_CUBE_S*2+GRID_MARGIN))

    ## Draw the 3x3 black frames
    for ip in ipoints:
        # List of coordinates of the upper left cornner of the frames of the stickers
        stickers = []
        for i in range(3):
            for j in range(3):
                cw = ip[0] + j*GRID_SIZE
                ch = ip[1] + i*GRID_SIZE
                stickers.append((cw, ch))
        # Draw the face black frame
        for q in stickers:
            cv.rectangle(grid_frame, (q[0], q[1]), (q[0]+GRID_SIZE, q[1]+GRID_SIZE),
                        color=(0, 0, 0), thickness=2)
        # Draw the outer border (it's wider)
        cv.rectangle(grid_frame, (ip[0], ip[1]), (ip[0]+G_CUBE_S, ip[1]+G_CUBE_S),
                        color=(0, 0, 0), thickness=4)

    ## Draw the text on the image (solution of the cube or error message)
    # If there is a solution, write the solution
    if moves != []: 
        # Postion of the text
        pos = (GRID_MARGIN, G_CUBE_S*2+GRID_SIZE*2)
        drawTextS(grid_frame, 'MOVES: ', (pos[0], pos[1]-30), 0.8, GRID_TEXT_COL, 2)
        # Split the moves to fit text in the window
        line_w = 9
        moves_splited = [ moves[x:x+line_w] for x in range(0, len(moves), line_w)]
        # Draw the texts line by line
        for i, sub_moves in enumerate(moves_splited):
            new_pos = (pos[0], pos[1] + i*30)
            # Checks if is the last line to write a dot instead of a comma
            if i < len(moves_splited)-1:
                drawTextS(grid_frame, str(sub_moves).strip("[]") + ',', new_pos,
                         0.8, GRID_TEXT_COL, thickness=1)
            else:
                drawTextS(grid_frame, str(sub_moves).strip("[]") + '.', new_pos,
                         0.8, GRID_TEXT_COL, thickness=1)
    elif len(cube) == 6:  # Solving error (moves are empty but all faces scanned)
            drawTextS(grid_frame, 'ERROR while scanning the cube.', 
                    (GRID_MARGIN, G_CUBE_S*2+GRID_SIZE*2), 0.75, GRID_TEXT_COL, 2)
            drawTextS(grid_frame, 'Duplicated pieces.', 
                    (GRID_MARGIN, G_CUBE_S*2+GRID_SIZE*3), 0.75, GRID_TEXT_COL, 2)
    else:   # Faces not scanned yet
        drawTextS(grid_frame, 'FACES: ' + str(len(cube)) + '/6',
                 (GRID_MARGIN, G_CUBE_S*2+GRID_SIZE*2), 0.75, GRID_TEXT_COL, 2)
    # Return the result image
    return grid_frame

###############################################################################
#                           OPENCV LAYOUT
###############################################################################

## Window creation and placement
cv.namedWindow('color grid')
cv.moveWindow('color grid', 700, 100)
cv.namedWindow('input')
cv.moveWindow('input', 0, 100)

## TRACKBARS
# Auxiliar function for trackbars
def nothing(x): pass
# Trackbar for border dilatation value
cv.createTrackbar('Sensibility', 'input', INIT_BORDER_DIL, 6, nothing)
cv.setTrackbarMin('Sensibility', 'input', 1)
cv.createTrackbar('Face delay (frames)', 'input', INIT_FRAMES_DELAY, 120, nothing)

## HELP WINDOW
help = Help(
"""
-------------------
|       HELP WINDOW       |
-------------------

f: remove last face scanned
r: reset the scan

1: Show boxes ON/OFF
2: Show contours ON/OFF
3: Open/close canny window
4: Open/close model window
5: Open/close grid window

h: show/hide help
""")

###############################################################################
#                           MODEL IMAGE
###############################################################################
# We load an image with the desired outile
model_img = cv.imread(MODEL_FILE)
# Gray conversion and threshold
gmodel = 255 - cv.cvtColor(model_img, cv.COLOR_BGR2GRAY)
_, thres_model = cv.threshold(gmodel, 128, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# Extract the contours
model_contours, _ = cv.findContours(thres_model, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)   
model_contours = [ c.reshape(-1,2) for c in model_contours ]
# Contours ordered from largest to smallest area
model_contours = sorted(model_contours, key=cv.contourArea, reverse=True)
model_contours = [ c.reshape(-1,2) for c in model_contours ]
# The largest countour is the model
model = model_contours[0]
model = cv.approxPolyDP(model, EPSILON_POLY, True)  # The one used to compare
# Just for showing in a window
model_img = cv.drawContours(model_img, [model], -1, 255, 2)

###############################################################################
#                           MAIN PROGRAM
###############################################################################

###     GLOBAL VARIABLES    ###
# List of the faces of the cube (each face is a list of colors)
cube = []
# List of the color of the center stickers of the face (color model)
centers = []
# Moves to solve the cube
moves = []
# Remaining frames of delay
rem_delay = 0

# CUBE RESETTING
def reset_cube():
    global cube, centers, moves
    cube = []
    centers = []
    moves = []

###     MAIN LOOP    ###
for key, frame in autoStream():
    # FRAME DELAY
    if rem_delay > 0:   # If there is remaining delay moves on to the next iteration
        rem_delay -= 1
        # If delay is taken, we indicate with text in green (instead of white)
        drawTextC(frame, str(len(centers)) + '/6', (10, 30), 0.75, (60, 255, 60), 2)
        cv.imshow('input', frame)
        continue
    
    ### SHAPE DETECTION ###
    # Edge detection with Canny
    edges = cv.Canny(cv.GaussianBlur(frame, (0,0), 2).astype(np.uint8), 20, 60, L2gradient = True)
    # Edge dilatation
    border_dir = cv.getTrackbarPos('Sensibility', 'input')
    edges = cv.dilate(edges, np.ones((border_dir, border_dir), np.uint8))
    # Find the countours
    contours, _ = cv.findContours(edges, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)
    contours = [ c.reshape(-1,2) for c in contours ]    # Remove the redundant dimension
    # Node reducction with approxPolyDP
    contours = [ cv.approxPolyDP(c, EPSILON_POLY, True) for c in contours ] 
    # Select the contours with a descriptor very similar to the model
    sim_contours = [ c for c in contours if cv.matchShapes(c, model, 1, 0.0) < MAX_DIST_MODEL ]

    ### CONTOUR FILTER ###
    points_detected = []        # Center points of the countours
    radiuses_detected = []      # Radius of the contours
    rectangles = []             # Bounding box of countours
    # Find the bounding rectangles
    for c in sim_contours:
        x, y, w, h = cv.boundingRect(c)
        # Filter the boxes with the appropriate properties. This are:
        # 1. Height and width not too small
        # 2. Square shape
        # 3. Are not contained in other boxes already added
        if w >= MIN_SIZE_STICKER and h >= MIN_SIZE_STICKER and abs(h-w) < 20 and \
            not isPointInRectangles((x+w/2,y+h/2), rectangles):
            # We save the point and the radius 
            point = (int(x+w/2), int(y+h/2))
            points_detected.append(point)
            radiuses_detected.append(max(w/2, h/2))
            # Rectangles to check if next point is not contained here
            rectangles.append([(x,y), (x+w,y+h)])
            # Show boxes of the contours which passes this filter (in green)
            if SHOW_BOXES:
                cv.rectangle(frame, (x,y), (x+w,y+h), (72, 255, 80), 2)
    
    ### SEARCH FOR WELL-PLACED POINTS ###
    # At this point we have a list of candidates. We could have more than 9.
    # To ensure that the points are correct, there must be a point which, 
    # by constructing a sufficiently large square centered on it, 
    # must include eight other points. We will work with these 9 points.
    # If there is no center detected, the function returns [], []
    reasonable_points, bouding_square = getReasonablePoints(points_detected, radiuses_detected)

    ### COLOR EXTRACTION, CLASSIFICATION AND SOLVING ###
    # (If search has been successful)
    if reasonable_points:
        # Show the bounding box of the candidate points (in blue)
        if SHOW_BOXES:
            cv.rectangle(frame, bouding_square[0], bouding_square[1], (201, 145, 56), 4)

        ### COLOR EXTRACTION ###
        # Sort the coordinates of the points
        reasonable_points = sort_points(reasonable_points)
        # Extract the colors from the frame
        colors_detected = []
        for p in reasonable_points:
            colors_detected.append(np.int16(frame[p[1], p[0], :]))
        # Takes the center color
        actual_center = colors_detected[4]
        # Add the colors to the cube structure if face is not already scanned
        # (Checks if last center color is similiar to the actual one)
        if len(centers) == 0 or not simColor(centers[-1], actual_center):
            # If it is the first face after being scanned
            if len(centers) == 6: reset_cube()
            # Add the elements to the structures
            cube.append(colors_detected)
            centers.append(actual_center)
            rem_delay = cv.getTrackbarPos('Face delay (frames)', 'input')

            # IF ALL FACES HAVE BEEN SCANNED
            if len(cube) == 6:
                # List (of string) of the cube colors
                cube_scanned = []
                ### COLOR CLASSIFICATION ###
                # The idea is that the colors of the centers work as a color model
                # We compare each color of the cube with the nine center colors
                # The nearest center color is assigned to that color
                for face in cube:
                    # Difference between every color and all centers
                    difcolors = [ abs (face-c) for c in centers]
                    # Sum the values of every difference component
                    sumcolors = [ np.sum(dif, axis=1) for dif in difcolors ]
                    # Selects the nearest center
                    colorsdetected  = [ STRING_COLORS[d] for d in np.argmin(sumcolors, axis=0) ]
                    # Add the colors to the cube
                    cube_scanned.append(colorsdetected)

                ### ADAPT TO THE SOLVER FORMAT  ###
                # The order of the stickers on the scanned cube has to be the same as the order used by the solver. 
                # We adapt this order:
                cube_ord = []
                # UP face stays the same
                cube_ord.append(cube_scanned[0])
                # The (scan) order FRONT-RIGHT-BACK-LEFT has to be LEFT-FRONT-RIGHT-BACK
                cube_ord.append(cube_scanned[4])
                cube_ord.append(cube_scanned[1])
                cube_ord.append(cube_scanned[2])
                cube_ord.append(cube_scanned[3])
                # The BOTTOM face has to rotate -90 degrees
                bottomface = (np.array(cube_scanned[5])).reshape(3, 3)
                bottomface[:,[0, 2]] = bottomface[:,[2, 0]]
                bottomface = bottomface.transpose()
                cube_ord.append((bottomface.flatten()).tolist())
                # Convert de list to string
                cube_str = ""
                for nm in cube_ord:
                    rows = ["".join(r) for r in  nm]
                    cube_str += "".join(rows)

                ### CHECK FOR DUPLICATED PIECES ###
                # Check if there is 9 stickers of each color
                count_list = list(map(lambda x: cube_str.count(x), STRING_COLORS))
                print('Cube scanned: ' + cube_str)
                # If no duplicated colors
                if count_list == [9, 9, 9, 9, 9, 9]:
                    try:
                        ### CUBE SOLVING ###
                        moves = utils.solve(cube_str, 'Kociemba')
                        print('> Solution: ' + str(moves))
                    except:
                        print("> Not all edges or corners exist exactly once.")
                else:    # Something went wrong while scanning
                    print('> Duplicated piece: ' + str(STRING_COLORS) + ' scanned ' + str(count_list) + ' times.')

    # SHOW WINDOWS
    # Draw contours
    if SHOW_CONTOURS:
        # Contours found with canny
        if len(contours) > 0:
            cv.drawContours(frame, contours, -1, (67,212,255), 2)
        # Contours found similar to the model
        if len(sim_contours) > 0:
            cv.drawContours(frame, sim_contours, -1, (255,255,255), 2)
        # Draw the order of the nine points of the face
        if reasonable_points:
            for i in range(9):
                drawTextS(frame, str(i), reasonable_points[i], 0.4, (255, 255, 255), 1)
    # Frames scanned text
    drawTextC(frame, str(len(centers)) + '/6', (10, 30), 0.75, (255, 255, 255), 2)
    # Input window
    cv.imshow('input', frame)
    # Canny window
    if SHOW_CANNY: cv.imshow('canny', edges)
    # Color grid window
    if SHOW_GRID:
        grid_frame = drawGrid(cube, moves)
        cv.imshow('color grid', grid_frame)

    # KEYBOARD INPUT:
     # Help window
    help.show_if(key, ord('h'))
    # Pressed key
    if key == ord('f'):         # Remove last face scanned
        cube = cube[:-1]
        centers = centers[:-1]
        rem_delay = cv.getTrackbarPos('Face delay (frames)', 'input')
    elif key == ord('r'):       # Reset the cube
        reset_cube()
        rem_delay = cv.getTrackbarPos('Face delay (frames)', 'input')
    elif key == ord('1'):       # Show boxes ON/OFF
        SHOW_BOXES = not SHOW_BOXES
    elif key == ord('2'):       # Show contours ON/OFF
        SHOW_CONTOURS = not SHOW_CONTOURS
    elif key == ord('3'):       # Open/close canny window
        SHOW_CANNY = not SHOW_CANNY
        if not SHOW_CANNY: cv.destroyWindow('canny')
    elif key == ord('4'):       # Open/close model window
        SHOW_MODEL = not SHOW_MODEL
        if not SHOW_MODEL: cv.destroyWindow('model')
        else: cv.imshow('model', model_img) # This window just shows ones 
    elif key == ord('5'):       # Open/close grid window
        SHOW_GRID = not SHOW_GRID
        if not SHOW_GRID: cv.destroyWindow('color grid')
        else:
            cv.namedWindow('color grid') 
            cv.moveWindow('color grid', 700, 100)

cv.destroyAllWindows()