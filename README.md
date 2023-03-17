# Rubik-OpenCV

**rubikDetector0.py**: Initial version of the program. The cube can only be scanned in a fixed position. Uses the cube centers to extract the color model from the cube.

**rubikDetector.py**: Final version of the program. The cube can be scanned at any position on the screen. Uses shape detection to locate the pieces. This detection is implemented by comparing each shape obtained with a model image (an image of a sticker of the cube).
