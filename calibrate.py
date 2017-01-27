'''
calibrate.py takes 20 chessboard images and uses them to
calibrate the camera. It can then undistort a test image
'''
import numpy as np
import cv2
import glob

# set object points to 0's to start
objp = np.zeros((6*8, 3), np.float32)
# set objp to the coordinates of the grid
objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
# print('objp .shape', objp[9])

#store the the object points and image points
objpoints = []
imgpoints = []

images = glob.glob('camera_cal/*')

#step through images and find corners for each
for idx, fname in enumerate(images):
  # read in image
  img = cv2.imread(fname)
  # convert to gray colors
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

  #get the corners 
  #ret is boolean true false if found corners
  ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

  #if found, add object and image points
  if ret == True:
    objpoints.append(objp)
    imgpoints.append(corners)

    #draw and display the corners
    #8 by 6 corners. draw  
    cv2.drawChessboardCorners(img, (8, 6), corners, ret)



# if __name__ == '__main__':
  # print('hello world')