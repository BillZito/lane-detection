'''
calibrate.py takes 20 chessboard images and uses them to
calibrate the camera. It can then undistort a test image
'''
import os
import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

'''
calibrate_cam goes through the 20 chessboard images and calibrates the camera
'''
def calibrate_cam():
  nx = 9
  ny = 6
  
  # set object points to 0's to start
  objp = np.zeros((ny*nx, 3), np.float32)
  # set objp to the coordinates of the grid
  objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

  #store the the object points and image points
  objpoints = []
  imgpoints = []

  images = glob.glob('camera_cal/*.jpg')

  #step through images and find corners for each
  for idx, fname in enumerate(images):
    # read in image
    img = cv2.imread(fname)
    # convert to gray colors
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #get the corners 
    #ret is boolean true/false if found corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    #if found, add object and image points
    if ret == True:
      objpoints.append(objp)
      imgpoints.append(corners)

      #draw and display the corners
      cv2.drawChessboardCorners(img, (nx, ny), corners, ret)

      # show the image
      cv2.imshow('img', img)
      cv2.waitKey(500)

  cv2.destroyAllWindows()

  return objpoints, imgpoints

'''
undistort takes the 20 obj/imgpoints from calibration and uses them to distort the 6 test iamges
'''
def undist(objpoints, imgpoints):
  
  # change name of file to distort a different image
  # img = cv2.imread('test_images/test1.jpg')
  img_size = (img.shape[1], img.shape[0])

  #does it return, what is the distance, what are the r and t vals
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
  
  test_list = os.listdir('test_images')

  #undistort all images
  for index, img_name in enumerate(test_list):
    if img_name.startswith('test'):
      print('img name is', img_name)

      img = cv2.imread('test_images/' + img_name)
      dst = cv2.undistort(img, mtx, dist, None, mtx)
      cv2.imwrite('output_images/' + img_name + '_undistorted.jpg', dst)

  # save the matrix and distance so dont have to calculate each time
  dist_pickle = {}
  dist_pickle["mtx"] = mtx
  dist_pickle["dist"] = dist
  pickle.dump( dist_pickle, open("test_dist_pickle.p", "wb"))
  
  #show images
  # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
  # ax1.imshow(img)
  # ax1.set_title('Original Image', fontsize=30)
  # ax2.imshow(dst)
  # ax2.set_title('Undistoredted Image', fontsize=30)


if __name__ == '__main__':
  # objpoints, imgpoints = calibrate_cam()
  # print('calibration complete')
  # undist(objpoints, imgpoints)
  # print('undistort complete')
