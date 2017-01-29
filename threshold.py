import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

'''
calculate the threshold of x or y sobel given certain thesh and kernel sizes
'''
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
  # grayscale image
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  # find abs sobel thresh
  if orient == 'x':
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  else:
    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
  
  #get abs value
  abs_sobel = np.absolute(sobel)
  # need to scale from 64 bit image to 8 bit... just needs to be uniform
  # so shouldnt mess up image?
  scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
  
  grad_binary = np.zeros_like(scaled)
  grad_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
  return grad_binary


'''
calculate magnitude of gradient given an image and threshold
'''
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
  # gray scale
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  
  # given the magnitude of threshold for the combined two, return
  abs_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
  abs_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

  mag = np.sqrt(abs_x ** 2 + abs_y ** 2)
  scaled = (255*mag/np.max(mag))

  binary_output = np.zeros_like(scaled)
  binary_output[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
  return binary_output

'''
calculate direction of gradient given image and thresh
'''
def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  
  # given the magnitude of threshold for the combined two, return
  abs_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
  abs_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

  sobel_dir = np.arctan2(abs_x, abs_y)
  scaled = (255*sobel_dir/np.max(sobel_dir))

  binary_output = np.zeros_like(scaled)
  binary_output[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
  return binary_output

'''
combine the thresholding functions
'''
def combo_thresh():
  x_thresholded = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(50, 200))
  mag_thresholded = mag_thresh(image, sobel_kernel=3, mag_thresh=(40, 160))
  dir_thresholded = dir_thresh(image, sobel_kernel=3, thresh=(0, np.pi/2))  
  
  binary_output = np.zeros_like(dir_thresholded)
  # using bitwise or + and, look up how working
  binary_output[(x_thresholded == 1) | ((mag_thresholded == 1) & (dir_thresholded == 1))] = 1
  return binary_output

if __name__ == '__main__':
  image = mpimg.imread('output_images/test2_undistorted.jpg')
  plt.imshow(image)
  plt.show()
  # thresholded = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(50, 200))
  # plt.imshow(thresholded, cmap='gray')
  # mag_thresholded = mag_thresh(image, sobel_kernel=3, mag_thresh=(40, 160))
  # plt.imshow(mag_thresholded, cmap='gray')
  # dir_thresholded = dir_thresh(image, sobel_kernel=3, thresh=(0, np.pi/2))  
  # plt.imshow(dir_thresholded, cmap='gray')

  combo = combo_thresh()
  plt.imshow(combo, cmap='gray')
  plt.show()