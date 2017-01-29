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
# def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):

'''
calculate direction of gradient given image and thresh
'''
# def dir_thresh(image, sobel_kernel=3, thresh=(0, np.pi/2)):

'''
combine the thresholding functions
'''
# def combo_thresh():

if __name__ == '__main__':
  image = mpimg.imread('output_images/test2_undistorted.jpg')
  # plt.imshow(image)
  # plt.show()
  thresholded = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(50, 200))
  plt.imshow(thresholded, cmap='gray')
  # print('converted')
  plt.show()