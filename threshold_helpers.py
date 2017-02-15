import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from calibrate import undist

'''
calculate the threshold of x or y sobel given certain thesh and kernel sizes
'''
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
  # grayscale image
  red = img[:, :, 0]

  # find abs sobel thresh
  if orient == 'x':
    sobel = cv2.Sobel(red, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  else:
    sobel = cv2.Sobel(red, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
  
  #get abs value
  abs_sobel = np.absolute(sobel)
  scaled = np.uint8(255*abs_sobel/np.max(abs_sobel))
  
  grad_binary = np.zeros_like(scaled)
  grad_binary[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 1
  return grad_binary


'''
calculate magnitude of gradient given an image and threshold
'''
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
  # gray scale
  red = img[:, :, 0]
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  
  # given the magnitude of threshold for the combined two, return
  abs_x = np.absolute(cv2.Sobel(red, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
  abs_y = np.absolute(cv2.Sobel(red, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

  mag = np.sqrt(abs_x ** 2 + abs_y ** 2)
  scaled = (255*mag/np.max(mag))

  binary_output = np.zeros_like(scaled)
  binary_output[(scaled >= mag_thresh[0]) & (scaled <= mag_thresh[1])] = 1
  return binary_output

'''
calculate direction of gradient given image and thresh
'''
def dir_thresh(img, sobel_kernel=3, thresh=(0, np.pi/2)):
  # red = img[:, :, 0]

  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  
  # given the magnitude of threshold for the combined two, return
  abs_x = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
  abs_y = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))

  sobel_dir = np.arctan2(abs_y, abs_x)

  binary_output = np.zeros_like(sobel_dir)
  binary_output[(sobel_dir >= thresh[0]) & (sobel_dir <= thresh[1])] = 1
  return binary_output

'''
calculate the threshold of the hls values
'''
def hls_thresh(img, thresh=(0, 255)):
  hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

  s_channel = hls[:, :, 2]

  binary_output = np.zeros_like(s_channel)
  binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

  return binary_output

'''
get v channel from hsv
'''
def hsv_thresh(img, thresh=(0, 255)):
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

  v_channel = hsv[:, :, 2]

  binary_output = np.zeros_like(v_channel)
  binary_output[(v_channel > thresh[0]) & (v_channel <= thresh[1])] = 1

  return binary_output

'''
combine the thresholding functions
'''
def combo_thresh(img):


  x_thresholded = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(12, 120))
  # plt.imshow(x_thresholded, cmap='gray')
  # plt.title('xthresh')
  # plt.show()

  y_thresholded = abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(25, 100))
  # plt.imshow(y_thresholded, cmap='gray')
  # plt.title('ythresh')
  # plt.show()

  # was 90
  hls_thresholded = hls_thresh(img, thresh=(100, 255))
  # plt.imshow(hls_thresholded, cmap='gray')
  # plt.title('hls')
  # plt.show()
  hsv_thresholded = hsv_thresh(img, thresh=(50, 255))

  dir_thresholded = dir_thresh(img, sobel_kernel=15, thresh=(.7, 1.2))  
  # plt.imshow(dir_thresholded, cmap='gray')  
  # plt.title('directional')
  # plt.show()
  

  mag_thresholded = mag_thresh(img, sobel_kernel=3, mag_thresh=(30, 100))
  # plt.imshow(mag_thresholded, cmap='gray')
  # plt.title('magnitude')
  # plt.show()
  

  # first_combo = np.zeros_like(dir_thresholded)
  # using bitwise or + and, look up how working
  # first_combo[(((dir_thresholded == 1) | (mag_thresholded == 1)) & (hls_thresholded == 1))] = 1
  # plt.imshow(first_combo, cmap='gray')
  # plt.title('(dir or mag) and hls')
  # plt.show()


  # second_combo = np.zeros_like(x_thresholded)
  # second_combo[((hls_thresholded == 1) & (x_thresholded == 1))] = 1
  # plt.imshow(second_combo, cmap='gray')
  # plt.title('x and hls')
  # plt.show()

  # third_combo = np.zeros_like(dir_thresholded)
  # # using bitwise or + and, look up how working
  # third_combo[((y_thresholded == 1) & (hls_thresholded == 1) & (x_thresholded == 1))] = 1
  # plt.imshow(third_combo, cmap='gray')
  # plt.title('x, y, and hls')
  # plt.show()


  binary_output = np.zeros_like(dir_thresholded)
  binary_output[((hsv_thresholded == 1) & (hls_thresholded == 1)) | ((x_thresholded == 1) & (y_thresholded == 1))] = 1

  # binary_output[(((dir_thresholded == 1) | (mag_thresholded == 1) ) & (hls_thresholded == 1)) | ((x_thresholded == 1) & (y_thresholded == 1))] = 1
  # 
  return binary_output

'''
given a directory, return an array of all images in it
'''
def get_file_images(directory):
  file_list = os.listdir(directory)
  first_image = mpimg.imread(directory + '/' + file_list[1])
  all_images = np.array([first_image])
  # print('all_images shape', all_images.shape)

  for img_num in range(2, len(file_list)):
    img_name = file_list[img_num]
    if not img_name.startswith('.'):
      # print('img name is', img_name)
      image = mpimg.imread(directory + '/' + img_name)
      # undist_img = undist(image, mtx, dist)
      all_images = np.append(all_images, np.array([image]), axis=0)

  # print('final shape', all_images.shape)
  return all_images

'''
for each image in array, print
'''
def show_images(images):
  fig = plt.figure()
  for num in range(1, len(images)):
    image = images[num]
    fig.add_subplot(3, 3, num)
    plt.title(num)
    plt.imshow(image, cmap='gray')

  plt.show()

'''
run thresholding function on each image so that can see how it works on all
'''
def threshold_all(directory, func):
  file_list = os.listdir(directory)
  first_image = mpimg.imread(directory + '/' + file_list[1])
  thresholded_image = func(first_image)
  result = np.array([thresholded_image])

  for img_num in range(0, len(file_list)):
    img_name = file_list[img_num]
    if not img_name.startswith('.'):
      image = mpimg.imread(directory + '/' + img_name)
      thresholded_image = func(image)
      result = np.append(result, np.array([thresholded_image]), axis=0)

  return result


if __name__ == '__main__':
  '''
  load undistortion matrix from camera 
  '''
  with open('test_dist_pickle.p', 'rb') as pick:
    dist_pickle = pickle.load(pick)
  mtx = dist_pickle['mtx']
  dist = dist_pickle['dist']


  # images = get_file_images('test_images')
  # show_images(images)

  # thresholded_images = threshold_all('test_images', combo_thresh)
  # show_images(thresholded_images)


  # image = mpimg.imread('test_images/test2.jpg')
  # undist_img = undist(image, mtx, dist)

  # binary_output = combo_thresh(undist_img)
  # plt.imshow(binary_output, cmap='gray')
  # plt.title('binary thresh')
  # plt.show()
