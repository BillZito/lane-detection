import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
combine the thresholding functions
'''
def combo_thresh(img):
  x_thresholded = abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(10, 120))
  plt.imshow(x_thresholded, cmap='gray')
  plt.title('xthresh')
  plt.show()

  y_thresholded = abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(15, 100))
  plt.imshow(y_thresholded, cmap='gray')
  plt.title('ythresh')
  plt.show()

  binary_output = np.zeros_like(x_thresholded)
  # using bitwise or + and, look up how working
  binary_output[((x_thresholded == 1) & (y_thresholded == 1))] = 1
  plt.imshow(binary_output, cmap='gray')
  plt.title('x and y')
  plt.show()

  hls_thresholded = hls_thresh(img, thresh=(90, 255))
  plt.imshow(hls_thresholded, cmap='gray')
  plt.title('hls')
  plt.show()
  
  mag_thresholded = mag_thresh(img, sobel_kernel=3, mag_thresh=(20, 160))
  plt.imshow(mag_thresholded, cmap='gray')
  plt.title('magnitude')
  plt.show()

  dir_thresholded = dir_thresh(img, sobel_kernel=15, thresh=(.7, 1.2))  
  plt.imshow(dir_thresholded, cmap='gray')  
  plt.title('directional')
  plt.show()

  binary_output = np.zeros_like(dir_thresholded)
  binary_output[((dir_thresholded == 1) & (mag_thresholded == 1) & (hls_thresholded == 1))] = 1
  plt.imshow(binary_output, cmap='gray')
  plt.title('dir and mag and hls')
  plt.show()


  binary_output = np.zeros_like(dir_thresholded)
  binary_output[((x_thresholded == 1) & (y_thresholded == 1)) | ((dir_thresholded == 1) & (mag_thresholded == 1) & (hls_thresholded == 1))] = 1
  # 
  return binary_output

def get_file_images(directory):
  file_list = os.listdir(directory)
  first_image = mpimg.imread(directory + '/' + file_list[1])
  result = np.array([first_image])
  print('result shape', result.shape)

  for img_num in range(2, len(file_list)):
    img_name = file_list[img_num]
    if not img_name.startswith('.'):
      print('img name is', img_name)
      image = mpimg.imread(directory + '/' + img_name)
      result = np.append(result, np.array([image]), axis=0)

  print('final shape', result.shape)
  return result



def show_images(images):

  fig = plt.figure()
  
  for num in range(1, len(images)):
    image = images[num]
    fig.add_subplot(3, 3, num)
    plt.title(num)
    plt.imshow(image)

  plt.show()

if __name__ == '__main__':
  images = get_file_images('test_images')
  show_images(images)
  # image = mpimg.imread('test_images/test1.jpg')
  # binary_output = combo_thresh(image)
  # plt.imshow(binary_output, cmap='gray')
  # plt.title('binary thresh')
  # plt.show()
#   print('hello world')

'''
run all 6 images through one threshold function and then print them
  store images in an array and print with show images function

run all 6 images through the combo threshold function and print them
'''
