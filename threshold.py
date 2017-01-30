# import cv
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
  # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  # find abs sobel thresh
  if orient == 'x':
    sobel = cv2.Sobel(red, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  else:
    sobel = cv2.Sobel(red, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
  
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
def combo_thresh():
  x_thresholded = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(10, 120))
  # plt.imshow(x_thresholded, cmap='gray')
  # plt.title('xthresh')
  # plt.show()

  y_thresholded = abs_sobel_thresh(image, orient='y', sobel_kernel=3, thresh=(15, 100))
  # plt.imshow(y_thresholded, cmap='gray')
  # plt.title('ythresh')
  # plt.show()

  binary_output = np.zeros_like(x_thresholded)
  # using bitwise or + and, look up how working
  binary_output[((x_thresholded == 1) & (y_thresholded == 1))] = 1
  # plt.imshow(binary_output, cmap='gray')
  # plt.title('x and y')
  # plt.show()

  hls_thresholded = hls_thresh(image, thresh=(90, 255))
  # plt.imshow(hls_thresholded, cmap='gray')
  # plt.title('hls')
  # plt.show()
  

  # binary_output = np.zeros_like(x_thresholded)
  # # using bitwise or + and, look up how working
  # binary_output[((x_thresholded == 1) & (y_thresholded == 1)) & (hls_thresholded == 1)] = 1
  # plt.imshow(binary_output, cmap='gray')
  # plt.title('x and y, and hls')
  # plt.show()

  mag_thresholded = mag_thresh(image, sobel_kernel=3, mag_thresh=(20, 160))
  # plt.imshow(mag_thresholded, cmap='gray')
  # plt.title('magnitude')
  # plt.show()

  dir_thresholded = dir_thresh(image, sobel_kernel=15, thresh=(.7, 1.2))  
  # plt.imshow(dir_thresholded, cmap='gray')  
  # plt.title('directional')
  # plt.show()

  binary_output = np.zeros_like(dir_thresholded)
  binary_output[((dir_thresholded == 1) & (mag_thresholded == 1))] = 1
  # plt.imshow(binary_output, cmap='gray')
  # plt.title('dir and mag')
  # plt.show()


  binary_output = np.zeros_like(dir_thresholded)
  binary_output[((x_thresholded == 1) & (y_thresholded == 1)) | ((dir_thresholded == 1) & (mag_thresholded == 1) & (hls_thresholded == 1))] = 1
  # 
  return binary_output


'''
warp the perspective based on 4 points
'''
def changePerspective(img):
  img_size = (image.shape[1], image.shape[0])
  # print('image shape is', img_size)
  # [0] is 720, [1] is 128-
  src = np.float32(
    [[(img_size[0] / 2) - 40, img_size[1] / 2 + 90],
    [((img_size[0] / 6) + 40), img_size[1]],
    [(img_size[0] * 5 / 6) + 115, img_size[1]],
    [(img_size[0] / 2 + 42), img_size[1] / 2 + 90]])
  # print('src is', src)

  dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
  # print('dst is', dst)

  # cv2.fillConvexPoly(image, src, 1)
  # plt.imshow(image)
  # plt.title('lines')
  # plt.show()
  M = cv2.getPerspectiveTransform(src, dst)
  shape = img.shape
  warped = cv2.warpPerspective(img, M, (shape[1], shape[0]))
  return warped

'''
get the left and right images
'''
def get_lr(warped_image):
  # divide vertically by 8 (720/8 is 90)
  height = int(warped_image.shape[0]/2)
  width = (200, 400)
  histogram = np.sum(warped_image[height:, width[0]:width[1]], axis=0)
  # find peak between 200/ and 500/ 
  # use a 50 pixel wide map
  max_hist = np.argmax(histogram)
  print('max hist', max_hist)

  # add to array
  # current sending values 0, 0 -- 150, 360, 0, 720
  left_width = width[0] + max_hist - 70
  right_width = width[0] + max_hist + 70
  print('warped image', warped_image.shape)
  left_vals = warped_image[height:, left_width: right_width]
  print('left vals inside getlr', left_vals[:, 30:40])

  '''
  i have pixels 1 values on two separate lines-- I want to send 
  only the one-values as points (with their x and y val) to 
  my calculating function
  --try numpy.where value == 1 (save as coordinates)
  --google how to do that
  --make custom for loop-- if value is one, save point to separate array
  '''
  result = np.array([[0, 0]])
  for ridx, row in enumerate(left_vals):
    # print('row')
    for cidx, val in enumerate(row):
      if val == 1:
        print('found a 1', 360 + ridx, left_width + cidx)
        result = np.append(result, [[360 + ridx, left_width + cidx]], axis=0)
  print('result shape', result.shape)

  # -fit line to those pixels
  
  plt.plot(result)
  plt.title('only 1s')
  plt.show()
  # print('hello world')
  return result

'''
calculate the curve of the lines based on the pixels
'''
def calc_curve(left_vals):
  #replace the y and x data with my data and this code should work...
  #make fake y-range data
  yvals = np.linspace(0, 100, num=100)*7.2
  # print('yvals len', yvals.shape[0])
  # print('leftx len', leftx.shape[0])
  #y-range as image... what does that mean?
  # leftx = np.array([200 + (elem**2)*4e-4 + np.random.randint(-50, high=51) for idx, elem in enumerate(yvals)])
  #reverse to match top-to-bottom in y (because np images reversed?)

  # print('leftx', leftx)
  #gen right images
  rightx = np.array([900 + (elem**2)*4e-4 + np.random.randint(-50, high=51) for idx, elem in enumerate(yvals)])

  # rightx = leftx
  rightx = rightx[::-1]
  # leftx = leftx[::-1]
  # print('rightx', rightx)
  left_yvals = np.array([elem[0] for idx, elem in enumerate(left_vals)])
  leftx = np.array([elem[1] for idx, elem in enumerate(left_vals)])
  print('left yvals.shape', left_yvals.shape)
  print('leftx.shape', leftx.shape)
  #fit to second order polynomial
  left_fit = np.polyfit(left_yvals, leftx, 2)
  left_fitx = left_fit[0]*left_yvals**2 + left_fit[1]*left_yvals + left_fit[2]
  right_fit = np.polyfit(yvals, rightx, 2)
  right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]

  plt.plot(leftx, left_yvals, 'o', color='red')
  plt.plot(rightx, yvals, 'o', color='blue')
  plt.xlim(0, 1280)
  plt.ylim(0, 720)
  plt.plot(left_fitx, left_yvals, color='green', linewidth=3)
  plt.plot(right_fitx, yvals, color='green', linewidth=3)
  plt.gca().invert_yaxis()
  plt.show()

if __name__ == '__main__':
  # image = mpimg.imread('straight_road_1x.jpg')
  image = mpimg.imread('output_images/test5_undistorted.jpg')
  # plt.imshow(image)
  # plt.title('starter')
  # plt.show()

  combo_image = combo_thresh()
  # plt.imshow(combo_image, cmap='gray')
  # plt.title('combo_image')
  # plt.show()

  warped_image = changePerspective(combo_image)
  plt.imshow(warped_image, cmap='gray')
  plt.title('warped_image')
  plt.show()
  # print('warped shape', warped_image.shape)
  # print('warped shape[0]/2', int(warped_image.shape[0]/2))
  left_vals = get_lr(warped_image)
  calc_curve(left_vals)

  # x_thresholded = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(10, 120))
  # plt.imshow(x_thresholded, cmap='gray')
  # plt.title('xthresh')
  # plt.show()




  # y_thresholded = abs_sobel_thresh(image, orient='y', sobel_kernel=3, thresh=(15, 100))
  # plt.imshow(y_thresholded, cmap='gray')
  # plt.title('ythresh')
  # plt.show()

  # hls_thresholded = hls_thresh(image, thresh=(90, 255))
  # plt.imshow(hls_thresholded, cmap='gray')
  # plt.title('hls')
  # plt.show()

  # mag_thresholded = mag_thresh(image, sobel_kernel=3, mag_thresh=(20, 100))
  # plt.imshow(mag_thresholded, cmap='gray')
  # plt.title('magnitude')
  # plt.show()

  # dir_thresholded = dir_thresh(image, sobel_kernel=15, thresh=(.7, 1.2))  
  # plt.imshow(dir_thresholded, cmap='gray')  
  # plt.title('directional')
  # plt.show()