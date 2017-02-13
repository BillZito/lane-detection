import cv2
import pickle
import numpy as np
import scipy.misc as sci
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

#import helper methods from other files
from calibrate import undist
from threshold_helpers import *

'''
load undistortion matrix from camera 
'''
with open('test_dist_pickle.p', 'rb') as pick:
  dist_pickle = pickle.load(pick)

mtx = dist_pickle['mtx']
dist = dist_pickle['dist']

'''
warp the perspective based on 4 points
'''
def change_perspective(img):
  img_size = (img.shape[1], img.shape[0])

  # set fixed transforms based on image size
  src = np.float32(
    [[(img_size[0] / 2) - 31, img_size[1] / 2 + 82],
    [((img_size[0] / 6) + 37), img_size[1]],
    [(img_size[0] * 5 / 6) + 100, img_size[1]],
    [(img_size[0] / 2 + 30), img_size[1] / 2 + 82]])
  # print('src is 1:', (img_size[0] / 2) - 33, img_size[1] / 2 + 82, '2:', (img_size[0] / 6) + 37, img_size[1], \
    # '3:', (img_size[0] * 5 / 6) + 118, img_size[1], '4:', (img_size[0] / 2 + 33), img_size[1] / 2 + 82)

  dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

  # used to test that src points matched line
  # cv2.fillConvexPoly(img, src.astype('int32'), 1)
  # plt.imshow(img)
  # plt.title('lines')
  # plt.show()

  # create a transformation matrix based on the src and destination points
  M = cv2.getPerspectiveTransform(src, dst)

  #transform the image to birds eye view given the transform matrix
  warped = cv2.warpPerspective(img, M, (img_size[0], img_size[1]))
  # sci.imsave('./output_images/warped_5.jpg', warped)
  return warped

'''
get the pixels using the lecture code
'''
def lr_curvature(binary_warped):
  midpoint = int(binary_warped.shape[0]/2)
  # print('midpoint', midpoint)
  histogram = np.sum(binary_warped[midpoint :, :], axis=0)
  out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

  # plt.plot(histogram)
  # plt.title('histo')
  # plt.show()

  plt.imshow(out_img)
  plt.title('before windows')
  plt.show()

  leftx_base = np.argmax(histogram[:midpoint])
  rightx_base = np.argmax(histogram[midpoint:]) + midpoint
  # print('leftxbase', leftx_base, 'rightxbase', rightx_base)

  # 2. crate sliding windows
  nwindows = 9
  #set sliding window height
  window_height = np.int(binary_warped.shape[0]/nwindows)
  nonzero = binary_warped.nonzero()
  nonzeroy = np.array(nonzero[0])
  nonzerox = np.array(nonzero[1])
  # print('nonzeroy', nonzeroy.shape, 'nonzerox', nonzerox.shape)

  leftx_current = leftx_base
  rightx_current = rightx_base

  #width of sliding windows
  margin = 100
  #min number of pixels found to recenter
  minpix = 50

  left_lane_inds = []
  right_lane_inds = []

  #step through the windows
  for window in range(nwindows):
    #identify window boundaries in x and y (and right and left)
    #starts at 0-- height is 1/9th image
    win_y_low = binary_warped.shape[0] - (window + 1) * window_height
    #starts at 0 --- high is top of image 
    win_y_high = binary_warped.shape[0] - window * window_height 
    #left starts from -margin to + margin, same with right
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # print('all vals', win_xleft_high, win_xleft_low, win_xright_low, win_xright_high)
    
    #draw the windows on the image
    cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
    cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
    # cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
    # cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 

    #identify nozero pixels within the window
    #nonzero is numpy function that returns non zero vals
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

    #append these indices
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)

    #only recent if there are at least 50 dots (and therefore trustowrthy sample)
    if len(good_left_inds) > minpix:
      leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix: 
      rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

  #concat arrays of indices
  left_lane_inds = np.concatenate(left_lane_inds)
  right_lane_inds = np.concatenate(right_lane_inds)

  #extract left and right line pixel positions
  #getting all of the individal nonzero values from the lane_inds arrays?
  leftx = nonzerox[left_lane_inds]
  lefty = nonzeroy[left_lane_inds]
  rightx = nonzerox[right_lane_inds]
  righty = nonzeroy[right_lane_inds]

  #set up polyfit for left and right
  left_fit = np.polyfit(lefty, leftx, 2)
  right_fit = np.polyfit(righty, rightx, 2)

  #plot and visualize
  ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
  left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
  right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

  out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
  out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
  plt.title('after windows')
  plt.imshow(out_img)
  # plt.show()
  plt.plot(left_fitx, ploty, color='yellow')
  plt.plot(right_fitx, ploty, color='yellow')
  plt.xlim(0, 1280)
  plt.ylim(720, 0)
  # plt.imshow(out_img)
  plt.show()

'''
get the pixels for the left and right lanes and return them
'''
def get_lr(warped_image):
  #to start, divide by 2 and get peak
  #eventually divide vertically by 8 (720/8 is 90)
  half_height = int(warped_image.shape[0]/2)
  left_range = (200, 400)
  right_range = (650, 1150)

  full_hist = np.sum(warped_image[half_height:, :], axis=0)
  left_histogram = np.sum(warped_image[half_height:, left_range[0]: left_range[1]], axis=0)
  right_histogram = np.sum(warped_image[half_height:, right_range[0]: right_range[1]], axis=0)

  # find peak within the above ranges
  left_max = np.argmax(left_histogram)
  right_max = np.argmax(right_histogram)
  # print('right max hist', right_max)

  # add pixels near left and right peaks into left/right arrays
  left_start = left_range[0] + left_max - 70
  left_end = left_range[0] + left_max + 70

  right_start = right_range[0] + right_max - 70
  right_end = right_range[0] + right_max + 70
  # print('right end is', right_end)

  left_vals = warped_image[half_height:, left_start: left_end]
  right_vals = warped_image[half_height:, right_start: right_end]

  # fit line to those pixels using get_points()
  left_xy = get_points(left_vals, left_start)
  right_xy = get_points(right_vals, right_start)

  #used for plotting the full histogram to see its distribution
  # plt.plot(full_hist)
  # plt.title('full hist')
  # plt.show()

  return left_xy, right_xy

'''
Given an array with the pixel values, convert to an array of where 
those pixels were found on the x/y axis
'''
def get_points(vals, width_offset):
  #initialize array
  result = np.array([[0, 0]])

  #for each row
  for ridx, row in enumerate(vals):
    #for each column
    for cidx, val in enumerate(row):
      #if pixel found
      if val == 1:
        # add x/y coordinates to results array
        # 360 is 1/2 heigh (hardcoded right now but could be var)
        #720 is bottom of image, and 1280 the far right
        result = np.append(result, [[360 + ridx, width_offset + cidx]], axis=0)

  #delete first value
  result = np.delete(result, 0, axis=0)
  return result

'''
calculate the curve of the lines based on the pixels
'''
def calc_curve(left_vals, right_vals):
  
  # set left/righty to the first values, and left/rightx to the second
  left_yvals = np.array([elem[0] for idx, elem in enumerate(left_vals)])
  leftx = np.array([elem[1] for idx, elem in enumerate(left_vals)])
  # print('leftx', leftx.shape)

  # print('left yvals.shape', left_yvals.shape)
  # print('leftx.shape', leftx.shape)
  right_yvals = np.array([elem[0] for idx, elem in enumerate(right_vals)])
  rightx = np.array([elem[1] for idx, elem in enumerate(right_vals)])
  # print('right x', rightx.shape)

  #invert
  # leftx = leftx[::-1]
  # rightx = rightx[::-1]

  #fit to second order polynomial
  left_fit = np.polyfit(left_yvals, leftx, 2)
  left_fitx = left_fit[0]*left_yvals**2 + left_fit[1]*left_yvals + left_fit[2]
  
  right_fit = np.polyfit(right_yvals, rightx, 2)
  right_fitx = right_fit[0]*right_yvals**2 + right_fit[1]*right_yvals + right_fit[2]

  #plot left (red) and right (blue) lanes 
  plt.plot(leftx, left_yvals, 'o', color='red')
  plt.plot(rightx, right_yvals, 'o', color='blue')
  plt.xlim(0, 1280)
  plt.ylim(0, 720)

  #and their polynomials with green best fit
  plt.plot(left_fitx, left_yvals, color='green', linewidth=3)
  plt.plot(right_fitx, right_yvals, color='green', linewidth=3)
  plt.gca().invert_yaxis()
  plt.show()

  #convert from pixel space to meter space
  ym_per_pix = 30/720
  xm_per_pix = 3.7/700

  left_fit_cr = np.polyfit(left_yvals*ym_per_pix, leftx*xm_per_pix, 2)
  right_fit_cr = np.polyfit(right_yvals*ym_per_pix, rightx*xm_per_pix, 2)

  #calculate radisu of curvature
  left_eval = np.max(left_yvals)
  right_eval = np.max(right_yvals)
  left_curverad = ((1 + (2*left_fit_cr[0]*left_eval + left_fit_cr[1])**2)**1.5)/np.absolute(2*left_fit_cr[0])
  right_curverad = ((1 + (2*right_fit_cr[0]*right_eval + right_fit_cr[1])**2)**1.5)/np.absolute(2*right_fit_cr[0])
  
  # plt.show()

  # print('left curverad', left_curverad)
  # print('rightcurverad', right_curverad)


  # calculate left_min by finding minimum value in first index of array
  left_min = np.amin(leftx, axis=0)
  # print('left_min', left_min)
  right_max = np.amax(rightx, axis=0)
  # print('right max', right_max)
  actual_center = (right_max + left_min)/2
  dist_from_center =  actual_center - (1280/2)
  # print('pix dist from center', dist_from_center)
  meters_from_center = xm_per_pix * dist_from_center
  string_meters = str(round(meters_from_center, 2))
  # right_string = str(round(right_max, 2))
  # print('string meters', string_meters)
  # ', right_max: ' + right_string 
  # print('meters from center', meters_from_center)
  full_text = 'left: ' + str(round(left_curverad, 2)) + ', right: ' + \
    str(round(right_curverad, 2)) + ', dist from center: ' + string_meters 

  if abs(left_curverad - right_curverad) < 2000 \
    and right_max < 1100 and rightx.shape[0] > 100 or not lane.curve['full_text']:
    # print('setting vals now')
    lane.curve['left_fitx'] = left_fitx
    lane.curve['left_yvals'] = left_yvals
    lane.curve['right_fitx'] = right_fitx
    lane.curve['right_yvals'] = right_yvals
    lane.curve['full_text'] = full_text
  else:
    # print('getting previous vals')
    left_fitx= lane.curve['left_fitx'] 
    left_yvals = lane.curve['left_yvals'] 
    right_fitx = lane.curve['right_fitx'] 
    right_yvals = lane.curve['right_yvals']
    full_text = lane.curve['full_text']
  
  return left_fitx, left_yvals, right_fitx, right_yvals, full_text


'''
given left and right lines values, add to original image
'''
def draw_on_road(img, warped, left_fitx, left_yvals, right_fitx, right_yvals):
  #create img to draw the lines on
  warp_zero = np.zeros_like(warped).astype(np.uint8)
  color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

  #recast x and y into usable format for cv2.fillPoly
  pts_left = np.array([np.transpose(np.vstack([left_fitx, left_yvals]))])
  pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, right_yvals])))])
  # print('pts left', pts_left.shape, 'pts right', pts_right.shape)
  pts = np.hstack((pts_left, pts_right))

  #draw the lane onto the warped blank img
  cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

  img_size = (img.shape[1], img.shape[0])

  dst = np.float32(
    [[(img_size[0] / 2) - 33, img_size[1] / 2 + 82],
    [((img_size[0] / 6) + 37), img_size[1]],
    [(img_size[0] * 5 / 6) + 118, img_size[1]],
    [(img_size[0] / 2 + 33), img_size[1] / 2 + 82]])
  # print('src is', src)
    # [[(img_size[0] / 2) - 36, img_size[1] / 2 + 90],
    # [((img_size[0] / 6) + 50), img_size[1]],
    # [(img_size[0] * 5 / 6) + 80, img_size[1]],
    # [(img_size[0] / 2 + 36), img_size[1] / 2 + 90]]

  src = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
  # print('dst is', dst)

  # cv2.fillConvexPoly(image, src, 1)
  # plt.imshow(image)
  # plt.title('lines')
  # plt.show()
  Minv = cv2.getPerspectiveTransform(src, dst)


  #warp the blank back oto the original image using inverse perspective matrix
  newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

  #combine the result with the original 
  result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
  # print('result shape', result.shape)
  # plt.imshow(result)
  # plt.show()
  return result

'''
Run all steps of processing on an image. 
0. Undistort image
1. Create binary thresholds
2. Change to birds-eye-view
3. Calculate curvature of left/right lane
4. map back onto road
'''
def process_image(img):

  undist_img = undist(img, mtx, dist)
  plt.imshow(undist_img)
  plt.title('undist_img')
  plt.show()

  combo_image = combo_thresh(undist_img)
  plt.imshow(combo_image, cmap='gray')
  plt.title('combo_image')
  plt.show()

  warped_image = change_perspective(combo_image)
  plt.imshow(warped_image, cmap='gray')
  plt.title('warped_image')
  plt.show()

  # lr_curvature(warped_image)
  # print('warped shape[0]/2', int(warped_image.shape[0]/2))
  left_vals, right_vals = get_lr(warped_image)
  left_fitx, left_yvals, right_fitx, right_yvals, full_text = calc_curve(left_vals, right_vals)
  result = draw_on_road(img, warped_image, left_fitx, left_yvals, right_fitx, right_yvals)
  cv2.putText(result, full_text, (200, 100), cv2.FONT_HERSHEY_COMPLEX, 1, 255)
  # sci.imsave('./output_images/final_6.jpg', result)
  return result


'''
create a line class to keep track of important information about each line
'''
class Lane():
  def __init__(self):
    #if line was deteced in last iteration
    self.curve = {'full_text': ''}
    # self.detected = False
    # # x values of last n fits
    # self.recent_xfitted = []
    # #average x values of the fitted line over the last n iterations
    # self.bestx = None
    # #polynomial coefficients averaged over the last n
    # self.best_fit = None
    # #polynomial coefficients of the most recent fit
    # self.current_fit = [np.array([False])]
    # #raidus of curvature of the line in some units
    # self.radius_of_curvature = None
    # #distance in meters of vehicle center from the line
    # self.line_base_pos = None
    # #difference in fit coefficients between last and new fits
    # self.diffs = np.array([0, 0, 0], dtype='float')
    # #xvalues for detected line pixels
    # self.allx = None
    # #yvals 
    # self.ally = None


if __name__ == '__main__':
  lane = Lane()
  # #set video variables
  # proj_output = 'output.mp4'
  # clip1 = VideoFileClip('project_video.mp4')

  # #run process image on each video clip and save to file
  # output_clip = clip1.fl_image(process_image)
  # output_clip.write_videofile(proj_output, audio=False)


  # left = Line()
  # right = Line()
  # image = mpimg.imread('test_images/straight_road_1x.jpg')
  image = mpimg.imread('test_images/test1.jpg')
  # plt.imshow(image)
  # plt.title('norm image')
  # plt.show()

  colored_image = process_image(image)

  plt.imshow(colored_image)
  plt.title('colored_image')
  plt.show()

