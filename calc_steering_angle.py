import os
import cv2
import numpy as np
import math
from PIL import Image

_SHOW_IMAGE = True
_CURVES = True
_REDUCED_ROI = False

'''
Hough Line Function Parameters
'''
rho = 5             # Resolution of accumulator buckets in pixels, Larger -> more lines
angle = 1           # Angle resolution of the accumulator in degrees, converted to radians. Larger -> fewer lines found
min_threshold = 5   # minimal of votes to determine line exists. Larger -> fewer lines
minLineLength = 1   # minimum length of segment to be considered a line. In pixels
maxLineGap = 20     # maximum distance between 2 segments to be considered a continuous line

'''
Waypoint Parameters
'''
x_threshold = 25
y_threshold = 15
waypoint_steps = 20

class HandCodedLaneFollower(object):

    def __init__(self, car=None, img_name=None):
        self.car = car
        self.curr_steering_angle = 90
        self.img_name = img_name

    def follow_lane(self, frame):

        frame = np.array(frame)
        show_image("1-orig", frame)

        lane_lines, frame = detect_lane(frame)
        final_frame = self.steer(frame, lane_lines)

        return final_frame

    def steer(self, frame, lane_lines):
        if len(lane_lines) == 0:
            return frame

        curr_heading_image, waypoints = compute_steering_angle(frame, lane_lines)

        show_image("7-heading", curr_heading_image)

        if (_CURVES):
            draw_waypoint_lines(frame, waypoints)
            curr_heading_image = draw_waypoint_lines(frame, waypoints)
            show_image("8-waypoints", curr_heading_image)

        return curr_heading_image

############################
# Frame processing steps
############################
def detect_lane(frame):
    edges = detect_edges(frame)
    show_image('2-edges', edges)

    cropped_edges = region_of_interest(edges)
    show_image('4-edges cropped', cropped_edges)

    line_segments = detect_line_segments(cropped_edges)
    line_segment_image = display_lines(frame, line_segments)
    show_image("5-hough line segments", line_segment_image)

    lane_lines = average_slope_intercept(frame, line_segments)
    lane_lines_image = display_lines(frame, lane_lines)
    show_image("6-lane lines", lane_lines_image)

    return lane_lines, lane_lines_image

def detect_edges(frame):
    edges = cv2.Canny(np.array(frame), 50, 200)
    return edges

def region_of_interest(img):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    mask = np.zeros_like(img)
    height, width = img.shape

    scale_w = 7 / 16
    scale_h = 11 / 18

    left_bottom = [0, height - 1]
    right_bottom = [width - 1, height - 1]
    left_up = [0, 0]
    right_up = [width - 1, 0]
    if (_REDUCED_ROI):
        left_up = [scale_w * width, scale_h * height]
        right_up = [(1 - scale_w) * width, scale_h * height]
    vertices = np.array([[left_bottom, left_up, right_up, right_bottom]], dtype=np.int32)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    show_image("3-ROI", mask)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def detect_line_segments(cropped_edges):
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle *  np.pi / 180, min_threshold, 
                                    np.array([]), minLineLength=minLineLength, maxLineGap=maxLineGap)
    return line_segments

def average_slope_intercept(frame, line_segments):
    """
    If _CURVES is true, this function returns the line segments as is

    else if _CURVES is false, this function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        print("No lane lines found")
        return lane_lines
    
    if (_CURVES):
        return line_segments

    height, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1/3
    left_region_boundary = width * (1 - boundary)   # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = width * boundary        # right lane line segment should be on left 2/3 of the screen

    threshold_angle = 25  # discard lines between -25 and +25 degrees
    threshold_slope = math.tan(threshold_angle / 180 * math.pi)

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]

            if x1 == x2:                        # line is vertical, skip
                continue
            if abs(slope) < threshold_slope:    # remove horizontal lines
                continue
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(left_fit_average[0], left_fit_average[1], 'l', height))
    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(right_fit_average[0], right_fit_average[1], 'r', height))

    return lane_lines


def compute_steering_angle(frame, lane_lines):
    """ 
    If _CURVES is true, this function returns an image with boundary lines and waypoints
    else if _CURVES is false, this function returns the steering angle based on a pair of lane lines
    """
    if len(lane_lines) == 0:
        return 90

    if (_CURVES):
        waypoints = []

        img_with_lines = np.zeros_like(frame)
        img_with_lines = display_lines(img_with_lines, lane_lines, line_width=1)

        lower_green = np.array([0,250,0])
        upper_green = np.array([1,255,1])
        mask = cv2.inRange(img_with_lines, lower_green, upper_green)
        coord = cv2.findNonZero(mask)           # already sorted by y
        num_lines, _, _ = np.shape(coord)

        if coord is not None:
            for i in range(0, num_lines-1):
                if i % waypoint_steps == 0:
                    x1 = coord[i][0][0]
                    y1 = coord[i][0][1]
                    x2 = coord[i+1][0][0]
                    y2 = coord[i+1][0][1]

                    if (abs(x1 - x2) < x_threshold):
                        continue                # don't draw circle between points too close together
                    if (y1 == y2):
                        if (len(waypoints) > 0 and (abs(y1 - waypoints[-1][1]) < y_threshold)):
                            continue
                        x_mid = int((x1 + x2) / 2)
                        cv2.circle(img_with_lines, (x_mid, y1), 5, (0, 0, 255), -1)
                        waypoints.append((x_mid, y1))

        return img_with_lines, waypoints

    if len(lane_lines) == 1:
        mid_start_x, mid_start_y, mid_end_x, mid_end_y = lane_lines[0][0]
    else:
        left_x1, left_y1, left_x2, left_y2 = lane_lines[0][0]
        right_x1, right_y1, right_x2, right_y2 = lane_lines[1][0]

        mid_start_x = int((left_x1 + right_x1) / 2)
        mid_start_y = int((left_y1 + right_y1) / 2)
        mid_end_x = int((left_x2 + right_x2) / 2)
        mid_end_y = int((left_y2 + right_y2) / 2)

    # Find slope of line connecting 2 points
    y_diff = mid_end_y - mid_start_y
    x_diff = mid_end_x - mid_start_x
    if x_diff == 0:
        steering_angle = 90
    else:
        slope = y_diff / x_diff
        steering_angle = math.degrees(math.atan(slope))

    heading_line_img = display_heading_line(frame, (mid_start_x, mid_start_y), (mid_end_x, mid_end_y))
    print("Pre-stabilized steering angle: ", steering_angle)

    return steering_angle, heading_line_img

############################
# Utility Functions
############################
def draw_waypoint_lines(frame, waypoints):
    """ 
    Draws lines between waypoints
    """
    img_with_lines = np.zeros_like(frame)
    for i in range(0, len(waypoints)-1):
        x1 = waypoints[i][0]
        y1 = waypoints[i][1]
        x2 = waypoints[i+1][0]
        y2 = waypoints[i+1][1]
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (255, 0, 0), 5)
    img_with_lines = cv2.addWeighted(frame, 0.8, img_with_lines, 1, 1)
    return img_with_lines

def display_lines(frame, lines, line_color=(0, 255, 0), line_width=10):
    '''
    Displays all lines onto the given frame
    '''
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

def display_heading_line(frame, p1, p2, line_color=(255, 0, 0), line_width=5):
    heading_image = np.zeros_like(frame)
    cv2.line(heading_image, p1, p2, line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    return heading_image

def show_image(title, frame, show=_SHOW_IMAGE):
    '''
    Renders an image to an opencv popup window.
    '''
    if show:
        img = Image.fromarray(frame)
        path = ['img_output', 'process', title + '.png']
        img.save(os.path.join(*path))
        cv2.imshow(title, frame)
        cv2.waitKey()

pre_l_slopes = []
pre_l_inters = []  
pre_r_slopes = []
pre_r_inters = []
def make_points(slope, inter, side, height):
    '''
    Given a slope and intercept, output start and end points of that line.
    '''
    number_buffer_frames = 3
    scale_y = 0.65
    top_y = int(float(height) * scale_y)  # fix the y coordinates of the top point, so that the line is more stable

    if side == 'l':
        if len(pre_l_slopes) == number_buffer_frames:  # reach the max
            pre_l_slopes.pop(0)  # remove the oldest frame
            pre_l_inters.pop(0)

        pre_l_slopes.append(slope)
        pre_l_inters.append(inter)
        slope = sum(pre_l_slopes) / len(pre_l_slopes)
        inter = sum(pre_l_inters) / len(pre_l_inters)

        p1_y = height-1
        p1_x = int((float(p1_y)-inter)/slope)
        p2_y = top_y
        p2_x = int((float(p2_y)-inter)/slope)
    else: 
        if len(pre_r_slopes) == number_buffer_frames:  # reach the max
            pre_r_slopes.pop(0)  # remove the oldest frame
            pre_r_inters.pop(0)

        pre_r_slopes.append(slope)
        pre_r_inters.append(inter)
        slope = sum(pre_r_slopes) / len(pre_r_slopes)
        inter = sum(pre_r_inters) / len(pre_r_inters)

        p1_y = height-1
        p1_x = int((float(p1_y)-inter)/slope)
        p2_y = top_y
        p2_x = int((float(p2_y)-inter)/slope)

    return [[p1_x, p1_y, p2_x, p2_y]]
