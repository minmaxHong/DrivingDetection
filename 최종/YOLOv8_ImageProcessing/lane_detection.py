from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import cv2
import sys

# RGB -> HSL
def convert_hsl(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

def HSL_color_selection(image):
    converted_image = convert_hsl(image)

    # == White == 
    lower_threshold = np.uint8([0, 200, 0])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    # == Yellow color mask ==
    lower_threshold = np.uint8([10, 0, 100])
    upper_threshold = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    # == Combine white and yellow masks == 
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask = mask)

    return masked_image

# == HSL -> GRAY == 
def gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        
# == Gaussian ==
def gaussian_filter(image, kernel_size = 13):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0) 

# == Edge(canny) == 
def canny_filter(image, low_thres = 50, high_thres = 150):
    return cv2.Canny(image, low_thres, high_thres)

# == Edge(Sobel) == 
def sobel_filter(image, delta = 128):
    return cv2.Sobel(image, cv2.CV_8U, 1, 0, 3)

# == Edge(Laplacian)
def laplacian_filter(image, color = cv2.CV_8U, ksize = 3):
    return cv2.Laplacian(image, color, ksize)


# == ROI == 
def region_selection(image):
    mask = np.zeros_like(image)
    
    # RGB or HSV
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
        
    # Gray
    else:
        ignore_mask_color = 255

    rows, cols = image.shape[:2]

    # Region
    bottom_left  = [cols * 0.1, rows * 0.95]
    top_left     = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right    = [cols * 0.6, rows * 0.6]
    
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color) 
    masked_image = cv2.bitwise_and(image, mask) # and 연산 -> 둘다 흰색인 곳만 흰색

    return masked_image

# == Hough Transform == 
def hough_transform(image):
    
    rho = 1
    theta = np.pi / 180
    threshold = 20
    minLineLength = 20
    maxLineGap = 300
    
    return cv2.HoughLinesP(image, rho = rho, theta = theta, threshold = threshold,
                           minLineLength = minLineLength, maxLineGap = maxLineGap)

def draw_hough_lines(image, lines, color = (0, 0, 255), thickness = 5):
    image = np.copy(image)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


# == 여러 개의 선을 검출 -> 각 차선에 1개의 선만 그리기(Extrapolate) ==
def average_slope_intercept(lines):
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # Check if the denominator is not zero
            if x1 == x2:
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope * x1)
            length = np.sqrt(((y2 - y1) ** 2 + (x2 - x1) ** 2))

            if slope < 0:
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                right_lines.append((slope, intercept))
                right_weights.append(length)

    # Check for empty lists to avoid division by zero
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane


def pixel_points(y1, y2, line, min_length=10, max_length=float('inf')):
    if line is None:
        return None

    slope, intercept = line

    # Check if the slope is zero
    if slope == 0:
        x1 = int(intercept)
        x2 = int(intercept)
    else:
        # Check if the slope is finite
        if np.isfinite(slope):
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
        else:
            x1 = int(intercept)
            x2 = int(intercept)

    y1 = int(y1)
    y2 = int(y2)

    # Check line length
    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    if min_length <= line_length <= max_length:
        return ((x1, y1), (x2, y2))
    else:
        return None


def lane_lines(image, lines):
    
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.6
    left_line  = pixel_points(y1, y2, left_lane)
    right_line = pixel_points(y1, y2, right_lane)
    return left_line, right_line
    
def draw_lane_lines(image, lines, color=[0, 0, 255], thickness=12):
    line_image = np.zeros_like(image)
    
    for line in lines:
        if line is not None:
            cv2.line(line_image, line[0], line[1], color, thickness)
            
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)
