import cv2
import sys
import lane_detection
from ultralytics import YOLO


class object_detection:
    def __init__(self, weight_path = None, video_path = None):
        # == Model Upload == 
        self.weight_path = weight_path
        self.model = YOLO(self.weight_path)
        
        # == Cap & Video == 
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)
    
    # == frmae별로 object detect & frame 읽기 == 
    def detect_and_display(self):

        ret, frame = self.cap.read()

        if not ret:
            print('Warning')
            sys.exit()
        
        # == Lane == 
        hsl_img = lane_detection.HSL_color_selection(frame)
        gray_img = lane_detection.gray_scale(hsl_img)
        blur_img = lane_detection.gaussian_filter(gray_img)
        edge_img = lane_detection.canny_filter(blur_img)
        roi_img = lane_detection.region_selection(edge_img)
        hough_lines = lane_detection.hough_transform(roi_img)
        line_images = lane_detection.draw_hough_lines(frame, hough_lines)
        lane_images = lane_detection.draw_lane_lines(frame, lane_detection.lane_lines(line_images, hough_lines))

        # == Object == 
        detect_video = self.model(lane_images, conf = 0.6)
        detect_video_frame = detect_video[0].plot()

        return detect_video_frame

    # == While문에서 frame별로 읽기(detect하지 않은 video임) -> 확인용 == 
    def read_frame(self):
        ret, frame = self.cap.read()

        if ret:
            return frame
        
    # == cap == 
    def cap_opened(self):
        return self.cap