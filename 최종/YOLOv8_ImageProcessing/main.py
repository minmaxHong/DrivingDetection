import cv2
import numpy as np
import yolov8_class
import lane_detection

def main():

    weight_path = "최종\yolov8n.pt"
    video_path = "아이로드 T10 블랙박스 FHD 30프레임 주행 영상.mp4"
    yolo8 = yolov8_class.object_detection(weight_path = weight_path, video_path = video_path)

    while True:
        # == Detect ==
        detect_frame = yolo8.detect_and_display()
        cv2.imshow('Detected Video', detect_frame)


        # ===================================================
        # detect_frame = yolo8.read_frame()

        # # == Frame 읽기 == 
        # if detect_frame is not None:
        #     cv2.imshow('Detect Frame', detect_frame)
        # else:
        #     break
        # ===================================================

        # == CallBack 함수 == 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # == Cap이 열려있다면 닫게 하기 == 
    cap = yolo8.cap_opened()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
