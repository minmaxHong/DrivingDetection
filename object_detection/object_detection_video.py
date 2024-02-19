import cv2
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

# == 가중치, 모델 == 
cfg_yolo = 'yolov3.cfg'
weights_yolo = 'yolov3.weights'

# == video ==
video_path = 'challenge.mp4'
cap = cv2.VideoCapture(video_path)

# == Label == 
labels_to_names_seq = {0:'person',1:'bicycle',2:'car',3:'motorbike',4:'aeroplane',5:'bus',6:'train',7:'truck',8:'boat',9:'traffic light',10:'fire hydrant',
                        11:'stop sign',12:'parking meter',13:'bench',14:'bird',15:'cat',16:'dog',17:'horse',18:'sheep',19:'cow',20:'elephant',
                        21:'bear',22:'zebra',23:'giraffe',24:'backpack',25:'umbrella',26:'handbag',27:'tie',28:'suitcase',29:'frisbee',30:'skis',
                        31:'snowboard',32:'sports ball',33:'kite',34:'baseball bat',35:'baseball glove',36:'skateboard',37:'surfboard',38:'tennis racket',39:'bottle',40:'wine glass',
                        41:'cup',42:'fork',43:'knife',44:'spoon',45:'bowl',46:'banana',47:'apple',48:'sandwich',49:'orange',50:'broccoli',
                        51:'carrot',52:'hot dog',53:'pizza',54:'donut',55:'cake',56:'chair',57:'sofa',58:'pottedplant',59:'bed',60:'diningtable',
                        61:'toilet',62:'tvmonitor',63:'laptop',64:'mouse',65:'remote',66:'keyboard',67:'cell phone',68:'microwave',69:'oven',70:'toaster',
                        71:'sink',72:'refrigerator',73:'book',74:'clock',75:'vase',76:'scissors',77:'teddy bear',78:'hair drier',79:'toothbrush' 
                        }

# == Yolo Model == 
yolo_net = cv2.dnn.readNet(weights_yolo, cfg_yolo)
# yolo_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# yolo_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

layer_names = yolo_net.getLayerNames()
outlayer_names = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

def detect_objects(img, yolo_net, labels_to_names_seq, conf_threshold=0.5, nms_threshold=0.4):
    rows, cols, _ = img.shape

    yolo_net.setInput(cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False))
    cv_outs = yolo_net.forward(outlayer_names)

    green_color = (0, 255, 0)
    red_color = (0, 0, 255)

    class_ids, confidences, boxes = [], [], []

    for output in cv_outs:
        scores = output[:, 5:]
        class_ids_batch = np.argmax(scores, axis=1)
        confidences_batch = scores[np.arange(len(scores)), class_ids_batch]
        mask = confidences_batch > conf_threshold

        class_ids.append(class_ids_batch[mask])
        confidences.append(confidences_batch[mask])

        detections = output[mask]
        detections[:, 0:4] *= np.array([cols, rows, cols, rows])

        centers = detections[:, 0:2]
        sizes = detections[:, 2:4]
        boxes.append(np.concatenate([centers - sizes / 2, sizes], axis=1))

    class_ids = np.concatenate(class_ids)
    confidences = np.concatenate(confidences)
    boxes = np.concatenate(boxes)

    idxs = cv2.dnn.NMSBoxes(boxes.tolist(), confidences, conf_threshold, nms_threshold).flatten()

    if len(idxs) > 0:
        for i in idxs:
            left, top, width, height = boxes[i]
            caption = "{}: {:.4f}".format(labels_to_names_seq[int(class_ids[i])], confidences[i])

            cv2.rectangle(img, (int(left), int(top)), (int(left + width), int(top + height)), color=green_color, thickness=2)
            cv2.putText(img, caption, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1)

while True:
    ret, img = cap.read()
    
    if not ret:
        break    
    
    detect_objects(img, yolo_net, labels_to_names_seq)

    cv2.imshow('img',img)
    # cv2.imshow('img', frame)
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()