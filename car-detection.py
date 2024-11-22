from ultralytics import YOLO
import cv2 as cv
import cvzone
import math
from sort import *

model = YOLO("YOLO weights/yolov8n.pt")

classNames = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat",\
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",\
    "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",\
    "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",\
    "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",\
    "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","sofa",\
    "pottedplant","bed","diningtable","toilet","tvmonitor","laptop","mouse","remote","keyboard",\
    "cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors",\
    "teddy bear","hair drier","toothbrush",
]


# Loading vdeo
cap = cv.VideoCapture(r"video/cars1.mp4")

# Loading mask
mask = cv.imread("./mask/mask.png")

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Counter Line
line_dims = [400, 300, 680, 300]
total_count = []

while True:
    success, frame = cap.read()
    if (not success) or cv.waitKey(1) == ord("q"):
        break

    img_region = cv.bitwise_and(frame, mask)

    graphics = cv.imread("./icon/graphics.png", cv.IMREAD_UNCHANGED)
    frame = cvzone.overlayPNG(frame, graphics, (0, 0))
    results = model(img_region, stream=True)

    detections = np.empty((0, 5))

    for result in results:
        boxes = result.boxes
        for box in boxes:
        # Bounding Boxes
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Confidence Score
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            current_class = classNames[int(box.cls[0])]

            if current_class == "car":
                curent_array = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, curent_array))

    results_tracker = tracker.update(detections)
    line = cv.line(frame,(line_dims[0], line_dims[1]),(line_dims[2], line_dims[3]),(255, 0, 0),5)

    for result in results_tracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w, h = x2 - x1, y2 - y1
        bbox = x1, y1, w, h

        if current_class == "car":
            # For cvzone fancy rectangle
            cvzone.cornerRect(frame, bbox, colorR=(255, 0, 255), l=9, t=2)
            cvzone.putTextRect(frame,f"{id} {current_class} {conf}",(max(30, x1 + 10),\
                max(30, y1 - 15)),scale=1,thickness=2,offset=5)
        
        # center of the detected car
        cx, cy = x1 + w // 2, y1 + h // 2

        if (line_dims[0] < cx < line_dims[2]) and ((line_dims[1] - 10) < cy < (line_dims[3] + 20)):
            if total_count.count(id) == 0:
                line = cv.line(frame,(line_dims[0], line_dims[1]),(line_dims[2], line_dims[3]),\
                    (0, 255, 0),5)
                total_count.append(id)

    cv.putText(frame,str(len(total_count)),(255, 100),cv.FONT_HERSHEY_PLAIN,5,(255, 0, 0),8)
    cv.imshow("Image", frame)

cv.destroyAllWindows()
cap.release()
