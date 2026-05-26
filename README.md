# workshop2-Object-Detection-using-Web-camera

## Aim
To perform real-time object detection using webcam with OpenCV DNN and MobileNet-SSD using Python.

## Software Required
- Anaconda – Python 3.7
- OpenCV
- NumPy

## Algorithm
Step 1:  Load the necessary packages.

Step 2: Load the MobileNet-SSD model files.

Step 3: Access the webcam using OpenCV.

Step 4: Capture video frames from the webcam.

Step 5: Convert the frame into blob format and perform object detection.

Step 6: Draw bounding boxes and labels around detected objects.

Step 7: Display the output frame with detected objects.

## Program
## Developed By : Harini S
## Register Number : 212224240049

```python
import cv2
import numpy as np
import time

# Load model files
PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL = "MobileNetSSD_deploy.caffemodel"

# Class labels
CLASSES = ["background","aeroplane","bicycle","bird","boat",
 "bottle","bus","car","cat","chair","cow","diningtable",
 "dog","horse","motorbike","person","pottedplant",
 "sheep","sofa","train","tvmonitor"]

CONF_THRESH = 0.5
FONT = cv2.FONT_HERSHEY_SIMPLEX

# Load the network
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise SystemExit("Webcam not accessible")

try:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        (h, w) = frame.shape[:2]

        # Create blob
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300,300)),
            0.007843,
            (300,300),
            127.5
        )

        net.setInput(blob)
        detections = net.forward()

        # Draw detections
        for i in range(detections.shape[2]):

            conf = float(detections[0,0,i,2])

            if conf > CONF_THRESH:

                idx = int(detections[0,0,i,1])

                box = detections[0,0,i,3:7] * np.array([w,h,w,h])

                (startX, startY, endX, endY) = box.astype("int")

                label = f"{CLASSES[idx]}: {conf:.2f}"

                cv2.rectangle(frame,
                              (startX,startY),
                              (endX,endY),
                              (0,255,0), 2)

                y = startY - 15 if startY > 30 else startY + 15

                cv2.putText(frame,
                            label,
                            (startX,y),
                            FONT,
                            0.5,
                            (0,255,0),
                            2)

        # Display output
        cv2.imshow("Object Detection", frame)

        # Press q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:

    cap.release()
    cv2.destroyAllWindows()
```
## Output
Original Webcam Detection<br> 

<img width="811" height="633" alt="image" src="https://github.com/user-attachments/assets/67dc20ad-4592-4f69-a4ff-8cea2f5f22e9" />

## Result
Thus the real-time objects are successfully detected using OpenCV DNN and MobileNet-SSD with webcam using Python.
