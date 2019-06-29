# USAGE
# python faceDetectionDNNMultiThread.py --prototxt deploy.prototxt.txt --model res10_300x300_ssd_iter_140000.caffemodel

from imutils.video import VideoStream
from imutils.video import FPS
from multiprocessing import Process
from multiprocessing import Queue
from ubidots import ApiClient
import numpy as np
import argparse
import imutils
import time
import cv2

def overlap(startX1, startY1, endX1, endY1, startX2, startY2, endX2, endY2, previousTime, currentTime):
    if currentTime - previousTime > 5:
        return False
    hoverlaps = (startX1 <= endX2) and (endX1 >= startX2)
    voverlaps = (startY1 <= endY2) and (endY1 >= startY2)
    return hoverlaps and voverlaps

def classify_frame(net, inputQueue, outputQueue):
	while True:
		if not inputQueue.empty():
			frame = inputQueue.get()
			blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
			net.setInput(blob)
			detections = net.forward()
			outputQueue.put(detections)

def save_count_ubidots():
    while True:
        currentTime = time.time()
        if currentTime - startTime > 120:
            startTime = time.time()
            savedValue = ubidotsCount.save_value({'value': count})
            print("Saved count in ubidots: ", savedValue)


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

inputQueue = Queue(maxsize=1)
outputQueue = Queue(maxsize=1)
detections = None
count = 0
peopleInLastFrame = 0
lastDetection = None
lastDetectionTime = None
startTime = time.time()
api = ApiClient(token='A1E-ef10fe32d5c9ff6ced2fb6eaaeb880cc1037')
ubidotsCount = api.get_variable('5d074d92c03f970688cf8483')

print("[INFO] starting face detection process...")
p = Process(target=classify_frame, args=(net, inputQueue, outputQueue,))
p.daemon = True
p.start()

print("[INFO] starting ubidots process...")
p = Process(target=save_count_ubidots)
p.daemon = True
p.start()

# initialize the video stream and allow the cammera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (h, w) = frame.shape[:2]

    if inputQueue.empty():
        inputQueue.put(frame)

    if not outputQueue.empty():
        detections = outputQueue.get()

    peopleInThisFrame = 0
    if detections is not None:
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence < args["confidence"]:
                continue

            # compute the (x, y)-coordinates of the bounding box
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            isOverlapping = False
            if lastDetection is not None:
                (lastStartX, lastStartY, lastEndX, lastEndY) = lastDetection
                isOverlapping = overlap(startX, startY, endX, endY, lastStartX, lastStartY, lastEndX, lastEndY, lastDetectionTime, time.time())

            if not isOverlapping:
                peopleInThisFrame = peopleInThisFrame + 1
                if peopleInThisFrame > peopleInLastFrame:
                    count = count + (peopleInThisFrame - peopleInLastFrame)

            # draw the bounding box of the face along with the associated probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            lastDetection = (startX, startY, endX, endY)
            lastDetectionTime = time.time()
                        
    peopleInLastFrame = peopleInThisFrame

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()
