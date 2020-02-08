# USAGE

#  python recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle

#  python recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle


# python recognize_video.py --detector face_detection_model \
#	--embedding-model openface_nn4.small2.v1.t7 \
#	--recognizer output/recognizer.pickle \
#	--le output/le.pickle

# import the necessary packages
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils.video import FPS
from array import array 
import pyfiglet 
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

# gstreamer test
# troy feel free to delete
def gstreamer_pipeline (capture_width=3280, capture_height=2464, display_width=1000, display_height=800, framerate=21, flip_method=0) :   
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))


ascii_banner = pyfiglet.figlet_format("Team 883: Face Recognition")
print(ascii_banner)
ascii_banner = pyfiglet.figlet_format("")
print(ascii_banner) 

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--detector", required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-m", "--embedding-model", required=True,
	help="path to OpenCV's deep learning face embedding model")
ap.add_argument("-r", "--recognizer", required=True,
	help="path to model trained to recognize faces")
ap.add_argument("-l", "--le", required=True,
	help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())



# load our serialized face detector from disk
print("[INFO] loading face detector...")
#protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
#modelPath = os.path.sep.join([args["detector"],
#	"res10_300x300_ssd_iter_140000.caffemodel"])
protoPath = "/home/suavemente/X/face_detection_model/deploy.prototxt"
modelPath = "/home/suavemente/X/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch(args["embedding_model"])

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open(args["recognizer"], "rb").read())
le = pickle.loads(open(args["le"], "rb").read())

# initialize the video stream, then allow the camera sensor to warm up
#print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
#time.sleep(2.0)
vs = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
#if cap.isOpened(): # While camera is open/success
# start the FPS throughput estimator
fps = FPS().start()

face_cascade = cv2.CascadeClassifier('/home/x/opencv/data/haarcascades')

# loop over frames from the video file stream
while True:
	# grab the frame from the threaded video stream
	ret, frame = vs.read()
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert input image to GRAYSCALE Mode
        #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	#shape = predictor(frame, faces)
    	#shape = face_utils.shape_to_np(shape)
	# resize the frame to have a width of 600 pixels (while
	# maintaining the aspect ratio), and then grab the image
	# dimensions
	frame = imutils.resize(frame, width=600)
	(h, w) = frame.shape[:2]







	# construct a blob from the image
	imageBlob = cv2.dnn.blobFromImage(
		cv2.resize(frame, (300, 300)), 1.0, (300, 300),
		(104.0, 177.0, 123.0), swapRB=False, crop=False)

	# apply OpenCV's deep learning-based face detector to localize
	# faces in the input image
	detector.setInput(imageBlob)
	detections = detector.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the face
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# extract the face ROI
			face = frame[startY:endY, startX:endX]
			(fH, fW) = face.shape[:2]

			# ensure the face width and height are sufficiently large
			if fW < 20 or fH < 20:
				continue

			# construct a blob for the face ROI, then pass the blob
			# through our face embedding model to obtain the 128-d
			# quantification of the face
			faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
				(96, 96), (0, 0, 0), swapRB=True, crop=False)
			embedder.setInput(faceBlob)
			vec = embedder.forward()

			# perform classification to recognize the face
			preds = recognizer.predict_proba(vec)[0]
			j = np.argmax(preds)
			proba = preds[j]
			name = le.classes_[j]

			# draw the bounding box of the face along with the
			# associated probability
			text = "{}: {:.2f}%".format(name, proba * 100)
			y = startY - 10 if startY - 10 > 10 else startY + 10
			if (name == "mubeen" or name == "troy" or name == "waleed" or name == "joe") :
				cv2.rectangle(frame, (startX, startY), (endX, endY),
				(255, 0, 0), 2)
			else : cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0,0,255), 2) 
			cv2.putText(frame, text, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
			# to get authorization (blue color) = (255,0,0)

	# update the FPS counter
	fps.update()

	# show the output frame
	cv2.imshow("FaceReco", frame)
	key = cv2.waitKey(30) & 0xff

	# if the `ESC' key was pressed, break from the loop
	if key == 27:
		break
 
# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


