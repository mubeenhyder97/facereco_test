#pip install pyfiglet 

import sys 
import pyfiglet
import cv2
import numpy as np 
import pickle 
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
from scipy.spatial import distance as dist
import argparse

ascii_banner = pyfiglet.figlet_format("Welcome")
print(ascii_banner)
ascii_banner = pyfiglet.figlet_format("Mubeen")
print(ascii_banner) 

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
### frontal face detector 68 very slow
# could try: shape_predictor_5_face_landmarks.dat
ap.add_argument("-p", "--shape-predictor",default="shape_predictor_68_face_landmarks.dat",
	help="path to facial landmark predictor")
#ap.add_argument("-p", "--shape-predictor",default="shape_predictor_5_face_landmarks.dat",
#	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="camera",
	help="path to input video file")
ap.add_argument("-t", "--threshold", type = float, default=0.27,
	help="threshold to determine closed eyes")
ap.add_argument("-f", "--imgs", type = int, default=2,
	help="the number of consecutive imgs the eye must be below the threshold")


def gstreamer_pipeline (capture_width=3280, capture_height=2464, display_width=800, display_height=600, framerate=21, flip_method=2) :   
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))


def face_detect() :
    
    # variables for blink detection
    args = vars(ap.parse_args())
    EYE_AR_THRESH = args['threshold']
    EYE_AR_CONSEC_imgS = args['imgs']

    # initialize the img counters and the total number of blinks
    COUNTER = 0
    TOTAL = 0

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args["shape_predictor"])
 
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    ### Load OpenCV Cascade Modules for face & eye 
    face_cascade = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_eye.xml')
    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if cap.isOpened(): # While camera is open/success
	ascii_banner = pyfiglet.figlet_format("Face Detection in-Progress")

	print(ascii_banner)
        cv2.namedWindow('FaceReco_Detect', cv2.WINDOW_AUTOSIZE)
        while cv2.getWindowProperty('FaceReco_Detect',0) >= 0:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert input image to GRAYSCALE Mode
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	    blinkfaces = detector(gray, 0)
		
            for (x,y,w,h) in faces:
		#ascii_banner = pyfiglet.figlet_format("Face identified")
		print("face identified") # console output 
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,'Face ID',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)
		
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex,ey,ew,eh) in eyes:
		    	#ascii_banner = pyfiglet.figlet_format("Facial Feature: eye: identified")
		    	print("facial feature: eye: identified") # console output 
                    	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	    ### blink detection start
		# determine the facial landmarks for the face region, then
    		# convert the facial landmark (x, y)-coordinates to a NumPy
    		# array
		#shape = shape_predictor(gray,face)
	    for face in blinkfaces: 
    		shape = predictor(gray, face)
    		shape = face_utils.shape_to_np(shape)
    
    		# extract the left and right eye coordinates, then use the
    		# coordinates to compute the eye aspect ratio for both eyes
    		leftEye = shape[lStart:lEnd]
    		rightEye = shape[rStart:rEnd]
    		leftEAR = eye_aspect_ratio(leftEye)
    		rightEAR = eye_aspect_ratio(rightEye)
    
    		# average the eye aspect ratio together for both eyes
    		ear = (leftEAR + rightEAR) / 2.0
    
    		# compute the convex hull for the left and right eye, then
    		# visualize each of the eyes
    		leftEyeHull = cv2.convexHull(leftEye)
    		rightEyeHull = cv2.convexHull(rightEye)
    		cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
    		cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
    		# threshold, and if so, increment the blink img counter
    		if ear < EYE_AR_THRESH:
    			COUNTER += 1
    
    		# otherwise, the eye aspect ratio is not below the blink
    		# threshold
    		else:
    			# if the eyes were closed for a sufficient number of
    			# then increment the total number of blinks
    			if COUNTER >= EYE_AR_CONSEC_imgS:
    				TOTAL += 1
				#ascii_banner = pyfiglet.figlet_format("Human Detection: True")
				#print(ascii_banner)
    
    			# reset the eye img counter
    			COUNTER = 0
    
    		# draw the total number of blinks on the img along with
    		# the computed eye aspect ratio for the img
    		cv2.putText(img, "Blinks: {}".format(TOTAL), (10, 30),
    			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    		cv2.putText(img, "Aspect Ratio: {:.2f}".format(ear), (300, 30),
    			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		print("Total Blinks:", TOTAL)
	    ### blink detection end 
            cv2.imshow('FaceReco_Detect',img)
            keyCode = cv2.waitKey(30) & 0xff
            # Use ESC Key to End Detection
            if keyCode == 27:
		print("Total Face Detections:")
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


#def blink_detect(): 
    # variables for blink detection
    #args = vars(ap.parse_args())
    #EYE_AR_THRESH = args['threshold']
    #EYE_AR_CONSEC_imgS = args['imgs']

    # initialize the img counters and the total number of blinks
    #COUNTER = 0
    #TOTAL = 0

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    #print("loading facial landmark predictor...")
    #detector = dlib.get_frontal_face_detector()
    #predictor = dlib.shape_predictor(args["shape_predictor"])
 
    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    #(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    #(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    #blinkfaces = detector(gray, 0)

    #for face in blinkfaces: 
    	#shape = predictor(gray, face)
    	#shape = face_utils.shape_to_np(shape)
    
   	# extract the left and right eye coordinates, then use the
    	# coordinates to compute the eye aspect ratio for both eyes
    	#leftEye = shape[lStart:lEnd]
    	#rightEye = shape[rStart:rEnd]
    	#leftEAR = eye_aspect_ratio(leftEye)
    	#rightEAR = eye_aspect_ratio(rightEye)
    
    	# average the eye aspect ratio together for both eyes
    	#ear = (leftEAR + rightEAR) / 2.0
    
    	# compute the convex hull for the left and right eye, then
    	# visualize each of the eyes
    	#leftEyeHull = cv2.convexHull(leftEye)
    	#rightEyeHull = cv2.convexHull(rightEye)
    	#cv2.drawContours(img, [leftEyeHull], -1, (0, 255, 0), 1)
   	#cv2.drawContours(img, [rightEyeHull], -1, (0, 255, 0), 1)
	# check to see if the eye aspect ratio is below the blink
   	# threshold, and if so, increment the blink img counter
    	#if ear < EYE_AR_THRESH:
    	#	COUNTER += 1
    
    	# otherwise, the eye aspect ratio is not below the blink
    	# threshold
    	#else:
    	# if the eyes were closed for a sufficient number of
    	# then increment the total number of blinks
    	#	if COUNTER >= EYE_AR_CONSEC_imgS:
    	#		TOTAL += 1
			#ascii_banner = pyfiglet.figlet_format("Human Detection: True")
			#print(ascii_banner)
    
    			# reset the eye img counter
    	#		COUNTER = 0
    
    		# draw the total number of blinks on the img along with
    		# the computed eye aspect ratio for the img
    	#		cv2.putText(img, "Blinks: {}".format(TOTAL), (10, 30),
    	#		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    	#		cv2.putText(img, "Aspect Ratio: {:.2f}".format(ear), (300, 30),
    	#		cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	#		print("Total Blinks:", TOTAL)

if __name__ == '__main__':
    #show_camera()
    face_detect() 
    #blink_detect()
