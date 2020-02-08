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


if __name__ == '__main__':
    #show_camera()
    face_detect() 
    #blink_detect()
