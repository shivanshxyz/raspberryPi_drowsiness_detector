
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def eucdDistance(ptA, ptB):
	return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
	A = eucdDistance(eye[1], eye[5])
	B = eucdDistance(eye[2], eye[4])
	C = eucdDistance(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear


if args["alarm"] > 0:
	from gpiozero import TrafficHat
	th = TrafficHat()

 
EYE_THRESHOLD = 0.3
EYE_ASPECT_RATIO = 16
COUNTER = 0
ALARM_ON = False

detector = cv2.CascadeClassifier(args["cascade"])
predictor = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

vs = VideoStream(usePiCamera=True).start()
time.sleep(1.0)

while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

	for (x, y, w, h) in rects:
		rect = dlib.rectangle(int(x), int(y), int(x + w),
			int(y + h))
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		left_EAR = eye_aspect_ratio(leftEye)
		right_EAR = eye_aspect_ratio(rightEye)

		ear = (left_EAR + right_EAR) / 2.0

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		if ear < EYE_THRESHOLD:
			COUNTER += 1

			if COUNTER >= EYE_ASPECT_RATIO:
				if not ALARM_ON:
					ALARM_ON = True

					if args["alarm"] > 0:
						th.buzzer.blink(0.1, 0.1, 10,
							background=True)

				cv2.putText(frame, "ALERT!!!! YOU ARE DROWSY", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		else:
			COUNTER = 0
			ALARM_ON = False 

		cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()