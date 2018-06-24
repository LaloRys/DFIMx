
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import RPi.GPIO as GPIO
import argparse
import imutils
import time
import dlib
import cv2

GPIO.setmode(GPIO.BCM)

for i in (23, 24, 25, 16, 20, 21):
    GPIO.setup(i, GPIO.OUT)

def euclidean_dist(ptA, ptB):
	
	return np.linalg.norm(ptA - ptB)

def eye_aspect_ratio(eye):
	A = euclidean_dist(eye[1], eye[5])
	B = euclidean_dist(eye[2], eye[4])

	C = euclidean_dist(eye[0], eye[3])

	ear = (A + B) / (2.0 * C)

	return ear

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
	help = "camino hacia donde reside la cascada de la cara")
ap.add_argument("-p", "--shape-predictor", required=True,
	help="camino al predictor de hitos faciales")
ap.add_argument("-a", "--alarm", type=int, default=0,
	help="boolean utilizado para indicar si TraffHat debería usarse")
args = vars(ap.parse_args())


if args["alarm"] > 0:
	from gpiozero import TrafficHat
	th = TrafficHat()
	print("[INFO] usando la alarma TrafficHat...")
 

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 16


COUNTER = 0
ALARM_ON = False


print("[INFO] carga predictor de hitos faciales...")
detector = cv2.CascadeClassifier(args["cascade"])
predictor = dlib.shape_predictor(args["shape_predictor"])


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

print("[INFO] Iniciando hilo de flujo de video...")
vs = VideoStream(src=0).start()

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

		ojoizquierdo = shape[lStart:lEnd]
		ojoderecho = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(ojoizquierdo)
		rightEAR = eye_aspect_ratio(ojoderecho)

		ear = (leftEAR + rightEAR) / 2.0

		leftEyeHull = cv2.convexHull(ojoizquierdo)
		rightEyeHull = cv2.convexHull(ojoderecho)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		if ear < EYE_AR_THRESH:
			COUNTER += 1

			if COUNTER >= EYE_AR_CONSEC_FRAMES:

				if not ALARM_ON:
					ALARM_ON = True

					if args["alarm"] > 0:
						th.buzzer.blink(0.1, 0.1, 10,
							background=True)

				GPIO.output(16,GPIO.HIGH)		
				cv2.putText(frame, "ALERTA DE SOMNOLENCIA!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

		else:
			COUNTER = 0
			ALARM_ON = False
			GPIO.output(16,GPIO.LOW)

		cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.stop()