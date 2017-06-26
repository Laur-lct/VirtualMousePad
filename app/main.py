__author__ = "Roman Semenyk"
__copyright__ = "Copyright 2017, VirtualMousePad"
__license__ = "GPLv3"
__version__ = "1.0.0"
__email__ = "r.semenyk(at)gmail.com"

import os
import cv2
import time
import numpy as np
import utils as u
import threading
import mouseAndKeyboard as mouse

# paths & constants
landmarks_fn = './classifier/shape_predictor_68_face_landmarks.dat'
if not os.path.exists(landmarks_fn):
    print "Please copy 'shape_predictor_68_face_landmarks.dat' from D-lib python_examples to ./classifier folder."
    print "You can download the file here:"
    print "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    exit()
nn_definition_file = 'classifier/model_deploy.prototxt'
nn_weights_file = 'classifier/model_weights_97.22.caffemodel'

from collections import deque
from faceDetector import FaceAndMovementDetector
from blinkDetector import BlinkDetector
from motionAndBlinkAnalyzer import MotionAndBlinkAnalyzer
from motionAndBlinkAnalyzer import BlinkEvent


# globals
fd = FaceAndMovementDetector(landmarks_fn)
bd = BlinkDetector(nn_definition_file, nn_weights_file)
ma = MotionAndBlinkAnalyzer()
imgContainer = {'gray': None, 'prev': None, 'vis': None}
showHelpPopup = True
mouseCaptureEnabled = False


# method to grab frames from the web camera and show FPS
stopFlag = False
def grab_frames():
    fpsQ = deque([], maxlen=5)
    lastFrameMs = int(round(time.time() * 1000))

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 25)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while not stopFlag:
        ret, img = cam.read()
        imgContainer['prev'] = imgContainer['gray']
        flipped = cv2.flip(img, 1)
        imgContainer['gray'] = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
        imgContainer['vis'] = flipped
        # draw FPS
        nowMs = int(round(time.time() * 1000))
        fpsQ.append(int(round(1000 / (nowMs - lastFrameMs + 1))))
        lastFrameMs = nowMs
        cv2.putText(flipped, 'cam FPS: %.0f' % np.mean(fpsQ), (25, 25), cv2.FONT_HERSHEY_COMPLEX, 1, 255)


# start the web cam grabber in separate thread
thread = threading.Thread(target=grab_frames)
thread.daemon = True
thread.start()
# wait for the first frame
while imgContainer['vis'] is None:
    time.sleep(0.05)

# start async face detector
fd.start_detect_face_async(image_container=imgContainer)

if showHelpPopup:
    u.showHelpMessageBox("Welcome Note")
    mouse.center_mouse()

# main loop
# regardless of camera actual frame rate, the processing loop is maintained at 30 FPS
# this allows smooth mouse moves even on low cam FPS. This approach should be optimized in future.
lastFaceDetectionTs = 0
while True:
    startTime = time.time()
    vis = imgContainer['vis']

    if fd.isFaceDetected and fd.lastResultStamp != lastFaceDetectionTs:
        lastFaceDetectionTs = fd.lastResultStamp
        u.draw_rects(vis, [fd.detectedFaceArea], (0, 255, 0))

    if int(round(time.time() * 1000)) - lastFaceDetectionTs > 20000:
        lastFaceDetectionTs = 0

    if lastFaceDetectionTs > 0:
        relMove = fd.get_relative_motion(imgContainer['gray'], imgContainer['prev'])
        relMoveFiltered = ma.get_mouse_pointer_move(relMove[0], relMove[1])
        blinkEvent = BlinkEvent.NoBlink
        # check state only if mouse is not moving
        if u.distance(relMoveFiltered, [0, 0]) < 5:
            lblink, rblink = bd.predict_states(imgContainer['gray'], fd.detectedEyeAreas[0], fd.detectedEyeAreas[1])
            blinkEvent = ma.analyze_blink_event((lblink, rblink))

        if mouseCaptureEnabled:
            mouse.move_mouse_pointer(relMoveFiltered[0], relMoveFiltered[1])
            if blinkEvent != BlinkEvent.NoBlink:
                mouse.blink_event_to_action(blinkEvent)
        else:
            cv2.putText(vis, 'press \'z\' to toggle mouse capture', (20, 220), cv2.FONT_HERSHEY_COMPLEX, 1, 255)

        # visualise
        u.draw_rects(vis, fd.detectedEyeAreas)
        u.draw_points(vis, fd.trackedPoint)
        u.draw_blink_event(vis, blinkEvent)
    else:
        cv2.putText(vis, 'detecting face', (210, 460), cv2.FONT_HERSHEY_COMPLEX, 1, 255)

    cv2.imshow('Preview', vis)

    endTime = time.time()
    # print "Processing delay ", int(round((endTime - startTime) * 1000)), " ms"

    # limit processing speed up to 30 FPS
    if endTime - startTime < 0.032:
        time.sleep(0.032 - (endTime - startTime))

    key = cv2.waitKey(2)
    if key == 27:
        break
    elif key == ord('z'):
        mouseCaptureEnabled = not mouseCaptureEnabled

# stop all
cv2.destroyAllWindows()
stopFlag = True
fd.stop()
thread.join(3)
os._exit(0)
