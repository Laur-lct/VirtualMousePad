import threading
import time
import cv2
import numpy as np
import utils as u
import dlib


class FaceAndMovementDetector:
    def __init__(self, landmarks_file):
        self.isDetecting = False
        self.lastResultStamp = 0
        self.isFaceDetected = False
        self.detectedFaceArea = (0, 0, 0, 0)
        self.detectedEyeAreas = [[0, 0, 0, 0], [0, 0, 0, 0]]

        self.lastEyeCenters = [[0, 0], [0, 0]]

        self.__faceDetector = dlib.get_frontal_face_detector()
        self.__landmarksPredictor = dlib.shape_predictor(landmarks_file)
        self.__lastEyeHalfSize = 0
        self.__image_container = None
        self.__accumulatedMovement = 0.
        self.__moveAccumulatorThreshold = 40
        self.__intervalFound = 6000
        self.__intervalNotFound = 500
        self.__backgroundThread = None

        # motion detection
        self.trackedPoint = np.array([[0., 0.]], dtype=np.float32)  # nose
        self.__termCriteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 40, 0.03)

    # detects face and eye positions in background thread
    def __detectFaceAsync(self):
        while self.isDetecting:
            self.__accumulatedMovement = 0.
            startStamp = int(round(time.time() * 1000))
            self.isFaceDetected = False
            workImage = cv2.equalizeHist(self.__image_container['gray'])
            detections = self.__faceDetector(workImage, 0)
            if len(detections) > 0:
                face = u.biggest_dlib_rect(detections)
                self.isFaceDetected = True
                self.faceDetectionStamp = int(round(time.time() * 1000))
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                self.detectedFaceArea = (x1, y1, x2, y2)

                # update eye size
                expectedEyeHalfSize = int((x2 - x1 + y2 - y1) / 16)
                if abs(self.__lastEyeHalfSize - expectedEyeHalfSize) > 3:
                    self.__lastEyeHalfSize = expectedEyeHalfSize
                else:
                    self.__lastEyeHalfSize = int((self.__lastEyeHalfSize + expectedEyeHalfSize) / 2)

                self.__doDetectLandmarks = True

            self.lastResultStamp = int(round(time.time() * 1000))
            # sleep some time, depending on the last detection result and accumulated move
            delay = (self.__intervalFound if self.isFaceDetected else self.__intervalNotFound) - (
                self.lastResultStamp - startStamp)
            while delay > 0 and self.__accumulatedMovement < self.__moveAccumulatorThreshold and self.isDetecting:
                time.sleep(0.1)
                delay -= 100

    def start_detect_face_async(self, image_container):
        self.isDetecting = True
        self.__image_container = image_container
        self.__backgroundThread = threading.Thread(target=self.__detectFaceAsync)
        self.__backgroundThread.daemon = True
        self.__backgroundThread.start()

    def stop(self):
        self.isDetecting = False
        if self.__backgroundThread:
            self.__backgroundThread.join(1)

    # Movement detection part
    # returns relative nose point move, called every frame
    def get_relative_motion(self, curr_frame, prev_frame):
        landmarkPoints = self.__landmarksPredictor(curr_frame, dlib.rectangle(
                left=self.detectedFaceArea[0],
                top=self.detectedFaceArea[1],
                right=self.detectedFaceArea[2],
                bottom=self.detectedFaceArea[3]))
        self.__set_areas_from_landmarks(landmarkPoints, prev_frame)

        newPoints, status, err = cv2.calcOpticalFlowPyrLK(prevImg=prev_frame, nextImg=curr_frame,
                                                          prevPts=self.trackedPoint, nextPts=None,
                                                          winSize=(25, 25), maxLevel=4,
                                                          criteria=self.__termCriteria)
        relativeMove = [0., 0.]
        # point is found on new frame
        if status[0]:
            relativeMove[0] = newPoints[0][0]-self.trackedPoint[0][0]
            relativeMove[1] = newPoints[0][1]-self.trackedPoint[0][1]
            self.trackedPoint[0] = newPoints[0]

        self.__accumulatedMovement += u.distance(relativeMove, (0, 0))

        # return the nose point move
        return relativeMove

    def __set_areas_from_landmarks(self, landm, frame):
        # left eye landmarks  36, 39
        newCLeft = [(landm.part(36).x + landm.part(39).x) / 2, (landm.part(36).y + landm.part(39).y) / 2]
        self.lastEyeCenters[0] = u.updateCenter(self.lastEyeCenters[0], newCLeft)

        # right eye landmarks  42, 45
        newCRight = [(landm.part(42).x + landm.part(45).x) / 2, (landm.part(42).y + landm.part(45).y) / 2]
        self.lastEyeCenters[1] = u.updateCenter(self.lastEyeCenters[1], newCRight)

        self.detectedEyeAreas[0] = u.rect_around_center(self.lastEyeCenters[0],
                                                        self.__lastEyeHalfSize, self.__lastEyeHalfSize)
        self.detectedEyeAreas[1] = u.rect_around_center(self.lastEyeCenters[1],
                                                        self.__lastEyeHalfSize, self.__lastEyeHalfSize)
        # nose tip  landmark 33
        noseTip = [landm.part(33).x, landm.part(33).y]
        # fa = self.detectedFaceArea
        # self.detectedFaceArea = u.rect_around_center(noseTip, (fa[2]-fa[0])/2, (fa[3]-fa[1])/2)
        # if tracked point drifted too far from nose tip, adjust it
        dist = u.distance(self.trackedPoint[0], noseTip)
        if dist > self.__lastEyeHalfSize / 2:
            self.trackedPoint[0] = noseTip
            self.__accumulatedMovement += dist
            cv2.cornerSubPix(frame, self.trackedPoint,
                             (int(self.__lastEyeHalfSize / 2), int(self.__lastEyeHalfSize / 2)),
                             (-1, -1), self.__termCriteria)
