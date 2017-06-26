import math
import time
from collections import deque


# enumeration class
class BlinkEvent:
    # nothing happened
    NoBlink = 0
    # blink detected, but considered as natural blinking
    NaturalBlink = 1
    # regular both eyes blink. This and below treated as user input
    BlinkBoth = 2
    # long blink both eyes
    LongBlink = 3
    # quick 2-time blink
    DoubleBlink = 4
    # only left eye closed
    LeftEyeClosed = 5
    # only left eye opened
    LeftEyeOpened = 6
    # only right eye closed
    RightEyeClosed = 7
    # only right eye opened
    RightEyeOpened = 8

    @staticmethod
    def blink_event_to_text(blinkEventValue):
        if blinkEventValue == BlinkEvent.NoBlink:
            return "NoBlink"
        elif blinkEventValue == BlinkEvent.NaturalBlink:
            return "NaturalBlink"
        elif blinkEventValue == BlinkEvent.BlinkBoth:
            return "BlinkBoth"
        elif blinkEventValue == BlinkEvent.LongBlink:
            return "LongBlink"
        elif blinkEventValue == BlinkEvent.DoubleBlink:
            return "DoubleBlink"
        elif blinkEventValue == BlinkEvent.LeftEyeClosed:
            return "LeftEyeClosed"
        elif blinkEventValue == BlinkEvent.LeftEyeOpened:
            return "LeftEyeOpened"
        elif blinkEventValue == BlinkEvent.RightEyeClosed:
            return "RightEyeClosed"
        elif blinkEventValue == BlinkEvent.RightEyeOpened:
            return "RightEyeOpened"


# Class aggregates filtering algorithms for pointer movement and detected eye openness probabilities.
# basically, it maps the tracked point delta move to mouse pointer delta move
# and eye open state estimates to one of the BlinkEvents
class MotionAndBlinkAnalyzer:
    def __init__(self):
        self.filterNaturalBlinks = True
        self.naturalBlinkDelay = 6000
        self.longBlinkDelay = 400
        self.doubleBlinkDelay = 370

        nowMs = int(round(time.time() * 1000))
        # timestamps of last blink event start % end. Starting from now by default
        self.__lastBlinkBothStamp = nowMs
        self.__lastBlinkOneStamp = nowMs
        self.__lastBlinkEventStartStamp = nowMs
        # detected ongoing blink event
        self.__startedBlinkEvent = BlinkEvent.NoBlink
        # accumulated pointer move since last blink event
        self.__accumulatedMovement = 0.
        # minimal time allowed between consequent blink detections
        self.__minBLinkBothInterval = 210
        # minimal time allowed between consequent one eye blink detections
        self.__minBlinkOneInterval = 350
        # stored eye openness probabilities
        self.__predictions = deque([], maxlen=10)
        self.__predictions.append([1.0, 1.0])

        # mouse pointer movement public settings
        self.reverseX = False
        self.reverseY = False
        self.mouseMoveEnabled = True

        # last delta moves
        self.__m_dxLast = 0.
        self.__m_dyLast = 0.
        # sensitivity factors
        self.__m_sensFactorX = 1
        self.__m_sensFactorY = 1
        # pointer 'mass', used in smoothness
        self.__m_actualMotionWeight = 0.
        # jitter threshold
        self.__m_minDeltaThreshold = 0.7
        # pointer acceleration values
        self.__m_accelerationLevel = 0
        self.__m_accelerationArray = [1] * 10

        # set initial settings
        self.set_acceleration_level(3)
        self.set_sensitivity(25)
        self.set_smoothness(10.)

    # fills the __m_accelerationArray with desired factors
    def __fill_acceleration_array(self, delta1, factor1, delta2=9999, factor2=1.0):
        listLen = len(self.__m_accelerationArray)
        if delta1 > listLen:
            delta1 = listLen
        if delta2 > listLen:
            delta2 = listLen

        for i in range(delta1):
            self.__m_accelerationArray[i] = 1
        for i in range(delta1, delta2):
            self.__m_accelerationArray[i] = factor1
        j = 0.
        for i in range(delta2, listLen):
            self.__m_accelerationArray[i] = factor1 * factor2 + j
            j += 0.1

    # sets the pointer acceleration setting level from 0 to 5
    def set_acceleration_level(self, accelLevel=2):
        if accelLevel > 5:
            accelLevel = 5
        self.__m_accelerationLevel = accelLevel
        if accelLevel == 0:
            self.__fill_acceleration_array(9999, 1)
        elif accelLevel == 1:
            self.__fill_acceleration_array(7, 1.5)
        elif accelLevel == 2:
            self.__fill_acceleration_array(7, 2.0)
        elif accelLevel == 3:
            self.__fill_acceleration_array(7, 1.5, 14, 2.0)
        elif accelLevel == 4:
            self.__fill_acceleration_array(7, 2.0, 14, 1.5)
        else:
            self.__fill_acceleration_array(7, 2.0, 14, 2.0)

    # sets pointer smoothness, from 2 to 20
    def set_smoothness(self, smoothness):
        smoothness /= 2
        if smoothness < 1.05:
            smoothness = 1.05
        elif smoothness > 9.:
            smoothness = 9.
        self.__m_actualMotionWeight = math.log10(smoothness)

    # sets pointer sensitivity, from 0 to 50
    def set_sensitivity(self, sens):
        if sens > 50:
            sens = 50
        sensY = sens + 2
        if sensY > 50:
            sensY = 50
        self.__m_sensFactorX = math.pow(math.e, sens/10.)
        self.__m_sensFactorY = math.pow(math.e, sensY/10.)

    # returns filtered delta move of mouse pointer in pixels. Input is track point dx and dy
    def get_mouse_pointer_move(self, dx, dy):
        # apply sensitivity
        dx *= self.__m_sensFactorX
        dy *= self.__m_sensFactorY

        # apply low-pass filter
        dx = dx * (1.0 - self.__m_actualMotionWeight) + self.__m_dxLast * self.__m_actualMotionWeight
        dy = dy * (1.0 - self.__m_actualMotionWeight) + self.__m_dyLast * self.__m_actualMotionWeight
        self.__m_dxLast = dx
        self.__m_dyLast = dy

        # apply acceleration
        dist = math.sqrt(dx * dx + dy * dy)
        if self.__m_accelerationLevel > 0:
            arrIdx = int(round(dist))
            if arrIdx >= len(self.__m_accelerationArray):
                arrIdx = len(self.__m_accelerationArray) - 1
            dx *= self.__m_accelerationArray[arrIdx]
            dy *= self.__m_accelerationArray[arrIdx]

        moveValue = [0, 0]
        if self.mouseMoveEnabled:
            # apply delta threshold
            if abs(dx) > self.__m_minDeltaThreshold:
                moveValue[0] = int(round(dx))
            if abs(dy) > self.__m_minDeltaThreshold:
                moveValue[1] = int(round(dy))

            if moveValue[0] != 0 or moveValue[1] != 0:
                if self.reverseX:
                    moveValue[0] *= -1
                if self.reverseY:
                    moveValue[1] *= -1
                self.__accumulatedMovement += math.sqrt(dx * dx + dy * dy)

        return moveValue

    # takes eye openness probabilities and returns the detected blink event (see BlinkEvent enum class)
    def analyze_blink_event(self, probs):
        # simple analysis for now, considering only current and previous probabilities
        isOpenLPrev = self.__predictions[-1][0] > 0.5
        isOpenRPrev = self.__predictions[-1][1] > 0.5
        isOpenL = probs[0] > 0.5
        isOpenR = probs[1] > 0.5
        trendL = probs[0] - self.__predictions[-1][0]
        trendR = probs[1] - self.__predictions[-1][1]
        self.__predictions.append(probs)

        returnedEvent = BlinkEvent.NoBlink
        nowMs = int(round(time.time() * 1000))
        isTimeToAnalyze = nowMs - self.__lastBlinkBothStamp > self.__minBLinkBothInterval
        if isTimeToAnalyze:
            # no ongoing blink event, check for start
            if self.__startedBlinkEvent == BlinkEvent.NoBlink:
                self.__startedBlinkEvent = \
                    self.__checkStartEventBothEyes(nowMs, isOpenL, isOpenR) \
                    or self.__checkStartOneEyeEvent(nowMs, BlinkEvent.LeftEyeClosed,
                                                    isOpenL, isOpenR, isOpenLPrev, isOpenRPrev, trendR) \
                    or self.__checkStartOneEyeEvent(nowMs, BlinkEvent.RightEyeClosed,
                                                    isOpenR, isOpenL, isOpenRPrev, isOpenLPrev, trendL)
                # these events are continuous
                if self.__startedBlinkEvent == BlinkEvent.LeftEyeClosed \
                        or self.__startedBlinkEvent == BlinkEvent.RightEyeClosed:
                    returnedEvent = self.__startedBlinkEvent

            # ongoing blink event, check for end
            else:
                if self.__startedBlinkEvent <= BlinkEvent.DoubleBlink:
                    returnedEvent = self.__checkEndEventBothEyes(nowMs, isOpenL, isOpenR)
                elif self.__startedBlinkEvent == BlinkEvent.LeftEyeClosed:
                    returnedEvent = self.__checkEndOneEyeEvent(nowMs, BlinkEvent.LeftEyeOpened,
                                                               isOpenL, isOpenR, isOpenLPrev, isOpenRPrev)
                elif self.__startedBlinkEvent == BlinkEvent.RightEyeClosed:
                    returnedEvent = self.__checkEndOneEyeEvent(nowMs, BlinkEvent.RightEyeOpened,
                                                               isOpenR, isOpenL, isOpenRPrev, isOpenLPrev)

                if returnedEvent != BlinkEvent.NoBlink:
                    self.__startedBlinkEvent = BlinkEvent.NoBlink

        return returnedEvent

    def __checkStartEventBothEyes(self, nowMs, openL, openR):
        # just reacts on both eyes closed
        if not openL and not openR:
            self.__lastBlinkEventStartStamp = nowMs
            self.mouseMoveEnabled = False
            return BlinkEvent.BlinkBoth
        return BlinkEvent.NoBlink

    def __checkStartOneEyeEvent(self, nowMs, retEvent, openLR, openOther, openLRPrev, openOtherPrev, trendOther):
        if not openLR and not openLRPrev and openOther and openOtherPrev and trendOther > -0.1:
            if nowMs - self.__lastBlinkOneStamp > self.__minBlinkOneInterval:
                self.__lastBlinkEventStartStamp = nowMs
                if retEvent == BlinkEvent.RightEyeClosed:
                    self.mouseMoveEnabled = False
                return retEvent
        return BlinkEvent.NoBlink

    def __checkEndEventBothEyes(self, nowMs, openL, openR):
        retEvent = BlinkEvent.NoBlink
        # just reacts on both eyes opened
        if openL and openR:
            retEvent = BlinkEvent.BlinkBoth
            # depending on timings, this can be adjusted to the long, natural or double BlinkEvent
            if nowMs - self.__lastBlinkEventStartStamp > self.longBlinkDelay:
                retEvent = BlinkEvent.LongBlink
            elif self.filterNaturalBlinks \
                    and nowMs - self.__lastBlinkBothStamp > self.naturalBlinkDelay\
                    and self.__accumulatedMovement < 80:
                retEvent = BlinkEvent.NaturalBlink
            elif nowMs - self.__lastBlinkBothStamp < self.doubleBlinkDelay:
                retEvent = BlinkEvent.DoubleBlink

            self.__lastBlinkBothStamp = nowMs
            self.__accumulatedMovement = 0.0
            self.mouseMoveEnabled = True

        return retEvent

    def __checkEndOneEyeEvent(self, nowMs, retEvent, openLR, openOther, openLRPrev, openOtherPrev):
        if openLR and openLRPrev and openOther and openOtherPrev:
            self.__lastBlinkOneStamp = nowMs
            self.__accumulatedMovement = 0.0
            self.mouseMoveEnabled = True
            return retEvent
        return BlinkEvent.NoBlink
