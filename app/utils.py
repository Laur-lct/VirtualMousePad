import math
import cv2
import Tkinter
import tkMessageBox
from motionAndBlinkAnalyzer import BlinkEvent


# updates center smoothly
def updateCenter(lastCenter, newCenter):
    dist = distance(lastCenter, newCenter)
    if dist > 6:
        lastCenter = newCenter
    else:
        lastCenter[0] = int((newCenter[0] + lastCenter[0])/2)
        lastCenter[1] = int((newCenter[1] + lastCenter[1])/2)
    return lastCenter


# checks if point p is inside the given rectangle r
def point_inside_rect(p, r):
    if p[0] < r[0] or p[1] < r[1] or p[0] >= r[2] or p[1] >= r[3]:
        return False
    return True


#  returns rectangle around the center point c
def rect_around_center(c, half_width, half_height):
    return [c[0] - half_width, c[1] - half_height, c[0] + half_width, c[1] + half_height]


def biggest_dlib_rect(rects):
    maxSize = 0
    biggest = None
    for r in rects:
        sz = distance((r.left(), r.top()), (r.right(), r.bottom()))
        if maxSize < sz:
            biggest = r
            maxSize = sz
    return biggest


# simple distance between 2 points
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# draws the rectangles on image
def draw_rects(img, rects, color=(255, 0, 0)):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)


# draws the rectangles on image
def draw_points(img, points, color=(255, 0, 0)):
    for x1, y1 in points:
        cv2.circle(img, (x1, y1), 3, color, -1)


lastDrawnText = ""
showFrames = 0


# draws blink event text on image and holds it visible for a few frames
def draw_blink_event(img, blinkEvent, color=(255, 0, 0)):
    global showFrames
    global lastDrawnText
    if blinkEvent != BlinkEvent.NoBlink:
        showFrames = 30 if blinkEvent != BlinkEvent.RightEyeClosed and blinkEvent != BlinkEvent.LeftEyeClosed else 900
        lastDrawnText = BlinkEvent.blink_event_to_text(blinkEvent)

    if showFrames > 0:
        cv2.putText(img, lastDrawnText, (210, 460), cv2.FONT_HERSHEY_COMPLEX, 1, color)
        showFrames -= 1

help_text = "Attention! This application can control your mouse." \
            "\nMouse capture is DISABLED by default" \
            "\nDismiss this note, try blinking and check the preview window" \
            "\nPress 'z' to toggle mouse capture when done practicing" \
            "\n" \
            "\nActions mapping:" \
            "\nRegular blink - left click" \
            "\nLong blink - double click" \
            "\nDouble blink - middle click" \
            "\nLeft eye close/open - left button press/release" \
            "\nRight eye close/open - right button press/release"


# shows a system info dialog with specified title and text
def showHelpMessageBox(title, text=help_text):
    root = Tkinter.Tk()
    root.withdraw()
    tkMessageBox.showinfo(title, text)
