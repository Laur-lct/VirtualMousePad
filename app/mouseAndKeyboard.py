from pymouse import PyMouse
from motionAndBlinkAnalyzer import BlinkEvent


pm = PyMouse()


# actually move the mouse pointer
def move_mouse_pointer(dx, dy):
    x, y = pm.position()
    pm.move(x + dx, y + dy)


def center_mouse():
    x, y = pm.screen_size()
    pm.move(x/2, y/2)


# maps detected blink event to mouse action
def blink_event_to_action(blinkEvent):
    x, y = pm.position()
    # there is a bug in PyMouse for Windows multi screen systems.
    # Valid on-screen coordinates can be negative, depending on positioning of the screens,
    # but PyMouse treats them as unsigned ints, and subsequent click method fails.
    # Constants below worked for my system, may need to be changed on other systems
    if x > 100000:
        x -= 4294967295

    print x, y, BlinkEvent.blink_event_to_text(blinkEvent)

    # Button is defined as 1 = left, 2 = right, 3 = middle."""
    if blinkEvent == BlinkEvent.BlinkBoth:
        pm.click(x, y, 1)
    if blinkEvent == BlinkEvent.LongBlink:
        pm.click(x, y, 1)
        pm.click(x, y, 1)
    if blinkEvent == BlinkEvent.DoubleBlink:
        pm.click(x, y, 3)

    if blinkEvent == BlinkEvent.LeftEyeClosed:
        pm.press(x, y, 1)
    if blinkEvent == BlinkEvent.LeftEyeOpened:
        pm.release(x, y, 1)
    if blinkEvent == BlinkEvent.RightEyeClosed:
        pm.press(x, y, 2)
    if blinkEvent == BlinkEvent.RightEyeOpened:
        pm.release(x, y, 2)
    return
