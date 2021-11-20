import time
from grabScreen import grabScreen, grabOneFrame
from objectDetection import objectDetection
from objectSelection import objectSelection
import pyautogui
import cv2
def processing():
    start_time = time.time()
    while 1:
        if time.time() - start_time > 60: # 超时自动退出
            exit()
        img, hz = grabOneFrame()
        candidates = objectDetection(img)        
        target = objectSelection(candidates)
        if target is not None:
            pyautogui.moveTo(x=target[0], y=target[1],duration=0, tween=pyautogui.linear)
            # pyautogui.click(x=None, y=None, clicks=1, interval=0, button='left', duration=0.0, tween=pyautogui.linear)
        


if __name__ == '__main__':
    processing()