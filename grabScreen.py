from PIL import ImageGrab
from PIL import Image
import time
import numpy as np
import cv2
import mss

factor = 1
def grabScreen():
    while 1:
        start = time.time()
        bbox = (3 * 2560//8, 3 * 1440//8, 5 * 2560//8, 5 * 1440//8)
        im = ImageGrab.grab(bbox)
        img = np.array(im)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        hz = 1/(time.time()-start)
        img = cv2.resize(img,(int(np.shape(img)[1]* factor),int(np.shape(img)[0]* factor)))
        print('%.2f'%hz)
        cv2.imshow('img',img)
        key = cv2.waitKey(1)
        if key == 27:
            exit(0)

def grabOneFrame():
    start = time.time()
    bbox = (3 * 2560//8, 3 * 1440//8, 5 * 2560//8, 5 * 1440//8)
    im = ImageGrab.grab(bbox)
    img = np.array(im)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hz = 1/(time.time()-start)
    img = cv2.resize(img,(int(np.shape(img)[1]* factor),int(np.shape(img)[0]* factor)))
    return img, hz


def grabOneFrame_faster(monitor = {'left': 3 * 2560//8, 'top': 3 * 1440//8, 'width': 2560//4, 'height': 1440//4}):
    start = time.time()
    with mss.mss() as sct:
        img  = np.asarray(sct.grab(monitor),dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        hz = 1/(time.time()-start)
    return img, hz

def grabTwoWindows():
    monitor = {'left':0, 'top':0, 'width': 5*2560//8, 'height': 5*1440//8}
    start = time.time()
    with mss.mss() as sct:
        img  = np.asarray(sct.grab(monitor),dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        mon1 = {'left': 28, 'top': 110, 'width':370-28, 'height': 451-110}
        mon2 = {'left':3*2560//8, 'top':3*1440//8, 'width':2560//4, 'height':1440//4}

        mon1['right'] = mon1['left'] + mon1['width']
        mon1['down'] = mon1['top'] + mon1['height']
        mon2['right'] = mon2['left'] + mon2['width']
        mon2['down'] = mon2['top'] + mon2['height']
        img1 = img[mon1['top']:mon1['down'], mon1['left']:mon1['right'], :]
        img2 = img[mon2['top']:mon2['down'], mon2['left']:mon2['right'], :]
    return img1, img2

def grabTwoWindowsThread(queue,lock):
    monitor = {'left':0, 'top':0, 'width': 5*2560//8, 'height': 5*1440//8}
    avg = 0
    count = 1
    with mss.mss() as sct:
        while 1:
            start = time.time()
            img  = np.asarray(sct.grab(monitor),dtype=np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            mon1 = {'left': 28, 'top': 110, 'width':370-28, 'height': 451-110}
            mon2 = {'left':3*2560//8, 'top':3*1440//8, 'width':2560//4, 'height':1440//4}
            mon1['right'] = mon1['left'] + mon1['width']
            mon1['down'] = mon1['top'] + mon1['height']
            mon2['right'] = mon2['left'] + mon2['width']
            mon2['down'] = mon2['top'] + mon2['height']

            img1 = img[mon1['top']:mon1['down'], mon1['left']:mon1['right'], :]
            img2 = img[mon2['top']:mon2['down'], mon2['left']:mon2['right'], :]

            # avg += time.time() - start
            # count += 1
            lock.acquire()
            while queue.qsize():
                queue.get()
            queue.put((img1, img2, time.time() - start))
            lock.release()
            # print(time.time() - start)

def multiprocessGrabScreen(share_img, lock, monitor = {'left':3*2560//8, 'top':3*1440//8, 'width':2560//4, 'height':1440//4}, print_time = False):
  total_time = 0
  count = 0
  with mss.mss() as sct:
      while 1:
        if total_time > 1e5: #防止溢出
          total_time = 0
          count = 0
        start = time.time()
        img  = sct.grab(monitor)
        img  = Image.frombytes("RGB", img.size, img.bgra, "raw", "BGRX")
        img = np.array(img,dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_shape = ( monitor['height'],monitor['width'],3)
        lock.acquire()
        buf = np.frombuffer(share_img, dtype=np.uint8).reshape(img_shape)
        np.copyto(buf, img)
        lock.release()
        count += 1
        total_time += time.time() - start
        avg_time = total_time / count
        if print_time:
          print('Screen grab: %.2fms %.2fhz'%(1000*avg_time, 1/avg_time))

if __name__ == '__main__':
    while 1:
        img,hz = grabOneFrame_faster(monitor = {'left': 28, 'top': 110, 'width':370-28, 'height': 451-110})
        cv2.imshow('img',img)
        key = cv2.waitKey(1)
        if key == 27:
            exit()