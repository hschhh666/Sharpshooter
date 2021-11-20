from ctypes import *
import PyHook3 as pyHook
import pythoncom
import time
import multiprocessing
import pythoncom
from grabScreen import grabScreen, grabOneFrame
from objectSelection import objectSelection
import pyautogui
import cv2
import mmcv
from mmskeleton.apis import init_pose_estimator, inference_pose_estimator
import numpy as np
import win32api
import win32con
import pydirectinput
p = None

def test():
    while 1:
        print(time.time())
        time.sleep(1)
        pydirectinput.move(300, None)
        time.sleep(0.1)
        pydirectinput.click()


cfg = mmcv.Config.fromfile('E:/MyProject/CSGOAI/mmskeleton-master/mmskeleton-master/configs/apis/pose_estimator.cascade_rcnn+hrnet.yaml')
model = init_pose_estimator(**cfg, device=0)


def objectDetection(frame):
  result = inference_pose_estimator(model, frame)
  if result['joint_preds'] is None:
    return None
  joint_preds = result['joint_preds']
  people_num = np.shape(joint_preds)[0]
  joint_num = np.shape(joint_preds)[1]
  print('%d enemies detected'%people_num)
  head = []
  for i in range(people_num):
    for j in range(1):
      head.append(joint_preds[i,j,:])
  head = np.array(head, dtype=int)
  print(head)
  return head

def processing():
    start_time = time.time()
    while 1:
        if time.time() - start_time > 600: # 超时自动退出
            exit()
        img, hz = grabOneFrame()
        candidates = objectDetection(img)        
        target = objectSelection(candidates)
        if target is not None:
            currentMouseX, currentMouseY = pyautogui.position()
            print('mouse X: %d, mouse Y: %d'%(currentMouseX, currentMouseY))
            targetX = target[0]
            targetY = target[1]
            print('target X: %d, target Y: %d'%(targetX, targetY))
            deltaX, deltaY = int(targetX - currentMouseX), int(targetY - currentMouseY)          
            print('delta X: %d, delta Y :%d'%(deltaX, deltaY))  
            cv2.circle(img, (targetX, targetY), 8, (0,0,255), thickness = -1)
            factor = 0.5
            img = cv2.resize(img,(int(np.shape(img)[1]* factor),int(np.shape(img)[0]* factor)))
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            cv2.moveWindow("img", 2560, 100)
            cv2.imshow('img',img)
            cv2.waitKey(10)
            k = 3842/360
            f = 954.39
            deltaX = int((np.arctan(deltaX/f)*180/np.pi) * k)
            deltaY = int((np.arctan(deltaY/f)*180/np.pi) * k)
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, deltaX, deltaY)
            # exit()
            pyautogui.click(x=None, y=None, clicks=3, interval=0.1, button='left', duration=0.0, tween=pyautogui.linear)

  
 
# 鼠标事件处理函数
def OnMouseEvent(event):
    # print('------------------')
    # print('MessageName:',event.MessageName)  #事件名称
    # print('Message:',event.Message)          #windows消息常量 
    # print('Time:',event.Time)                #事件发生的时间戳        
    # print('Window:',event.Window)            #窗口句柄         
    # print('WindowName:',event.WindowName)    #窗口标题
    # print('Position:',event.Position)        #事件发生时相对于整个屏幕的坐标
    # print('Wheel:',event.Wheel)              #鼠标滚轮
    # print('Injected:',event.Injected)        #判断这个事件是否由程序方式生成，而不是正常的人为触发。
    # print('------------------')
    if event.MessageName == 'mouse middle down': #当点击鼠标中键时，启动/终止子进程
        global p
        if p:
            print('close sub-process')
            p.terminate()
            p.join()
            p = None
        else:
            print('start sub-process')
            p = multiprocessing.Process(target=processing)
            p.start()
            pass
        
    # 返回True代表将事件继续传给其他句柄，为False则停止传递，即被拦截。换句话说，当被拦截时，其他程序就不知道鼠标被点了。
    return True

#键盘事件处理函数
def OnKeyboardEvent(event):
    # print('MessageName:',event.MessageName)          #同上，共同属性不再赘述
    # print('Message:',event.Message)
    # print('Time:',event.Time)
    # print('Window:',event.Window)
    # print('WindowName:',event.WindowName)
    # print('Ascii:', event.Ascii, chr(event.Ascii))   #按键的ASCII码
    # print('Key:', event.Key)                         #按键的名称
    # print('KeyID:', event.KeyID)                     #按键的虚拟键值
    # print('ScanCode:', event.ScanCode)               #按键扫描码
    # print('Extended:', event.Extended)               #判断是否为增强键盘的扩展键
    # print('Injected:', event.Injected)
    # print('Alt', event.Alt)                          #是某同时按下Alt
    # print('Transition', event.Transition)            #判断转换状态
    # print('---')
    if event.Key == 'Escape':
        hm.UnhookKeyboard()
        hm.UnhookMouse()
        if p is not None:
            p.terminate()
            p.join()
        print('Press Esc, program exit.')
        exit(0)
    # 返回值含义同上
    return True
  
if __name__ == '__main__':

    hm = pyHook.HookManager()
    hm.MouseAllButtonsDown = OnMouseEvent #将OnMouseEvent函数绑定到MouseAllButtonsDown事件上
    hm.KeyDown = OnKeyboardEvent          #将OnKeyboardEvent函数绑定到KeyDown事件上
    hm.HookMouse()        #设置鼠标钩子
    hm.HookKeyboard()   #设置键盘钩子
    print('running')
    pythoncom.PumpMessages()

    
