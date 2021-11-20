import PyHook3 as pyHook
import win32api
import win32con
import pyautogui
import pythoncom
import os

import time
import cv2
import numpy as np
import multiprocessing
import torch

from grabScreen import grabScreen, grabOneFrame, grabOneFrame_faster, grabTwoWindows
from objectSelection import objectSelection_from_yolo

# def friend_enemy_judge(img,x1,x2,origin_y):
#     # return 0
#     search_area = [5, 10, 15, 20]
#     for y in search_area:
#         y = origin_y - y
#         if y < 0:
#             return True
#         for x in range(x1,x2):
#             if img[y,x,0] > 150:
#                 return True # 是朋友
#     return False

def friend_judge(img, x, allFriend = False): # True 朋友 False 敌人
    if allFriend:
        return True
    k = 3840/360 # 转动角度与鼠标移动事件的换算系数，在游戏中采集数据计算出来的
    f = 958.2568832151012 # 焦距，在游戏中采集数据计算出来的，单位：像素
    cv2.circle(img,(170, 170), 2, (0,0,255), thickness = -1)
    lenth = 160
    theta = np.arctan(x/f)
    x = int(lenth * np.sin(theta))
    y = int(lenth * np.cos(theta))
    # cv2.line(img,(170,170), (170+x, 170-y),(255,255,255),1)
    for i in range(20,lenth):
        x = 170+int(i * np.sin(theta))
        y = 170-int(i * np.cos(theta))
        tmpx = 170+int((i-5) * np.sin(theta))
        tmpy = 170-int((i-5) * np.cos(theta))
        cv2.circle(img,(tmpx,tmpy),1,(255,255,255),thickness=-1)
        if img[y,x,0] > 180 and img[y,x,1]>180 and img[y,x,2] < 120:
            return True
        # if img[y,x,2] > 180 and img[y,x,0] < 50 and img[y,x,1] < 50:
        #     return False
    return False

def detection_and_operating(auto_attack_flag): # 默认不自动攻击
    yolo_model = torch.hub.load('C:\\Users\\A/.cache\\torch\\hub\\ultralytics_yolov5_master', 'yolov5s', source='local')  # or yolov5m, yolov5l, yolov5x, custom
    start_time = time.time()
    grab_time = 0
    detection_time = 0
    vis_time = 0
    click_time = 0
    total_time = 0
    while 1:
        begin = time.time()

        if time.time() - start_time > 1800: # 超时自动退出
            exit()
        print('---------------------')
        print('auto_attack_flag: ',auto_attack_flag.value)
        
        # img, hz = grabOneFrame_faster()
        # little_map, hz = grabOneFrame_faster({'left': 28, 'top': 110, 'width':370-28, 'height': 451-110})

        little_map, img = grabTwoWindows()
        
        grab_time = time.time() - begin
        print('Grab img time: %.2fms, fre: %.2fhz'%(1000*(grab_time),1/(grab_time)))
        detection_start = time.time()
        results = yolo_model(img)        
        pred = results.pred[0].cpu().numpy()
        detection_time = time.time() - detection_start
        print('Detection time: %.2fms, fre: %.2fhz'%(1000*(detection_time),1/(detection_time)))

        vis_start = time.time()
        attacking_target = False
        distance_to_center = 1000 # 当前目标距离准星的距离
        for one_obj in pred:
            x1, y1, x2, y2, confidence, obj_class = int(one_obj[0]), int(one_obj[1]), int(one_obj[2]), int(one_obj[3]), float(one_obj[4]), int(one_obj[5])
            if obj_class == 0 and confidence > 0.6 and y2-y1 > x2-x1 and x2-x1>30 and y2-y1>30: # 对预测结果初筛，排除误检、死尸
                if not friend_judge(little_map,((x1+x2)/2)-np.shape(img)[1]/2): # 敌人
                    attacking_target = True
                    img = cv2.rectangle(img, (x1,y1),(x2,y2),(0,0,255),2) # 敌人用红框标识
                    if abs((x1+x2)/2-np.shape(img)[1]/2) < distance_to_center: # 距离准星最近的敌人作为当前攻击目标，这样在面对多个敌人时可以有效防止不停切换攻击目标
                        distance_to_center =  abs((x1+x2)/2-np.shape(img)[1]/2) 
                        targetX = int((x1+x2)/2) # 攻击目标的坐标
                        targetY = int((y1+y2)/2 - abs(y2-y1)/4)# 攻击目标的坐标，大概在前胸的位置。因为脑袋不一定在bbox正中间，如果瞄得太靠上的话可能会打空
                else: # 友军
                    img = cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2) # 友军用绿框标识
                cv2.putText(img, '%.2f'%confidence, (x1,y1-10),1,1.5,(0,0,255),2) # 候选攻击目标的置信度
        if attacking_target:
            cv2.circle(img, (targetX, targetY), 8, (0,0,255), thickness = -1)
        factor = 1
        cv2.namedWindow('img', cv2.WINDOW_NORMAL) # 可视化检测结果与攻击目标
        cv2.resizeWindow("img",(int(np.shape(img)[1]* factor),int(np.shape(img)[0]* factor))) #设置窗口大小
        cv2.moveWindow("img", 1900, 0)
        img = cv2.resize(img,(int(np.shape(img)[1]* factor),int(np.shape(img)[0]* factor)))

        textOnImg = 'grab: %.0fms detection: %.0fms vis: %.0fms click: %.0fms total: %.0fms AI: %d'%(1000*grab_time,1000*detection_time,1000*vis_time,1000*click_time,1000*total_time,auto_attack_flag.value)
        cv2.putText(img,textOnImg,(20,20),1,1,(0,0,255,2))
        cv2.imshow('img',img)
        cv2.imshow('little',little_map)
        cv2.moveWindow("little", 2200, 400)
        cv2.waitKey(1)
        key = cv2.waitKey(1)
        vis_time = time.time() - vis_start
        print('Visualization time: %.2fms, fre: %.2fhz'%(1000*(vis_time), 1/(vis_time)))
        
        click_start = time.time()
        if attacking_target and auto_attack_flag.value:  # 鼠标操作
            currentMouseX, currentMouseY = pyautogui.position()
            targetX += 3 * 2560//8
            targetY += 3 * 1440//8
            deltaX, deltaY = int(targetX - currentMouseX), int(targetY - currentMouseY)          
            k = 3840/360 # 转动角度与鼠标移动事件的换算系数，在游戏中采集数据计算出来的
            f = 958.2568832151012 # 焦距，在游戏中采集数据计算出来的，单位：像素
            deltaX = int((np.arctan(deltaX/f)*180/np.pi) * k)
            deltaY = int((np.arctan(deltaY/f)*180/np.pi) * k)
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, deltaX, deltaY)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            # time.sleep(0.3)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0) # 这样点击更快
            time.sleep(0.05)
            
            click_time = time.time() - click_start
            print('Click time: %.2fms, fre: %.2fhz'%(1000*(click_time), 1/(click_time+0.001)))

        total_time = time.time() - begin
        print('Total time: %.2fms, fre: %.2fhz'%(1000*(total_time), 1/total_time))

# 鼠标事件处理函数
def OnMouseEvent(event):
    if event.MessageName == 'mouse middle down': #当点击鼠标中键时，启动/终止子进程
        global auto_attack_flag
        auto_attack_flag.value = 0 if auto_attack_flag.value == 1 else 1
    return True # 返回True代表将事件继续传给其他句柄，为False则停止传递，即被拦截。换句话说，当被拦截时，其他程序就不知道鼠标被点了。

#键盘事件处理函数
def OnKeyboardEvent(event):
    if event.Key == 'Q':
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
    auto_attack_flag = multiprocessing.Value('b', 0) # 是否自动攻击的标志位
    p = None
    p = multiprocessing.Process(target=detection_and_operating, args=(auto_attack_flag,))
    p.start()
    hm = pyHook.HookManager()
    hm.MouseAllButtonsDown = OnMouseEvent #将OnMouseEvent函数绑定到MouseAllButtonsDown事件上
    hm.KeyDown = OnKeyboardEvent          #将OnKeyboardEvent函数绑定到KeyDown事件上
    hm.HookMouse()        #设置鼠标钩子
    hm.HookKeyboard()   #设置键盘钩子
    print('running')
    pythoncom.PumpMessages()