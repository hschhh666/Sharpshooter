import mmcv
from mmskeleton.apis import init_pose_estimator, inference_pose_estimator
import cv2
import numpy as np
import time



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


if __name__ == '__main__':
  video = mmcv.VideoReader('E:/MyProject/CSGOAI/skeletonTest.mp4')
  factor = 0.5
  for i, frame in enumerate(video):
    if i < 160:
      continue
    frame = cv2.resize(frame,(int(np.shape(frame)[1]* factor),int(np.shape(frame)[0]* factor)))
    start = time.time()
    result = inference_pose_estimator(model, frame)
    print('hz %.2f'%(1/(time.time()-start)))
    print('Process the frame {}'.format(i))
    if result['joint_preds'] is None:
      continue
    joint_preds = result['joint_preds']
    people_num = np.shape(joint_preds)[0]
    joint_num = np.shape(joint_preds)[1]
    print('joint number =', joint_num)
    for i in range(people_num):
      for j in range(1):
        cv2.circle(frame, (int(joint_preds[i,j,0]),int(joint_preds[i,j,1])), 1, (0,0,255), thickness = 2)
    

    cv2.imshow('img',frame)
    cv2.waitKey(1)


