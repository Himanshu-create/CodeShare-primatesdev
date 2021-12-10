# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 19:11:32 2021

@author: Himanshu Singh
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


#img from camera frame by frame  


def viewScreen(speaking = 0, imgs = []):
    h = 720 #height of frame
    w = 1080 #width of frame
    
    frame = np.zeros([h,w,3], dtype=np.uint8)
    frame.fill(100)
    
    r = w - w//20
    l = w//20
    t = h//20
    b = h - h//20
    gw = w//20
    gh = h//20
    
    if(speaking == 1):
        img1 = imgs[0]
        img1 = cv2.resize(img1,(w - 2*l , h - 2*t))
        frame[t:b,l:r] = img1

    elif(speaking == 2):
        img1 = imgs[0]
        img2 = imgs[1]
        img1 = cv2.resize(img1,(w//2 - l - gw//2, h//2))
        img2 = cv2.resize(img1,(w//2 - l - gw//2, h//2))
        
        frame[h//4:h//4 + img1.shape[0],l:l + img1.shape[1]] = img1
        frame[h//4:h//4 + img1.shape[0],l + img1.shape[1] + gw: l + 2*img1.shape[1] + gw] = img2
        
    elif(speaking == 3):
        img1 = imgs[0]
        img2 = imgs[1]
        img3 = imgs[2]
        img1 = cv2.resize(img1,(w//2 - l - gw//2, h//2 - t - gh//2))
        img2 = cv2.resize(img2,(w//2 - l - gw//2, h//2 - t - gh//2))
        img3 = cv2.resize(img3,(w//2 - l - gw//2, h//2 - t - gh//2))
        
        frame[t:t + img1.shape[0],l: l + img1.shape[1]] = img1
        frame[t:t+img1.shape[0],l+img1.shape[1] + gw:l + 2*img1.shape[1] + gw] = img2
        frame[t+gh+ img1.shape[0]:t + gh + 2*img1.shape[0], w//4+l:w//4+l+ img1.shape[1]] = img3
        
    elif(speaking == 4):
        img1 = imgs[0]
        img2 = imgs[1]
        img3 = imgs[2]
        img4 = imgs[3]
        img1 = cv2.resize(img1,(w//2 - l - gw//2, h//2 - t - gh//2))
        img2 = cv2.resize(img2,(w//2 - l - gw//2, h//2 - t - gh//2))
        img3 = cv2.resize(img3,(w//2 - l - gw//2, h//2 - t - gh//2))
        img4 = cv2.resize(img4,(w//2 - l - gw//2, h//2 - t - gh//2))
        
        frame[t:t + img1.shape[0],l: l + img1.shape[1]] = img1
        frame[t:t+img1.shape[0],l+img1.shape[1] + gw:l + 2*img1.shape[1] + gw] = img2
        frame[t+gh+img1.shape[0]:t+gh+2*img1.shape[0], l:l+img1.shape[1]] = img3
        frame[t+gh+img1.shape[0]:t+gh+2*img1.shape[0], l+gw+img1.shape[1]:l+gw+2*img1.shape[1]] = img4
        
        
    elif(speaking == 5):
        img1 = imgs[0]
        img2 = imgs[1]
        img3 = imgs[2]
        img4 = imgs[3]
        img5 = imgs[4]
        
        img1 = cv2.resize(img1,((w -2*(l +gw))//3,h//3-t-gh//2))
        img2 = cv2.resize(img2,((w -2*(l +gw))//3,h//3-t-gh//2))
        img3 = cv2.resize(img3,((w -2*(l +gw))//3,h//3-t-gh//2))
        img4 = cv2.resize(img4,((w -2*(l +gw))//3,h//3-t-gh//2))
        img5 = cv2.resize(img5,((w -2*(l +gw))//3,h//3-t-gh//2))
        
        
        dh = (h - 2*(h//3-t-gh//2))//3
        
        frame[dh:dh+img1.shape[0],l:l+img1.shape[1]] = img1
        frame[dh:dh+img1.shape[0], l+img1.shape[1]+gw:l+2*img1.shape[1]+gw] = img2
        frame[dh:dh+img1.shape[0],l+2*img1.shape[1]+2*gw:l+3*img1.shape[1]+2*gw] = img3
        
        hx = 2*dh + (h//3-t-gh//2)
        wx = (w - 2*(img4.shape[1]) - gw)//2
        frame[hx:hx+img4.shape[0],wx:wx+img4.shape[1]] = img4
        frame[hx:hx+img4.shape[0],wx+img4.shape[1] + gw :wx+2*img4.shape[1] + gw ] = img5
        
    else:
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (255, 0, 0)
        org = (h//2,w//2)
        thickness = 2
        frame = cv2.putText(frame, 'No One is Speaking', org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    
    return frame


p1 = r"C:\Users\Himanshu Singh\Desktop\img\p.jpg"
img1 = cv2.imread(p1)
#img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) 
size = (1080, 720)
result = cv2.VideoWriter(r"C:\Users\Himanshu Singh\Desktop\img\Result_Video_Cam.avi", 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         25, size)

video = cv2.VideoCapture(2)
if (video.isOpened() == False): 
    print("Error reading video file")
    assert False

tp= time.time()
print(tp)

k = [1,2,3,4,5,3,0,2,1,4]
i = 0
while(True):
    if(i>= len(k)):
        break
    ret, img1 = video.read()
    
    frame = viewScreen(k[i],[img1,img1,img1,img1,img1])
    result.write(frame)
    if(time.time() - tp >= 1):
        i += 1
        tp = time.time()
        print(i)
result.release()
video.release()




#frame = viewScreen(5,[img1,img1,img1,img1,img1])
#plt.imshow(frame)
#plt.show()
#
#frame = viewScreen(4,[img1,img1,img1,img1,img1])
#plt.imshow(frame)
#plt.show()
#
#frame = viewScreen(3,[img1,img1,img1,img1,img1])
#plt.imshow(frame)
#plt.show()
#
#frame = viewScreen(2,[img1,img1,img1,img1,img1])
#plt.imshow(frame)
#plt.show()
#
#frame = viewScreen(1,[img1,img1,img1,img1,img1])
#plt.imshow(frame)
#plt.show()