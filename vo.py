# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 12:47:10 2020

@author: BhaveshAchhada
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
os.chdir('C:\\Users\\bhave\\Desktop\\MyVO')


'''##############################
#                               #
#   CONSTANTS DECLARATION START #
#                               #
##############################'''

### Paths to dataset images or video
dataset_path = os.path.join('KITTI_sample','images')

### Either list of images or a video
all_images = [ i for i in os.listdir(dataset_path) if i.endswith('.png')]

### Global Translation List
T = list()

### Global Rotation List
R = list()

### Reference Points to compare with
ref_pts = None

### Store previous image to compare with new image
last_image = None

fMATCHING_DIFF = 10  # Minimum difference in the KLT point correspondence

# Parameters used for cv2.calcOpticalFlowPyrLK (KLT tracker)
lk_params = dict(winSize=(21, 21),  
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

### Projection Matrix
P = np.array([ 
     [ 718.56, 0.0,     607.192, 0.0 ],
     [ 0.0,    718.85,  185.215, 0.0 ],
     [ 0.0,    0.0,     1.0,     0.0 ],     
    ])
decomposedP = np.array(cv2.decomposeProjectionMatrix(P))

### Camera Intrinsic Matrix
K = decomposedP[0].reshape((3, 3))

### Origin for reference
p0 = np.eye(4)[:3]
p0 = K.dot(p0)

### 3D points in N-1th Frame
last_points = None

### Global translation and rotation
rotation = None
translation = None

# global T, R, ref_pts, last_image, fMATCHIND_DIFF, lk_params, P, K, p0

'''##############################
#                               #
#   CONSTANTS DECLARATION END   #
#                               #
##############################'''

def getRelativeScale(prev_points, cur_points):
    
    ratios = []
    idx = min(prev_points.shape[0], cur_points.shape[0])
    for i in range(idx):
        xk = cur_points[i]
        xk_1 = cur_points[i - 1]
        p_xk = prev_points[i]
        p_xk_1 = prev_points[i - 1]
        
        if np.linalg.norm(p_xk - xk) != 0:
            ratios.append( np.linalg.norm(p_xk_1 - xk_1) / np.linalg.norm(p_xk - xk))
    
    return np.mean(ratios)
        

def featureTracking(prev_image=None, image=None, ref_pts=None):
    global fMATCHING_DIFF, lk_params
    
    kp2, st, err = cv2.calcOpticalFlowPyrLK(prev_image, image, ref_pts, None, **lk_params)
    kp1, st, err = cv2.calcOpticalFlowPyrLK(image, prev_image, kp2, None, **lk_params)
    
    d = abs(kp1 - ref_pts).reshape(-1,2).max(-1)
    good_pts = d < fMATCHING_DIFF
    kp1,kp2 = kp1[good_pts == True], kp2[good_pts == True]
    
    difference = np.mean(abs(kp1 - kp2).reshape(-1,2).max(-1))
    
    return kp1, kp2, difference

def update(idx, image):
    
    global T, R, ref_pts, last_image, K, p0, last_points, rotation, translation
    
    orb = cv2.ORB_create()
    ### Feature detection using ORB
    kp, des = orb.detectAndCompute(image, None)
    features = np.array([i.pt for i in kp], dtype=np.float32)
    
    if idx == 0:
        ref_pts = features
        T.append(([0],[0],[0]))
        R.append(tuple(np.zeros((3,3))))
        return None        
    else:
        ref_pts, features, _diff = featureTracking(last_image, image, ref_pts)
        E, mask = cv2.findEssentialMat(features, ref_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        # Estimate Rotation and translation vectors
        _, cur_R, cur_t, mask = cv2.recoverPose(E, features, ref_pts, K)
                              
        p1 = np.hstack((cur_R, cur_t))
        p1 = K.dot(p1)
        
        pts1, pts2 = ref_pts.reshape(2,-1), features.reshape(2,-1)
        
        ### Return homogeneous 4D points, output shape = (4,*)
        points = cv2.triangulatePoints(p0, p1, pts1, pts2)
        
        ### Reshape output to (*,4)
        points = points.reshape((-1,4))

        ### Convert 4D homogeneous coordinates to 3D        
        homo_factor = points[:,3]  
        homo_factor = homo_factor.reshape((-1,1))
        
        points = points/homo_factor
        points = points[:,:3]
        
        scale = getRelativeScale(ref_pts, features)
        
        # cur_t += scale*cur_R.dot(t)
        if translation is None:
            translation = cur_t
        else:
            translation += scale*rotation.dot(cur_t)
            
        if rotation is None:
            rotation = cur_R
        else:
            rotation = cur_R.dot(rotation)
        
        T.append(tuple(translation))
        R.append(tuple(rotation))       

        ref_pts = features        
        last_points = points
        return points


for idx, image_name in enumerate(all_images):
    
    image_path = os.path.join(dataset_path, image_name)
    image = cv2.imread(image_path)
    gray = cv2.imread(image_path, 0)
    
    #$$ TODO: Increase contrast slightly
    clahe = cv2.createCLAHE(clipLimit=5.0)
    gray = clahe.apply(gray)
    
    try:
        if idx == 0:
            _ = update(idx, gray)
            pass
        else:
            points = update(idx, gray)
            pass
        last_image = gray        
    except:
        pass
    finally:
        # _ = os.system('cls')
        # print(chr(27) + "[2J")
        print(idx+1, '/',len(all_images))