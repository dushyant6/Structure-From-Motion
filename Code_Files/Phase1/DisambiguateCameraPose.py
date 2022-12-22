#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 13:50:15 2022

@author: dushyant
"""
from LinearTriangulation import LinearTriangulation
import numpy as np
def DisambiguateCameraPose(inlsrc_pts, inldst_pts, P, P_dash, K, R, C):
    I = np.identity(3)
    max_positive_count = 0
    final_m = 5
    for m in range(4):
        count = 0
        World_points = LinearTriangulation(inlsrc_pts, inldst_pts, P, P_dash[m], K)
        Cam_center2 = -np.dot(I,np.dot(np.linalg.inv(R[m]),C[m]))
        Cam_center1 = [0,0,0]
        R1_2 = [0,0,1]
        for i in range(len(World_points)):
            cheirality1 = np.dot(R1_2,(World_points[i][0:3]-Cam_center1))
            cheirality2 = np.dot(np.transpose(R[m][:,2]),(World_points[i][0:3]-Cam_center2))
#original code            #cheirality2 = np.dot(np.transpose(R[m][2]),(World_points[i][0:3]-Cam_center2))
            if cheirality2>0 and cheirality1>0:
                count += 1
        print('number of positive Z values', count)
        if count > max_positive_count:
            max_positive_count = count
            final_m = m
#        X_proj_dst = np.dot(np.dot(K,P_dash[m]), World_points[4])
#        X_proj_src = np.dot(np.dot(K,P), World_points[4])
    
    pose_index = final_m
    World_points_for_correct_pose = LinearTriangulation(inlsrc_pts, inldst_pts, P, P_dash[final_m], K)
    
    return World_points_for_correct_pose, pose_index
