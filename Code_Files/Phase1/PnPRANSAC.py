#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:23:49 2022

@author: dushyant
"""

import numpy as np
from LinearPnP import LinearPnP

def PnPRANSAC(common_worldpoints_13, common_inliers3_13, K):
    n = 0
    iterations = 2000
    threshold = 10
    length = len(common_worldpoints_13)
    I = np.identity(3)    
    for i in range(iterations):
        while(True):
            random6_world_points = []
            random6_image_points = []
            flag = 0
            index = np.random.randint(length, size=(6))
            for k in range(6):
                random6_world_points.append(list(common_worldpoints_13[index[k]]))
                random6_image_points.append(list(common_inliers3_13[index[k]]))
            for m in range(6):
                count2 = random6_image_points.count(random6_image_points[m])
                if count2 > 1:
                    flag = 1
                    break
            #print('count of occurences',count2)
            if flag == 0:
                break

        PnP_Undecomposed, Linear_PnP_mat = LinearPnP(random6_world_points, random6_image_points, K)
        S = 0        
        for j in range(len(common_worldpoints_13)):
            #X_Proj = np.dot(np.dot(K, Linear_PnP_mat), common_worldpoints_13[j])
            X_Proj = np.dot(np.dot(I, PnP_Undecomposed), common_worldpoints_13[j])
            X_Proj = X_Proj/X_Proj[2]
            error_x = common_inliers3_13[j][0] - X_Proj[0]
            error_y = common_inliers3_13[j][1] - X_Proj[1]
            error = error_x**2 + error_y**2            
            if error < threshold:
                S = S + 1                
        if n < S:
            n = S
            PnP_Final = PnP_Undecomposed.copy()            
    return PnP_Final