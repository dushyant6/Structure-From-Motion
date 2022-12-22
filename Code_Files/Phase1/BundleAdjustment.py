#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 16:02:24 2022

@author: dushyant
"""
import numpy as np

def Bundle_Error(initial_list, All_Image_Points, Vis_Mat, P_size, W_size, K):
    PSet = np.reshape(initial_list[:P_size[0]*3*4], P_size)
    WorldSet = np.reshape(initial_list[P_size[0]*3*4: len(initial_list)], W_size)
    total_error = 0
    
    for i in range(WorldSet.shape[0]):
        #camera1 reprojection
        proj1 = np.dot(np.dot(K,PSet[0]), WorldSet[i])
        proj1 = proj1/proj1[2]
        err1_x =  proj1[0] - All_Image_Points[i][0][0]
        err1_y =  proj1[1] - All_Image_Points[i][0][1]
        err1 = (err1_x**2 + err1_y**2)
        err1 = Vis_Mat[0][i]*err1
        
        proj2 = np.dot(np.dot(K,PSet[1]), WorldSet[i])
        proj2 = proj2/proj2[2]
        err2_x =  proj2[0] - All_Image_Points[i][1][0]
        err2_y =  proj2[1] - All_Image_Points[i][1][1]
        err2 = (err2_x**2 + err2_y**2)
        err2 = Vis_Mat[1][i]*err2
        
        proj3 = np.dot(np.dot(K,PSet[2]), WorldSet[i])
        proj3 = proj3/proj3[2]
        err3_x =  proj3[0] - All_Image_Points[i][2][0]
        err3_y =  proj3[1] - All_Image_Points[i][2][1]
        err3 = (err3_x**2 + err3_y**2)
        err3 = Vis_Mat[2][i]*err3
        total_error = total_error + err1 + err2 + err3
    return total_error
