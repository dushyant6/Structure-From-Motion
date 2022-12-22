#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 21:51:57 2022

@author: dushyant
"""

import numpy as np
def BuildVisibilityMatrix(World_points, common_worldpoints_13, list_len1, list_len2):    
    Vis_Mat = np.zeros([3,len(World_points)])
    Vis_Mat[0] = 1
    Vis_Mat[1][0:list_len1] = 1
    Vis_Mat[2][list_len1:list_len1 + list_len2] = 1
    
    for i in range(len(World_points)):
        for j in range(len(common_worldpoints_13)):
            if (World_points[i][0] == common_worldpoints_13[j][0] and World_points[i][1] == common_worldpoints_13[j][1]):
                Vis_Mat[2][i]= 1
                
    return Vis_Mat

def AdaptiveVisibility(n, Vis_available, World_points, list_len_new, common_worldpoints):
    list_len1 = Vis_available.shape[1]
    print('list len 1', list_len1)
    new_column = np.zeros([len(World_points),1])
    new_column[list_len1: list_len1+list_len_new] = 1

    for i in range(len(World_points)):
        for j in range(len(common_worldpoints)):
            if (World_points[i][0] == common_worldpoints[j][0] and World_points[i][1] == common_worldpoints[j][1]):
                new_column[i]= 1

    new_visibility_mat = np.zeros([n,len(World_points)])
    new_visibility_mat[n-1] = np.ravel(new_column)
    old_rows, old_columns = Vis_available.shape
    new_visibility_mat[0:old_rows, 0:old_columns] = Vis_available
    new_visibility_mat[0] = 1

    return new_visibility_mat