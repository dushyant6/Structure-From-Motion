#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 14:26:26 2022

@author: dushyant
"""

import numpy as np
import scipy
from scipy import optimize
from scipy.optimize import least_squares

def reprojection_error(World_point, P, P_dash, inlsrc_pt, inldst_pt, K):
    #total_error = 0
    #print('world point', World_point)
    #print('np check', np.dot(K,P))
    X_proj_src = np.dot(np.dot(K,P), World_point)
    X_proj_src = X_proj_src/X_proj_src[2]
    err_src_x = -inlsrc_pt[0] + X_proj_src[0]
    err_src_y = -inlsrc_pt[1] + X_proj_src[1]
    err_src = np.sqrt(err_src_x**2 + err_src_y**2)
    
    X_proj_dst = np.dot(np.dot(K,P_dash), World_point)
    X_proj_dst = X_proj_src/X_proj_src[2]
    err_dst_x = -inldst_pt[0] + X_proj_dst[0]
    err_dst_y = -inldst_pt[1] + X_proj_dst[1]
    err_dst = np.sqrt(err_dst_x**2 + err_dst_y**2)
    error = err_dst + err_src

    return error

def NonlinearTriangulation(P, P_dash, inlsrc_pts, inldst_pts, World_points_test, K):
    optimized_world_points = []
    for i in range(len(World_points_test)):
        optimized_param_list = scipy.optimize.least_squares(reprojection_error, World_points_test[i], ftol = 0.0001, xtol = 0.0001, args = [P, P_dash, inlsrc_pts[i], inldst_pts[i], K])
        optimized_world_point = optimized_param_list.x
        optimized_world_point = optimized_world_point/ optimized_world_point[3]
        optimized_world_points.append(optimized_world_point)
        
        #reprojection_error(P, P_dash, inlsrc_pts[i], inldst_pts[i], World_points[i], K)
    
    return optimized_world_points

