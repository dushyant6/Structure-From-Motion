#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 22:00:55 2022

@author: dushyant
"""

import numpy as np

def create_rows(world_point, image_point):
    rows_per_point = []
    X = world_point[0]
    Y = world_point[1]
    Z = world_point[2]
    x = image_point[0]
    y = image_point[1]
    row1 = (X, Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x)
    row2 = (0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y)
    rows_per_point.append(row1)
    rows_per_point.append(row2)
    return rows_per_point


def LinearPnP(common_6world_points, common_6image_points, K):
    A_PnP = []
    for i in range(len(common_6world_points)):
        rows = create_rows(common_6world_points[i], common_6image_points[i])
        A_PnP.append(rows)
    A_PnP = np.asarray(A_PnP)    

    A_PnP = np.reshape(A_PnP,(A_PnP.shape[0]*A_PnP.shape[1],12))
    U_PnP, S_PnP, V_PnP = np.linalg.svd(A_PnP)
    PnP_elements = V_PnP[-1]
    PnP_Mat = [[PnP_elements[0], PnP_elements[1], PnP_elements[2], PnP_elements[3]],
                  [PnP_elements[4], PnP_elements[5], PnP_elements[6], PnP_elements[7]],
                  [PnP_elements[8], PnP_elements[9], PnP_elements[10], PnP_elements[11]]]
    PnP_Mat = np.asarray(PnP_Mat)

    rot_init = PnP_Mat[:,0:3]
    gamma_R = np.dot(np.linalg.inv(K), rot_init)

    U,D,V = np.linalg.svd(gamma_R)
    R_PnP = np.dot(U,V)
    gamma = D[0]
    #print('gamma', gamma)
    T_PnP = np.matmul(np.linalg.inv(K),PnP_Mat[:,3])/gamma
    
    PnP_Matrix = np.zeros_like(PnP_Mat)
    PnP_Matrix[:,0:3] = R_PnP
    PnP_Matrix[:,3] = T_PnP
    #print('PnP Matrix multiplied by K\n',np.dot(K,PnP_Matrix))
    PnP_undecomposed = PnP_Mat
    return PnP_undecomposed, PnP_Matrix