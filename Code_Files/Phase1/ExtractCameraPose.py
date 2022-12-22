#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 13:26:07 2022

@author: dushyant
"""
import numpy as np

def ExtractCameraPose(E, K):
    I = np.identity((3))
    C = []
    t = []
    R = []
    P = np.zeros([4,3,4])
    U_e, D_e, V_e = np.linalg.svd(E)
    W = np.zeros([3,3])
    W[0][1] = -1
    W[1][0] = 1
    W[2][2] = 1
    #print('W matrix', W)
    W_trans = np.transpose(W)

    C1 = U_e[:,2]
    R1 = np.dot(np.dot(U_e, W), V_e)
    if(np.linalg.det(R1) < 0):
        C1 = -1*C1
        R1 = -1*R1
    C.append(C1)
    R.append(R1)
    P[0][:,0:3]= R1
    P[0][:,3]= C1#-1*C1)
    
    C2 = -1*U_e[:,2]
    R2 = np.dot(np.dot(U_e, W), V_e)
    if(np.linalg.det(R2) < 0):
        C2 = -1*C2
        R2 = -1*R2
    C.append(C2)
    R.append(R2)
    P[1][:,0:3]= R2
    P[1][:,3]= C2#-1*C1)

    C3 = U_e[:,2]
    R3 = np.dot(np.dot(U_e, W_trans), V_e)
    if(np.linalg.det(R3) < 0):
        C3 = -1*C3
        R3 = -1*R3
    C.append(C3)
    R.append(R3)
    P[2][:,0:3]= R3
    P[2][:,3]= C3#-1*C1)

    C4 = -1*U_e[:,2]
    R4 = np.dot(np.dot(U_e, W_trans), V_e)
    if(np.linalg.det(R4) < 0):
        C4 = -1*C4
        R4 = -1*R4
    C.append(C4)
    R.append(R4)
    P[3][:,0:3]= R4
    P[3][:,3]= C4#-1*C1)

    #print('det of R1, R2, R3, R4', np.linalg.det(R[0]), np.linalg.det(R[1]), np.linalg.det(R[2]), np.linalg.det(R[3]))
    return(C, R, P)
