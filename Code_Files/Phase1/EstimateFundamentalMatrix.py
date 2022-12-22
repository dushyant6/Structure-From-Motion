#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 13:17:13 2022

@author: dushyant
"""
import numpy as np
import random
from random import randrange


def EstimateFundamentalMatrix(inlsrc_pts, inldst_pts):
    A = np.zeros([len(inlsrc_pts),9])
    for i in range(len(inlsrc_pts)):
        x1 = inlsrc_pts[i][0]
        x1dash = inldst_pts[i][0]
        y1 = inlsrc_pts[i][1]
        y1dash = inldst_pts[i][1]

        A[i][0] = x1*x1dash
        A[i][1] = x1*y1dash
        A[i][2] = x1
        A[i][3] = y1*x1dash
        A[i][4] = y1*y1dash
        A[i][5] = y1
        A[i][6] = x1dash
        A[i][7] = y1dash
        A[i][8] = 1

    U, S, V = np.linalg.svd(A)
    f = (V[-1,:])
    F = np.zeros([9,1])
    F = f
    F = np.reshape(F, [3,3])
    F = np.transpose(F)

    U1, S1, V1 = np.linalg.svd(F)
    S1[2] = 0
    S2 = np.diag(S1)
    F_new = np.dot(np.dot(U1, S2), V1)

    return F_new

def getRandomPoints(src_pts, dst_pts):
    indices = []
    while(len(indices)<9):
        index = randrange(len(src_pts))
        if index in indices:
            continue
        else:
            indices.append(index)
    
    randpt_src = []
    randpt_dst = []
    
    for i in range (8):
        randpt_src.append(src_pts[indices[i]])
        randpt_dst.append(dst_pts[indices[i]])

    return randpt_src, randpt_dst #sourceX,sourceY,destX,destY


def getError(src_pt, dst_pt, fundamentalMatrix):
    sourceX = src_pt[0]
    sourceY = src_pt[1]
    destX = dst_pt[0]
    destY = dst_pt[1]
    x1 = [sourceX,sourceY,1]
    x2 = [destX,destY,1]
    error = abs(np.dot(np.dot(x2,fundamentalMatrix),x1))
    return error


def getInliners(src_pts, dst_pts):
    n = 0
    iterations = 4000
    error = 0.03
    fFinal = []
    sFinal = []
    
    for i in range (iterations):
        randpt_src, randpt_dst = getRandomPoints(src_pts, dst_pts)
        fCalculated = EstimateFundamentalMatrix(randpt_src, randpt_dst)
        S = []
        src_inliers = []
        dst_inliers = []
        for j in range (len(src_pts)):
            errorCalculated = getError(src_pts[j], dst_pts[j], fCalculated)
            if(errorCalculated < error):
                S.append(j)
                src_inliers.append(src_pts[j])
                dst_inliers.append(dst_pts[j])
        if(n < len(S)):
            n = len(S)
            sFinal = S
            fFinal = fCalculated
            f_src_inliers = src_inliers
            f_dst_inliers = dst_inliers
    return sFinal,fFinal, f_src_inliers, f_dst_inliers
