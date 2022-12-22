#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 12:56:59 2022

@author: dushyant
"""
import numpy as np
from random import randint
import cv2 as cv
import math
import copy

def RANSAC(src_pts,dst_pts,threshold,N_max, inlper):
  length = len(src_pts)
  inlsrc_m = []
  inldst_m = []
  N = 1
  inlper_c = 0
  
  while((N < N_max) and (inlper_c < inlper)): 
    index = np.array([-1,-1,-1,-1])
    inlsrc_c = []
    inldst_c = []
    for i in range(4):
      a = randint(0,length-1)
      while(True):
        if (a not in index):
          index[i] = a
          break
        else:
          a = randint(0, length-1) 
    #print(index)
    randpt_src = [src_pts[i] for i in index]  
    randpt_dst = [dst_pts[i] for i in index]
    randpt_src = np.array(randpt_src)
    randpt_dst = np.array(randpt_dst)
    H,_ = cv.findHomography(randpt_src,randpt_dst) 
    if(H[2][2]!= 0):
      for i in range(0,length): 
       src = list(src_pts[i].astype('float'))
       dst = list(dst_pts[i].astype('float'))
       src.append(1)
       dst.append(1)
       dst_est = np.matmul(H,src)
       #print(dst_est)
       dst_est[0] = dst_est[0] / dst_est[2];
       dst_est[1] = dst_est[1] / dst_est[2];
       if (math.isnan(dst_est[0]) or math.isnan(dst_est[0])):
        continue
       else:
         dst_est[0] = round(dst_est[0])
         dst_est[1] = round(dst_est[1]) 
         error = math.sqrt((dst[0]-dst_est[0])**2 + (dst[1] - dst_est[1])**2)
         if error < threshold:
          inlsrc_c.append(src_pts[i])
          inldst_c.append(dst_pts[i])
      
      if len(inlsrc_c) > len(inlsrc_m):
        inlsrc_m = copy.deepcopy(inlsrc_c)
        inldst_m = copy.deepcopy(inldst_c)
      inlper_c = (float(len(inlsrc_m)) / float(len(src_pts))) * 100.0   
      N = N + 1
  return inlsrc_m,inldst_m