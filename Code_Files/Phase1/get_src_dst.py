#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 13:03:30 2022

@author: dushyant
"""

import numpy as np

def create_src_dst(img_src_to_dst):
    src_pts = []#img_1to2[1][1]
    dst_pts = []
    for i in range(len(img_src_to_dst)):
        a = img_src_to_dst[i][1]
        b = img_src_to_dst[i][2]
        src_pts.append(a)
        dst_pts.append(b)
    src_pts = np.asarray(src_pts)
    dst_pts = np.asarray(dst_pts)
    return src_pts, dst_pts
