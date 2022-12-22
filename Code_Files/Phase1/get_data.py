#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 15:03:37 2022

@author: dushyant
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 17:40:13 2022

@author: dushyant
"""

import cv2 as cv
import numpy as np
import glob 
from timeit import default_timer as timer

def read_match_file(image_number, file4_lines, count):
    im_1to2 = im_1to3 = im_1to4 = im_1to5 = im_2to3 = im_2to4 = im_2to5 = im_3to4 = im_3to5 = []
    im_4to5 = []    
    temp = []
    for m in range(1, 6-image_number):
        globals()["im_"+str(image_number) +"to"+str(image_number+m)]= []

    for i in range(1,count+1):
        num_of_matches = int(file4_lines[i][0])
        if num_of_matches == 2:
            _, r_value, g_value, b_value, x_cord, y_cord, match_image_index1, match_x_cord2, match_y_cord2 = file4_lines[i].strip().split(' ')
            r_value = int(r_value)
            g_value = int(g_value)
            b_value = int(b_value)
            x_cord = float(x_cord)
            y_cord = float(y_cord)
            match_image_index1 = int(match_image_index1)
            match_x_cord2 = float(match_x_cord2)
            match_y_cord2 = float(match_y_cord2)
            intensity_val = (r_value, g_value, b_value)
            im1_cord = (x_cord, y_cord)
            im2_cord = (match_x_cord2, match_y_cord2)
            globals()["im_"+str(image_number) +"to"+str(match_image_index1)].append([intensity_val,im1_cord, im2_cord])
            if image_number == 4:
                a = globals()["im_"+str(image_number) +"to"+str(match_image_index1)]
                temp = a
            
        
        
        if num_of_matches == 3:
            _, r_value, g_value, b_value, x_cord, y_cord, match_image_index1, match_x_cord2, match_y_cord2, match_image_index2, match_x_cord3, match_y_cord3 = file4_lines[i].strip().split(' ')
            match_image_index2 = int(match_image_index2)
            match_x_cord3 = float(match_x_cord3)
            match_y_cord3 = float(match_y_cord3)
            r_value = int(r_value)
            g_value = int(g_value)
            b_value = int(b_value)
            x_cord = float(x_cord)
            y_cord = float(y_cord)
            match_image_index1 = int(match_image_index1)
            match_x_cord2 = float(match_x_cord2)
            match_y_cord2 = float(match_y_cord2)
            intensity_val = (r_value, g_value, b_value)
            im1_cord = (x_cord, y_cord)
            im2_cord = (match_x_cord2, match_y_cord2)
            im3_cord = (match_x_cord3, match_y_cord3)
            
            globals()["im_"+str(image_number) +"to"+str(match_image_index1)].append([intensity_val,im1_cord, im2_cord])
            globals()["im_"+str(image_number) +"to"+str(match_image_index2)].append([intensity_val,im1_cord, im3_cord])
            
        if num_of_matches == 4:
            _, r_value, g_value, b_value, x_cord, y_cord, match_image_index1, match_x_cord2, match_y_cord2, match_image_index2, match_x_cord3, match_y_cord3, match_image_index3, match_x_cord4, match_y_cord4 = file4_lines[i].strip().split(' ')
            match_image_index3 = int(match_image_index3)
            match_x_cord4 = float(match_x_cord4)
            match_y_cord4 = float(match_y_cord4)

            match_image_index2 = int(match_image_index2)
            match_x_cord3 = float(match_x_cord3)
            match_y_cord3 = float(match_y_cord3)
            r_value = int(r_value)
            g_value = int(g_value)
            b_value = int(b_value)
            x_cord = float(x_cord)
            y_cord = float(y_cord)
            match_image_index1 = int(match_image_index1)
            match_x_cord2 = float(match_x_cord2)
            match_y_cord2 = float(match_y_cord2)
            intensity_val = (r_value, g_value, b_value)
            im1_cord = (x_cord, y_cord)
            im2_cord = (match_x_cord2, match_y_cord2)
            im3_cord = (match_x_cord3, match_y_cord3)
            im4_cord = (match_x_cord4, match_y_cord4)
            globals()["im_"+str(image_number) +"to"+str(match_image_index1)].append([intensity_val,im1_cord, im2_cord])
            globals()["im_"+str(image_number) +"to"+str(match_image_index2)].append([intensity_val,im1_cord, im3_cord])
            globals()["im_"+str(image_number) +"to"+str(match_image_index3)].append([intensity_val,im1_cord, im4_cord])
            
        if num_of_matches == 5:
            _, r_value, g_value, b_value, x_cord, y_cord, match_image_index1, match_x_cord2, match_y_cord2, match_image_index2, match_x_cord3, match_y_cord3, match_image_index3, match_x_cord4, match_y_cord4, match_image_index4, match_x_cord5, match_y_cord5 = file4_lines[i].strip().split(' ')
            match_image_index4 = int(match_image_index4)
            match_x_cord5 = float(match_x_cord5)
            match_y_cord5 = float(match_y_cord5)
            
            match_image_index3 = int(match_image_index3)
            match_x_cord4 = float(match_x_cord4)
            match_y_cord4 = float(match_y_cord4)

            match_image_index2 = int(match_image_index2)
            match_x_cord3 = float(match_x_cord3)
            match_y_cord3 = float(match_y_cord3)

            r_value = int(r_value)
            g_value = int(g_value)
            b_value = int(b_value)
            x_cord = float(x_cord)
            y_cord = float(y_cord)
            match_image_index1 = int(match_image_index1)
            match_x_cord2 = float(match_x_cord2)
            match_y_cord2 = float(match_y_cord2)
            intensity_val = (r_value, g_value, b_value)
            im1_cord = (x_cord, y_cord)
            im2_cord = (match_x_cord2, match_y_cord2)
            im3_cord = (match_x_cord3, match_y_cord3)
            im4_cord = (match_x_cord4, match_y_cord4)
            im5_cord = (match_x_cord5, match_y_cord5)
            globals()["im_"+str(image_number) +"to"+str(match_image_index1)].append([intensity_val,im1_cord, im2_cord])
            globals()["im_"+str(image_number) +"to"+str(match_image_index2)].append([intensity_val,im1_cord, im3_cord])
            globals()["im_"+str(image_number) +"to"+str(match_image_index3)].append([intensity_val,im1_cord, im4_cord])
            globals()["im_"+str(image_number) +"to"+str(match_image_index4)].append([intensity_val,im1_cord, im5_cord])
    #print('test',im_1to5)
    im_4to5 = temp
    
    return im_1to2, im_1to3, im_1to4, im_1to5, im_2to3, im_2to4, im_2to5, im_3to4, im_3to5, im_4to5

start1 = timer()
#image number 1
file = open(r'P3Data/matching1.txt', 'r')
file_lines = []
for count,line in enumerate(file):
    file_lines.append(line)
img_1to2, img_1to3, img_1to4, img_1to5, _, _, _, _, _, _  = read_match_file(1, file_lines, count)
#image number 2
file = open(r'P3Data/matching2.txt', 'r')
file_lines = []
for count,line in enumerate(file):
    file_lines.append(line)
_, _, _, _, img_2to3, img_2to4, img_2to5, _, _, _  = read_match_file(2, file_lines, count)
#image number 3
file = open(r'P3Data/matching3.txt', 'r')
file_lines = []
for count,line in enumerate(file):
    file_lines.append(line)
_, _, _, _, _, _, _, img_3to4, img_3to5, _  = read_match_file(3, file_lines, count)
#image number 4
file = open(r'P3Data/matching4.txt', 'r')
file_lines = []
for count,line in enumerate(file):
    file_lines.append(line)
_, _, _, _, _, _, _, _, _, img_4to5  = read_match_file(4, file_lines, count)

img_1to2 = im_1to2
x1 = img_1to2[0][1][0]
x1dash = img_1to2[0][2][0]
y1 = img_1to2[0][1][1]
y1dash = img_1to2[0][2][1]


A = np.zeros([9,9])
for i in range(9):
    x1 = img_1to2[i][1][0]
    x1dash = img_1to2[i][2][0]
    y1 = img_1to2[i][1][1]
    y1dash = img_1to2[i][2][1]
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
print('rank of F matrix', np.linalg.matrix_rank(F))


print('total time', timer()- start1)
