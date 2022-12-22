#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 12:51:44 2022

@author: dushyant
"""
import numpy as np
import copy
def read_match_file(image_number, file4_lines, count):
    im_1to2 = []
    im_1to3 = []
    im_1to4 = []
    im_1to5 = []
    im_2to3 = []
    im_2to4 = []
    im_2to5 = []
    im_3to4 = []
    im_3to5 = []
    im_4to5 = []
    #im_to_dest = []
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
            if (image_number == 4 and match_image_index1 == 5):
                im_4to5.append([intensity_val,im1_cord, im2_cord])
            
            if (image_number == 3 and match_image_index1 == 4):
                im_3to4.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 3 and match_image_index1 == 5):
                im_3to5.append([intensity_val,im1_cord, im2_cord])
            
            if (image_number == 2 and match_image_index1 == 3):
                im_2to3.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 2 and match_image_index1 == 4):
                im_2to4.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 2 and match_image_index1 == 5):
                im_2to5.append([intensity_val,im1_cord, im2_cord])
            
            if (image_number == 1 and match_image_index1 == 2):
                im_1to2.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 1 and match_image_index1 == 3):
                im_1to3.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 1 and match_image_index1 == 4):
                im_1to4.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 1 and match_image_index1 == 5):
                im_1to5.append([intensity_val,im1_cord, im2_cord])

        
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
            
            if (image_number == 3 and match_image_index1 == 4):
                im_3to4.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 3 and match_image_index2 == 4):
                im_3to4.append([intensity_val,im1_cord, im3_cord])
            
            if (image_number == 3 and match_image_index1 == 5):
                im_3to5.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 3 and match_image_index2 == 5):
                im_3to5.append([intensity_val,im1_cord, im3_cord])
            
            if (image_number == 2 and match_image_index1 == 3):
                im_2to3.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 2 and match_image_index2 == 3):
                im_2to3.append([intensity_val,im1_cord, im3_cord])

            if (image_number == 2 and match_image_index1 == 4):
                im_2to4.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 2 and match_image_index2 == 4):
                im_2to4.append([intensity_val,im1_cord, im3_cord])

            if (image_number == 2 and match_image_index1 == 5):
                im_2to5.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 2 and match_image_index2 == 5):
                im_2to5.append([intensity_val,im1_cord, im3_cord])
            
            if (image_number == 1 and match_image_index1 == 2):
                im_1to2.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 1 and match_image_index2 == 2):
                im_1to2.append([intensity_val,im1_cord, im3_cord])

            if (image_number == 1 and match_image_index1 == 3):
                im_1to3.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 1 and match_image_index2 == 3):
                im_1to3.append([intensity_val,im1_cord, im3_cord])

            if (image_number == 1 and match_image_index1 == 4):
                im_1to4.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 1 and match_image_index2 == 4):
                im_1to4.append([intensity_val,im1_cord, im3_cord])

            if (image_number == 1 and match_image_index1 == 5):
                im_1to5.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 1 and match_image_index2 == 5):
                im_1to5.append([intensity_val,im1_cord, im3_cord])
        


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
            if (image_number == 2 and match_image_index1 == 3):
                im_2to3.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 2 and match_image_index2 == 3):
                im_2to3.append([intensity_val,im1_cord, im3_cord])
            if (image_number == 2 and match_image_index3 == 3):
                im_2to3.append([intensity_val,im1_cord, im4_cord])

            if (image_number == 2 and match_image_index1 == 4):
                im_2to4.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 2 and match_image_index2 == 4):
                im_2to4.append([intensity_val,im1_cord, im3_cord])
            if (image_number == 2 and match_image_index3 == 4):
                im_2to4.append([intensity_val,im1_cord, im4_cord])

            if (image_number == 2 and match_image_index1 == 5):
                im_2to5.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 2 and match_image_index2 == 5):
                im_2to5.append([intensity_val,im1_cord, im3_cord])
            if (image_number == 2 and match_image_index3 == 5):
                im_2to5.append([intensity_val,im1_cord, im4_cord])
            
            if (image_number == 1 and match_image_index1 == 2):
                im_1to2.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 1 and match_image_index2 == 2):
                im_1to2.append([intensity_val,im1_cord, im3_cord])
            if (image_number == 1 and match_image_index3 == 2):
                im_1to2.append([intensity_val,im1_cord, im4_cord])

            if (image_number == 1 and match_image_index1 == 3):
                im_1to3.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 1 and match_image_index2 == 3):
                im_1to3.append([intensity_val,im1_cord, im3_cord])
            if (image_number == 1 and match_image_index3 == 3):
                im_1to3.append([intensity_val,im1_cord, im4_cord])

            if (image_number == 1 and match_image_index1 == 4):
                im_1to4.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 1 and match_image_index2 == 4):
                im_1to4.append([intensity_val,im1_cord, im3_cord])
            if (image_number == 1 and match_image_index3 == 4):
                im_1to4.append([intensity_val,im1_cord, im4_cord])

            if (image_number == 1 and match_image_index1 == 5):
                im_1to5.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 1 and match_image_index2 == 5):
                im_1to5.append([intensity_val,im1_cord, im3_cord])
            if (image_number == 1 and match_image_index3 == 5):
                im_1to5.append([intensity_val,im1_cord, im4_cord])


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

            if (image_number == 1 and match_image_index1 == 2):
                im_1to2.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 1 and match_image_index2 == 2):
                im_1to2.append([intensity_val,im1_cord, im3_cord])
            if (image_number == 1 and match_image_index3 == 2):
                im_1to2.append([intensity_val,im1_cord, im4_cord])
            if (image_number == 1 and match_image_index4 == 2):
                im_1to2.append([intensity_val,im1_cord, im5_cord])
                
            if (image_number == 1 and match_image_index1 == 3):
                im_1to3.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 1 and match_image_index2 == 3):
                im_1to3.append([intensity_val,im1_cord, im3_cord])
            if (image_number == 1 and match_image_index3 == 3):
                im_1to3.append([intensity_val,im1_cord, im4_cord])
            if (image_number == 1 and match_image_index4 == 3):
                im_1to3.append([intensity_val,im1_cord, im5_cord])

            if (image_number == 1 and match_image_index1 == 4):
                im_1to4.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 1 and match_image_index2 == 4):
                im_1to4.append([intensity_val,im1_cord, im3_cord])
            if (image_number == 1 and match_image_index3 == 4):
                im_1to4.append([intensity_val,im1_cord, im4_cord])
            if (image_number == 1 and match_image_index4 == 4):
                im_1to4.append([intensity_val,im1_cord, im5_cord])

            if (image_number == 1 and match_image_index1 == 5):
                im_1to5.append([intensity_val,im1_cord, im2_cord])
            if (image_number == 1 and match_image_index2 == 5):
                im_1to5.append([intensity_val,im1_cord, im3_cord])
            if (image_number == 1 and match_image_index3 == 5):
                im_1to5.append([intensity_val,im1_cord, im4_cord])
            if (image_number == 1 and match_image_index4 == 5):
                im_1to5.append([intensity_val,im1_cord, im5_cord])

    
    return im_1to2, im_1to3, im_1to4, im_1to5, im_2to3, im_2to4, im_2to5, im_3to4, im_3to5, im_4to5

def find_common(inliers1_13, inliers1_12, inliers3_13, optimized_world_points_12):
    X = copy.deepcopy(optimized_world_points_12)
    common_inliers3_13 = []
    common_inliers1_13 = []
    new_inliers3_13 = []
    new_inliers1_13 = []
    common_worldpoints_13 = []
    count = 0
    flag = 0
    for i in range(len(inliers1_13)):
        flag = 0
        for j in range(len(inliers1_12)):
            if inliers1_13[i][0] == inliers1_12[j][0] and inliers1_13[i][1] == inliers1_12[j][1]:
                flag = 1
                stored_index = j
                break
        if flag == 1:
            count +=1
            common_inliers3_13.append(inliers3_13[i])
            common_inliers1_13.append(inliers1_13[i])
            common_worldpoints_13.append(optimized_world_points_12[stored_index])
        if flag == 0:
            new_inliers3_13.append(inliers3_13[i])
            new_inliers1_13.append(inliers1_13[i])
    print('common point count',count)
    return common_inliers3_13, common_inliers1_13, common_worldpoints_13, new_inliers3_13, new_inliers1_13 

