#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 15:58:52 2022

@author: dushyant
"""

import numpy as np
import cv2 as cv
import copy
import scipy
from scipy import optimize
import matplotlib
import matplotlib.pyplot as plt
import read_matches
import RANSAC
import get_src_dst
from read_matches import read_match_file
from read_matches import find_common
from RANSAC import RANSAC
from get_src_dst import create_src_dst
from EstimateFundamental import EstimateFundamentalMatrix
from EstimateFundamental import getInliners
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from DisambiguateCameraPose import DisambiguateCameraPose
from NonlinearTriangulation import NonlinearTriangulation
from LinearPnP import LinearPnP
from LinearTriangulation import LinearTriangulation
from PnPRANSAC import PnPRANSAC
from NonlinearPnP import NonlinearPnP
from BuildVisibilityMatrix import BuildVisibilityMatrix
from BuildVisibilityMatrix import AdaptiveVisibility
from BundleAdjustment import Bundle_Error
import time

def plot_images(image_number, P_matrix, K_matrix, image_points, world_points, string):
    b = 'P3Data/%d.png'%image_number
    img1 = cv.imread(b)
    img = img1.copy()
    for i in range(len(image_points)):
        cv.circle(img, (int(image_points[i][0]),int(image_points[i][1])), 3, [0,0,255], 1)
        x_proj = np.dot(np.dot(K_matrix, P_matrix), world_points[i])
        x_proj = x_proj/x_proj[2]
        cv.circle(img, (int(x_proj[0]), int(x_proj[1])), 2, [255,0,0], -1)
    string = 'output_images/' + string + '.png'
    cv.imwrite(string, img)

def error_func(world_point, src_point, dst_point, P, P_dash, K):
    err = []
    src_proj = np.dot(np.dot(K,P), world_point)
    src_proj = src_proj/src_proj[2]
    dst_proj = np.dot(np.dot(K, P_dash), world_point)
    dst_proj = dst_proj/dst_proj[2]
    err_src_x = src_point[0] - src_proj[0]
    err_src_y = src_point[1] - src_proj[1]
    err_src = np.sqrt(err_src_x**2 + err_src_y**2)
    err_dst_x = dst_point[0] - dst_proj[0]
    err_dst_y = dst_point[1] - dst_proj[1]
    err_dst = np.sqrt(err_dst_x**2 + err_dst_y**2)
    err.append(err_src_x)
    err.append(err_src_y)
    err.append(err_dst_x)
    err.append(err_dst_y)
    return err

start = time.time()
#image number 1
file4 = open(r'P3Data/matching1.txt', 'r')
file4_lines = []
for count,line in enumerate(file4):
    file4_lines.append(line)
I = np.identity(3)
inliers_src = []
inliers_dst = []
#_______________________________________________________________________________________________________________________    
# 1) Import matches of first image with all other images
img_1to2, img_1to3, img_1to4, img_1to5, _, _, _, _, _, _  = read_match_file(1, file4_lines, count)
# 2) Implement RANSAC to remove outliers from original matches
#get source points from the image file e.g. img_1to2
im1_12, im2_12 = create_src_dst(img_1to2)

# 3) Find fundamental matrix
inliers1_12, inliers2_12 = RANSAC(im1_12, im2_12, 11, 1000, 95)

sFinal,fFinal, inliers1_12f, inliers2_12f = getInliners(inliers1_12, inliers2_12)
inliers_src.append(inliers1_12f)
inliers_dst.append(inliers2_12f)

print('rank of F', np.linalg.matrix_rank(fFinal))


P = np.zeros([3,4])
P[:,0:3] = np.identity(3)

# 4) Essential matrix
# intrinsic matrix
K = [[531.122155322710, 0, 407.192550839899],
     [0, 531.541737503901, 313.308715048366],
     [0, 0, 1]]

E = EssentialMatrixFromFundamentalMatrix(fFinal,K)
print('essential matrix', E)

# 5) Find P_dash matrix
# P_dash, R and C contains  
C, R, P_dash = ExtractCameraPose(E, K)
P = np.zeros([3,4])
P[0][0] = P[1][1] = P[2][2] = 1

# 6) Find real camera pose
Linear_World_points_12, real_pose = DisambiguateCameraPose(inliers1_12f, inliers2_12f, P, P_dash, K, R, C)
plot_images(1, P, K, inliers1_12f, Linear_World_points_12, 'img1 Linear world points')
plot_images(2, P_dash[real_pose], K, inliers2_12f, Linear_World_points_12, 'img2 Linear world points')
print('final P_dash matrix\n', P_dash[real_pose])

# 7) find optimized world points
optimized_world_points_12 = NonlinearTriangulation(P, P_dash[real_pose], inliers1_12f, inliers2_12f, Linear_World_points_12, K)

new_non_linear_12 = []
for i in range(len(Linear_World_points_12)):
    optimized_list12 = scipy.optimize.least_squares(error_func, Linear_World_points_12[i], ftol = 0.00001, xtol=0.00001 ,args = (inliers1_12f[i], inliers2_12f[i], P, P_dash[real_pose], K))
    optim = optimized_list12.x
    optim = optim/ optim[3]
    new_non_linear_12.append(optim)

optimized_world_points_12 = new_non_linear_12

#reprojection plots
plot_images(1, P, K, inliers1_12f, new_non_linear_12, 'img1 Non Linear world points')
plot_images(2, P_dash[real_pose], K, inliers2_12f, new_non_linear_12, 'img2 Non Linear world points')




#____________________________________________________________________________________________________________________
#Register next image image 3

im1_13, im3_13 = create_src_dst(img_1to3)
inliers1_13, inliers3_13 = RANSAC(im1_13, im3_13, 11, 700, 95)
_,_, inliers1_13f, inliers3_13f = getInliners(inliers1_13, inliers3_13)
inliers_src.append(inliers1_13f)
inliers_dst.append(inliers3_13f)


# find matches common between first, second, and third images 
common_inliers3_13, common_inliers1_13, common_worldpoints_13, new_inliers3_13f, new_inliers1_13f = find_common(inliers1_13f, inliers1_12f, inliers3_13f, optimized_world_points_12)

# 8) Linear, RANSAC
PnP_Undecomposed, PnP_Matrix = LinearPnP(common_worldpoints_13, common_inliers3_13, K)
PnP_RANSAC_Matrix = PnPRANSAC(common_worldpoints_13, common_inliers3_13, K)
PnP_RANSAC_Matrix_isolated = np.dot(np.linalg.inv(K), PnP_RANSAC_Matrix) 

nonlinear_PnP_Undecomposed, decomposed_non_linear_PnP = NonlinearPnP(PnP_RANSAC_Matrix, common_worldpoints_13, common_inliers3_13, K)
nonlinear_PnP_Undecomposed_isolated = np.dot(np.linalg.inv(K), nonlinear_PnP_Undecomposed)
new_world_points_13 = LinearTriangulation(new_inliers1_13f, new_inliers3_13f, P, nonlinear_PnP_Undecomposed_isolated, K)
optimized_world_points_13 = NonlinearTriangulation(P, nonlinear_PnP_Undecomposed_isolated, new_inliers1_13f, new_inliers3_13f, new_world_points_13, K)

new_non_linear_13 = []
for i in range(len(new_world_points_13)):
    optimized_list13 = scipy.optimize.least_squares(error_func, new_world_points_13[i], ftol = 0.00001, xtol=0.00001 ,args = (new_inliers1_13f[i], new_inliers3_13f[i], P, nonlinear_PnP_Undecomposed_isolated, K))
    optim = optimized_list13.x
    optim = optim/ optim[3]
    new_non_linear_13.append(optim)

#plotting reprojections
plot_images(3, PnP_Undecomposed, I, common_inliers3_13, common_worldpoints_13, 'img3 undecomposed PnP Linear')
plot_images(3, PnP_RANSAC_Matrix, I, common_inliers3_13, common_worldpoints_13, 'img3 undecomposed PnP RANSAC')
plot_images(1, P, K, common_inliers1_13, common_worldpoints_13, 'img1 undecomposed_PnP_Linear')
plot_images(1, P, K, common_inliers1_13, common_worldpoints_13, 'img1 undecomposed_PnP_RANSAC')
plot_images(3, nonlinear_PnP_Undecomposed, I, common_inliers3_13, common_worldpoints_13, 'img3 undecomposed PnP Non Linear')
plot_images(3, nonlinear_PnP_Undecomposed, I, new_inliers3_13f, new_world_points_13, 'img3 new world points linear')
plot_images(3, nonlinear_PnP_Undecomposed, I, new_inliers3_13f, new_non_linear_13, 'img3 new world points non linear')

Linear_World_Points = Linear_World_points_12.copy()
World_points = optimized_world_points_12.copy()
for i in range(len(new_non_linear_13)):
    World_points.append(new_non_linear_13[i])
    Linear_World_Points.append(new_world_points_13[i])

# Append all image points
All_Image_Points_list = []
a = np.zeros_like(inliers1_12f[0])
for i in range(len(inliers1_12f)):
    All_Image_Points_list.append((inliers1_12f[i], inliers2_12f[i], a, a, a))
for i in range(len(new_inliers1_13f)):
    All_Image_Points_list.append((new_inliers1_13f[i], a, new_inliers3_13f[i], a, a))
print('All image points length', len(All_Image_Points_list))

All_Image_Points = All_Image_Points_list.copy()
All_Image_Points = np.asarray(All_Image_Points)
for i in range(len(World_points)):
    for j in range(len(common_worldpoints_13)):
        if (World_points[i][0] == common_worldpoints_13[j][0] and World_points[i][1] == common_worldpoints_13[j][1]):
            All_Image_Points[i][2] = common_inliers3_13[j]
All_Image_Points_list = list(All_Image_Points)
# Append all camera matrices
P_set = []
P_set.append(P)
P_set.append(P_dash[real_pose])
P_set.append(nonlinear_PnP_Undecomposed_isolated)

PSet = np.asarray(P_set)
P_size = PSet.shape

# 10) Visibility matrix
Vis_Available = np.ones([2,len(inliers1_12f)])

list_len2 = len(inliers3_13f)
Vis_Mat_Ad = AdaptiveVisibility(3, Vis_Available, World_points, list_len2, common_worldpoints_13)

# 11) Bundle Adjustment
#create initial parameter list
WorldSet = np.asarray(World_points)
initial_list = np.hstack((PSet.ravel(), WorldSet.ravel()))
W_size = WorldSet.shape
    


optimized_list = scipy.optimize.least_squares(Bundle_Error, initial_list, xtol=1e-9, args =[All_Image_Points, Vis_Mat_Ad, P_size, W_size, K])
P_and_W = optimized_list.x

PSet_BA = np.reshape(P_and_W[:P_size[0]*3*4], P_size)
#print('recover size', PSet.shape)
WorldSet_BA = np.reshape(P_and_W[P_size[0]*3*4: len(initial_list)], W_size)

#____________________________________________________________________________________________________________________________________
#Register next image - image4

ref_inliers123 = inliers1_12f.copy()
for i in range(len(new_inliers1_13f)):
    ref_inliers123.append(new_inliers1_13f[i])
    
im1_14, im4_14 = create_src_dst(img_1to4)
inliers1_14, inliers4_14 = RANSAC(im1_14, im4_14, 11, 700, 95)
_,_, inliers1_14f, inliers4_14f = getInliners(inliers1_14, inliers4_14)
inliers_src.append(inliers1_14f)
inliers_dst.append(inliers4_14f)

common_inliers4_14, common_inliers1_14, common_worldpoints_14, new_inliers4_14f, new_inliers1_14f = find_common(inliers1_14f, ref_inliers123, inliers4_14f, World_points)

# 8) Linear, RANSAC
PnP_Undecomposed_4, PnP_Matrix_4 = LinearPnP(common_worldpoints_14, common_inliers4_14, K)
PnP_RANSAC_Matrix_4 = PnPRANSAC(common_worldpoints_14, common_inliers4_14, K)
PnP_RANSAC_Matrix_isolated_4 = np.dot(np.linalg.inv(K), PnP_RANSAC_Matrix_4) 
nonlinear_PnP_Undecomposed_4, decomposed_non_linear_PnP_4 = NonlinearPnP(PnP_RANSAC_Matrix_4, common_worldpoints_14, common_inliers4_14, K)
nonlinear_PnP_Undecomposed_isolated_4 = np.dot(np.linalg.inv(K), nonlinear_PnP_Undecomposed_4)

new_world_points_14 = LinearTriangulation(new_inliers1_14f, new_inliers4_14f, P, nonlinear_PnP_Undecomposed_isolated_4, K)

new_non_linear_14 = []
for i in range(len(new_world_points_14)):
    optimized_list14 = scipy.optimize.least_squares(error_func, new_world_points_14[i], ftol = 0.00001, xtol=0.00001 ,args = (new_inliers1_14f[i], new_inliers4_14f[i], P, nonlinear_PnP_Undecomposed_isolated_4, K))
    optim = optimized_list14.x
    optim = optim/ optim[3]
    new_non_linear_14.append(optim)

#reprojection plotting
plot_images(4, PnP_Undecomposed_4, I, common_inliers4_14, common_worldpoints_14, 'img4 undecomposed PnP Linear')
plot_images(4, PnP_RANSAC_Matrix_4, I, common_inliers4_14, common_worldpoints_14, 'img4 undecomposed PnP RANSAC')
plot_images(4, nonlinear_PnP_Undecomposed_4, I, common_inliers4_14, common_worldpoints_14, 'img4 undecomposed PnP Non Linear')
plot_images(4, nonlinear_PnP_Undecomposed_4, I, new_inliers4_14f, new_world_points_14, 'img4 new world points linear')
plot_images(4, nonlinear_PnP_Undecomposed_4, I, new_inliers4_14f, new_non_linear_14, 'img4 new world points non linear')

World_points1234 = World_points.copy()
Linear_World_Points1234 = Linear_World_Points.copy()
for i in range(len(new_world_points_14)):
    World_points1234.append(new_non_linear_14[i])
    Linear_World_Points1234.append(new_world_points_14[i])
    

# Append all image points
a = np.zeros_like(inliers1_12f[0])
for i in range(len(new_inliers1_14f)):
    All_Image_Points_list.append((new_inliers1_14f[i], a, a, new_inliers4_14f[i], a))

All_Image_Points = All_Image_Points_list.copy()
All_Image_Points = np.asarray(All_Image_Points)
for i in range(len(World_points1234)):
    for j in range(len(common_worldpoints_14)):
        if (World_points1234[i][0] == common_worldpoints_14[j][0] and World_points1234[i][1] == common_worldpoints_14[j][1]):
            All_Image_Points[i][3] = common_inliers4_14[j]
All_Image_Points_list = list(All_Image_Points)
print('All image points length', len(All_Image_Points_list))
# Append all camera matrices
P_set.append(nonlinear_PnP_Undecomposed_isolated_4)

PSet = np.asarray(P_set)
P_size = PSet.shape

# 10) Visibility matrix
Vis_Available = Vis_Mat_Ad
list_len2 = len(inliers4_14f)
Vis_Mat_Ad = AdaptiveVisibility(4, Vis_Available, World_points1234, list_len2, common_worldpoints_14)

# 11) Bundle Adjustment
#create initial parameter list
WorldSet = np.asarray(World_points1234)
initial_list = np.hstack((PSet.ravel(), WorldSet.ravel()))
W_size = WorldSet.shape
    

optimized_list = scipy.optimize.least_squares(Bundle_Error, initial_list, xtol=1e-8, args =[All_Image_Points, Vis_Mat_Ad, P_size, W_size, K])
P_and_W = optimized_list.x

PSet_BA = np.reshape(P_and_W[:P_size[0]*3*4], P_size)
#print('recover size', PSet.shape)
WorldSet_BA = np.reshape(P_and_W[P_size[0]*3*4: len(initial_list)], W_size)

#_________________________________________________________________________________________________________________________

#Register next image - image5
ref_inliers1234 = ref_inliers123.copy()
for i in range(len(new_inliers1_14f)):
    ref_inliers1234.append(new_inliers1_14f[i])
    
im1_15, im5_15 = create_src_dst(img_1to5)
inliers1_15, inliers5_15 = RANSAC(im1_15, im5_15, 11, 700, 95)
_,_, inliers1_15f, inliers5_15f = getInliners(inliers1_15, inliers5_15)
inliers_src.append(inliers1_15f)
inliers_dst.append(inliers5_15f)


common_inliers5_15, common_inliers1_15, common_worldpoints_15, new_inliers5_15f, new_inliers1_15f = find_common(inliers1_15f, ref_inliers1234, inliers5_15f, World_points1234)

PnP_Undecomposed_5, PnP_Matrix_5 = LinearPnP(common_worldpoints_15, common_inliers5_15, K)
PnP_RANSAC_Matrix_5 = PnPRANSAC(common_worldpoints_15, common_inliers5_15, K)
PnP_RANSAC_Matrix_isolated_5 = np.dot(np.linalg.inv(K), PnP_RANSAC_Matrix_5) 
nonlinear_PnP_Undecomposed_5, decomposed_non_linear_PnP_5 = NonlinearPnP(PnP_RANSAC_Matrix_5, common_worldpoints_15, common_inliers5_15, K)
nonlinear_PnP_Undecomposed_isolated_5 = np.dot(np.linalg.inv(K), nonlinear_PnP_Undecomposed_5)

new_world_points_15 = LinearTriangulation(new_inliers1_15f, new_inliers5_15f, P, nonlinear_PnP_Undecomposed_isolated_5, K)
new_non_linear_15 = []
for i in range(len(new_world_points_15)):
    optimized_list15 = scipy.optimize.least_squares(error_func, new_world_points_15[i], ftol = 0.00001, xtol=0.00001 ,args = (new_inliers1_15f[i], new_inliers5_15f[i], P, nonlinear_PnP_Undecomposed_isolated_5, K))
    optim = optimized_list15.x
    optim = optim/ optim[3]
    new_non_linear_15.append(optim)

#reprojection plotting 
plot_images(5, PnP_Undecomposed_5, I, common_inliers5_15, common_worldpoints_15, 'img5 undecomposed PnP Linear')
plot_images(5, PnP_RANSAC_Matrix_5, I, common_inliers5_15, common_worldpoints_15, 'img5 undecomposed PnP RANSAC')
plot_images(5, nonlinear_PnP_Undecomposed_5, I, common_inliers5_15, common_worldpoints_15, 'img5 undecomposed PnP Non Linear')
plot_images(5, nonlinear_PnP_Undecomposed_5, I, new_inliers5_15f, new_world_points_15, 'img5 new world points linear')
plot_images(5, nonlinear_PnP_Undecomposed_5, I, new_inliers5_15f, new_non_linear_15, 'img5 new world points non linear')


World_points12345 = World_points1234.copy()
Linear_World_Points12345 = Linear_World_Points1234.copy()
for i in range(len(new_world_points_15)):
    Linear_World_Points12345.append(new_world_points_15[i])
    World_points12345.append(new_non_linear_15[i])

# Append all image points
for i in range(len(new_inliers1_15f)):
    All_Image_Points_list.append((new_inliers1_15f[i], a, a, a, new_inliers5_15f[i]))

All_Image_Points = All_Image_Points_list.copy()
All_Image_Points = np.asarray(All_Image_Points)
for i in range(len(World_points12345)):
    for j in range(len(common_worldpoints_15)):
        if (World_points12345[i][0] == common_worldpoints_15[j][0] and World_points12345[i][1] == common_worldpoints_15[j][1]):
            All_Image_Points[i][4] = common_inliers5_15[j]

print('All image points length', len(All_Image_Points_list))
# Append all camera matrices
P_set.append(nonlinear_PnP_Undecomposed_isolated_5)

PSet = np.asarray(P_set)
P_size = PSet.shape
print('P set shape', P_size)

# 10) Visibility matrix
Vis_Available = Vis_Mat_Ad
list_len2 = len(inliers5_15f)
Vis_Mat_Ad = AdaptiveVisibility(5, Vis_Available, World_points12345, list_len2, common_worldpoints_15)



# 11) Bundle Adjustment
#create initial parameter list
WorldSet = np.asarray(World_points1234)
initial_list = np.hstack((PSet.ravel(), WorldSet.ravel()))
W_size = WorldSet.shape
    

optimized_list = scipy.optimize.least_squares(Bundle_Error, initial_list, xtol=1e-8, args =[All_Image_Points, Vis_Mat_Ad, P_size, W_size, K])
P_and_W = optimized_list.x

PSet_BA = np.reshape(P_and_W[:P_size[0]*3*4], P_size)
#print('recover size', PSet.shape)
WorldSet_BA = np.reshape(P_and_W[P_size[0]*3*4: len(initial_list)], W_size)


end = time.time()
print('total time', end - start)

#uncomment below code to see the scatter plot of x vs z for all world points and camera poses
'''
World15_x = np.zeros([len(World_points12345),1])
World15_z = np.zeros([len(World_points12345),1])
Linear_World15_x = np.zeros([len(World_points12345),1])
Linear_World15_z = np.zeros([len(World_points12345),1])

for i in range(len(World_points12345)):
    World15_x[i] = World_points12345[i][0]
    World15_z[i] = World_points12345[i][2]
    Linear_World15_x[i] = Linear_World_Points12345[i][0]
    Linear_World15_z[i] = Linear_World_Points12345[i][2]

#______________________________________________________________________________________________________________________
P1_x = 0
P1_z = 0
C2 = -1* np.dot(np.linalg.inv(R[real_pose]), P_dash[real_pose][:,3])
P2_x = C2[0]
P2_z = C2[2]

R3 = nonlinear_PnP_Undecomposed_isolated[:,0:3]
C3 = -1* np.dot(np.linalg.inv(R3), nonlinear_PnP_Undecomposed_isolated[:,3])
P3_x = C3[0]
P3_z = C3[2]

R4 = nonlinear_PnP_Undecomposed_isolated_4[:,0:3]
C4 = -1* np.dot(np.linalg.inv(R4), nonlinear_PnP_Undecomposed_isolated_4[:,3])
P4_x = C4[0]
P4_z = C4[2]

R5 = nonlinear_PnP_Undecomposed_isolated_5[:,0:3]
C5 = -1* np.dot(np.linalg.inv(R5), nonlinear_PnP_Undecomposed_isolated_5[:,3])
P5_x = C5[0]
P5_z = C5[2]
#______________________________________________________________________________________________________________________

plt.scatter(World15_x, World15_z, s = 1)
plt.scatter(Linear_World15_x, Linear_World15_z, s = 1)
plt.scatter(P1_x, P1_z, s = 60, marker="s")
plt.scatter(P2_x, P2_z, s = 60, marker="^")
plt.scatter(P3_x, P3_z, s = 60, marker="<")
plt.scatter(P4_x, P4_z, s = 60, marker=">")
plt.scatter(P5_x, P5_z, s = 60, marker="v")

plt.title('World Points including Image 1, 2 ,3, 4, 5')
plt.axis('off')
plt.xlim([-15, 15])
plt.ylim([-5, 25])
plt.show()
'''