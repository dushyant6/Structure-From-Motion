#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 15:53:03 2022

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
from EssentialMatrixFromFundamentalMatrix import EssentialMatrixFromFundamentalMatrix
from ExtractCameraPose import ExtractCameraPose
from DisambiguateCameraPose import DisambiguateCameraPose
from NonlinearTriangulation import NonlinearTriangulation
from LinearPnP import LinearPnP
from LinearTriangulation import LinearTriangulation
from PnPRANSAC import PnPRANSAC
from NonlinearPnP import NonlinearPnP
from BuildVisibilityMatrix import BuildVisibilityMatrix
from random import randrange
import numpy as np
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
    #cv.imshow(string, img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()
    string = 'output_images/' + string + '.png'
    cv.imwrite(string, img)

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

def error_func(world_point, src_point, dst_point, P, P_dash, K):
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
    #print('error src', err_src)
    #print('error dst', err_dst)
    err = err_dst + err_src
    return err

def Bundle_Error(initial_list, All_Image_Points, Vis_Mat, P_size, W_size, K):
    PSet = np.reshape(initial_list[:P_size[0]*3*4], P_size)
    #print('recover size', PSet.shape)
    WorldSet = np.reshape(initial_list[P_size[0]*3*4: len(initial_list)], W_size)
    #print('recover world', WorldSet.shape)
    total_error = 0
    #print('P1', PSet[2])
    
    for i in range(WorldSet.shape[0]):
        #camera1 reprojection
        proj1 = np.dot(np.dot(K,PSet[0]), WorldSet[i])
        proj1 = proj1/proj1[2]
        err1_x =  proj1[0] - All_Image_Points[i][0][0]
        err1_y =  proj1[1] - All_Image_Points[i][0][1]
        err1 = (err1_x**2 + err1_y**2)
        err1 = Vis_Mat[0][i]*err1
        
        proj2 = np.dot(np.dot(K,PSet[1]), WorldSet[i])
        proj2 = proj2/proj2[2]
        err2_x =  proj2[0] - All_Image_Points[i][1][0]
        err2_y =  proj2[1] - All_Image_Points[i][1][1]
        err2 = (err2_x**2 + err2_y**2)
        err2 = Vis_Mat[1][i]*err2
        
        proj3 = np.dot(np.dot(K,PSet[2]), WorldSet[i])
        proj3 = proj3/proj3[2]
        err3_x =  proj3[0] - All_Image_Points[i][2][0]
        err3_y =  proj3[1] - All_Image_Points[i][2][1]
        err3 = (err3_x**2 + err3_y**2)
        err3 = Vis_Mat[2][i]*err3
        #print(err1, err2, err3)
        #print('total error',total_error)
        total_error = total_error + err1 + err2 + err3
    #print('total error', total_error)
    #print('number of points',WorldSet.shape[0])
    return total_error


start = time.time()
#image number 1
file4 = open(r'P3Data/matching1.txt', 'r')
file4_lines = []
for count,line in enumerate(file4):
    file4_lines.append(line)
    
# 1) Import matches of first image with all other images
img_1to2, img_1to3, img_1to4, img_1to5, _, _, _, _, _, _  = read_match_file(1, file4_lines, count)
# 2) Implement RANSAC to remove outliers from original matches
#get source points from the image file e.g. img_1to2
im1_12, im2_12 = create_src_dst(img_1to2)

# 3) Find fundamental matrix
inliers1_12, inliers2_12 = RANSAC(im1_12, im2_12, 5, 1000, 95)

sFinal,fFinal, inliers1_12f, inliers2_12f = getInliners(inliers1_12, inliers2_12)
print('rank of F', np.linalg.matrix_rank(fFinal))

img1 = cv.imread('P3Data/1.png')
img2 = cv.imread('P3Data/2.png')
newimage = cv.hconcat([img1, img2])
'''
h,w,c = img1.shape
for i in range(len(inliers1_12f)-170,len(inliers1_12f)-150):
    cv.line(newimage, (int(inliers1_12f[i][0]),int(inliers1_12f[i][1])), (int(inliers2_12f[i][0])+w,int(inliers2_12f[i][1])), [255,0,0], 1)

cv.imshow('matches after F Ransac', newimage)
cv.waitKey(0)
cv.destroyAllWindows()
'''
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


#reprojection plots
plot_images(1, P, K, inliers1_12f, optimized_world_points_12, 'img1 Non Linear world points')
plot_images(2, P_dash[real_pose], K, inliers2_12f, optimized_world_points_12, 'img2 Non Linear world points')

#Register next images
# find inlier matches between first and thirs image
im1_13, im3_13 = create_src_dst(img_1to3)
inliers1_13, inliers3_13 = RANSAC(im1_13, im3_13, 7, 700, 95)
_,_, inliers1_13f, inliers3_13f = getInliners(inliers1_13, inliers3_13)


# find matches common between first, second, and third images 
common_inliers3_13, common_inliers1_13, common_worldpoints_13, new_inliers3_13f, new_inliers1_13f = find_common(inliers1_13f, inliers1_12f, inliers3_13f, optimized_world_points_12)


# 8) Linear, RANSAC

PnP_Undecomposed, PnP_Matrix = LinearPnP(common_worldpoints_13, common_inliers3_13, K)

PnP_RANSAC_Matrix = PnPRANSAC(common_worldpoints_13, common_inliers3_13, K)
PnP_RANSAC_Matrix_isolated = np.dot(np.linalg.inv(K), PnP_RANSAC_Matrix) 


I = np.identity(3)

#plotting reprojections
plot_images(3, PnP_Undecomposed, I, common_inliers3_13, common_worldpoints_13, 'img3 undecomposed PnP Linear')
plot_images(3, PnP_RANSAC_Matrix, I, common_inliers3_13, common_worldpoints_13, 'img3 undecomposed PnP RANSAC')

plot_images(1, P, K, common_inliers1_13, common_worldpoints_13, 'img1 undecomposed_PnP_Linear')
plot_images(1, P, K, common_inliers1_13, common_worldpoints_13, 'img1 undecomposed_PnP_RANSAC')

x_proj = np.dot(PnP_RANSAC_Matrix, common_worldpoints_13[0])
x_proj = x_proj/x_proj[2]

x_proj1 = np.dot(np.dot(K,PnP_RANSAC_Matrix),common_worldpoints_13[0])
x_proj1 = x_proj1/x_proj1[2]


nonlinear_PnP_Undecomposed, decomposed_non_linear_PnP = NonlinearPnP(PnP_RANSAC_Matrix, common_worldpoints_13, common_inliers3_13, K)
plot_images(3, nonlinear_PnP_Undecomposed, I, common_inliers3_13, common_worldpoints_13, 'img3 undecomposed PnP Non Linear')

nonlinear_PnP_Undecomposed_isolated = np.dot(np.linalg.inv(K), nonlinear_PnP_Undecomposed)

new_world_points_13 = LinearTriangulation(new_inliers1_13f, new_inliers3_13f, P, nonlinear_PnP_Undecomposed_isolated, K)
plot_images(3, nonlinear_PnP_Undecomposed, I, new_inliers3_13f, new_world_points_13, 'img3 new world points linear')

optimized_world_points_13 = NonlinearTriangulation(P, nonlinear_PnP_Undecomposed_isolated, new_inliers1_13f, new_inliers3_13f, new_world_points_13, K)


count1 = 0
new_non_linear_13 = []

for i in range(len(new_world_points_13)):
    optimized_list13 = scipy.optimize.least_squares(error_func, new_world_points_13[i], ftol = 0.001, xtol=0.001 ,args = (new_inliers1_13f[i], new_inliers3_13f[i], P, nonlinear_PnP_Undecomposed_isolated, K))
    optim = optimized_list13.x
    optim = optim/ optim[3]
    new_non_linear_13.append(optim)

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
    All_Image_Points_list.append((inliers1_12f[i], inliers2_12f[i], a))
for i in range(len(new_inliers1_13f)):
    All_Image_Points_list.append((new_inliers1_13f[i], a, new_inliers3_13f[i]))

All_Image_Points = All_Image_Points_list.copy()
All_Image_Points = np.asarray(All_Image_Points)
for i in range(len(World_points)):
    for j in range(len(common_worldpoints_13)):
        if (World_points[i][0] == common_worldpoints_13[j][0] and World_points[i][1] == common_worldpoints_13[j][1]):
            All_Image_Points[i][2] = common_inliers3_13[j]

# Append all camera matrices
P_set = []
P_set.append(P)
P_set.append(P_dash[real_pose])
P_set.append(nonlinear_PnP_Undecomposed_isolated)

PSet = np.asarray(P_set)
P_size = PSet.shape
print('P set shape', P_size)

# 10) Visibility matrix
list_len1 = len(inliers1_12f)
list_len2 = len(inliers3_13f)
Vis_Mat = BuildVisibilityMatrix(World_points, common_worldpoints_13, list_len1, list_len2)


# 11) Bundle Adjustment
#create initial parameter list
WorldSet = np.asarray(World_points)
initial_list = np.hstack((PSet.ravel(), WorldSet.ravel()))
W_size = WorldSet.shape
    

total_error = Bundle_Error(initial_list, All_Image_Points, Vis_Mat, P_size, W_size, K)
print('initial error', total_error)

optimized_list = scipy.optimize.least_squares(Bundle_Error, initial_list, xtol=1e-9, args =[All_Image_Points, Vis_Mat, P_size, W_size, K])
P_and_W = optimized_list.x

PSet_BA = np.reshape(P_and_W[:P_size[0]*3*4], P_size)
#print('recover size', PSet.shape)
WorldSet_BA = np.reshape(P_and_W[P_size[0]*3*4: len(initial_list)], W_size)


#Register next image - image4
# find inlier matches between first and forth image
ref_inliers123 = inliers1_12f.copy()
for i in range(len(new_inliers1_13f)):
    ref_inliers123.append(new_inliers1_13f[i])
    
im1_14, im4_14 = create_src_dst(img_1to4)
inliers1_14, inliers4_14 = RANSAC(im1_14, im4_14, 7, 700, 95)
_,_, inliers1_14f, inliers4_14f = getInliners(inliers1_14, inliers4_14)


# find matches common between first, second, and third images 
common_inliers4_14, common_inliers1_14, common_worldpoints_14, new_inliers4_14f, new_inliers1_14f = find_common(inliers1_14f, ref_inliers123, inliers4_14f, World_points)


# 8) Linear, RANSAC
PnP_Undecomposed_4, PnP_Matrix_4 = LinearPnP(common_worldpoints_14, common_inliers4_14, K)

PnP_RANSAC_Matrix_4 = PnPRANSAC(common_worldpoints_14, common_inliers4_14, K)
PnP_RANSAC_Matrix_isolated_4 = np.dot(np.linalg.inv(K), PnP_RANSAC_Matrix_4) 


I = np.identity(3)

#plotting reprojections
plot_images(4, PnP_Undecomposed_4, I, common_inliers4_14, common_worldpoints_14, 'img4 undecomposed PnP Linear')
plot_images(4, PnP_RANSAC_Matrix_4, I, common_inliers4_14, common_worldpoints_14, 'img4 undecomposed PnP RANSAC')


x_proj = np.dot(PnP_RANSAC_Matrix_4, common_worldpoints_14[0])
x_proj = x_proj/x_proj[2]

x_proj1 = np.dot(np.dot(K,PnP_RANSAC_Matrix_4),common_worldpoints_14[0])
x_proj1 = x_proj1/x_proj1[2]


nonlinear_PnP_Undecomposed_4, decomposed_non_linear_PnP_4 = NonlinearPnP(PnP_RANSAC_Matrix_4, common_worldpoints_14, common_inliers4_14, K)
plot_images(4, nonlinear_PnP_Undecomposed_4, I, common_inliers4_14, common_worldpoints_14, 'img4 undecomposed PnP Non Linear')

nonlinear_PnP_Undecomposed_isolated_4 = np.dot(np.linalg.inv(K), nonlinear_PnP_Undecomposed_4)

new_world_points_14 = LinearTriangulation(new_inliers1_14f, new_inliers4_14f, P, nonlinear_PnP_Undecomposed_isolated_4, K)
plot_images(4, nonlinear_PnP_Undecomposed_4, I, new_inliers4_14f, new_world_points_14, 'img4 new world points linear')

#optimized_world_points_13 = NonlinearTriangulation(P, nonlinear_PnP_Undecomposed_isolated, new_inliers1_13f, new_inliers3_13f, new_world_points_13, K)


count1 = 0
new_non_linear_14 = []

for i in range(len(new_world_points_14)):
    optimized_list14 = scipy.optimize.least_squares(error_func, new_world_points_14[i], ftol = 0.001, xtol=0.001 ,args = (new_inliers1_14f[i], new_inliers4_14f[i], P, nonlinear_PnP_Undecomposed_isolated_4, K))
    optim = optimized_list14.x
    optim = optim/ optim[3]
    new_non_linear_14.append(optim)



World_points1234 = World_points.copy()
Linear_World_Points1234 = Linear_World_Points.copy()
for i in range(len(new_world_points_14)):
    World_points1234.append(new_non_linear_14[i])
    Linear_World_Points1234.append(new_world_points_14[i])



#Register next image - image5
# find inlier matches between first and forth image
ref_inliers1234 = ref_inliers123.copy()
for i in range(len(new_inliers1_14f)):
    ref_inliers123.append(new_inliers1_14f[i])
    
im1_15, im5_15 = create_src_dst(img_1to5)
inliers1_15, inliers5_15 = RANSAC(im1_15, im5_15, 7, 700, 95)
_,_, inliers1_15f, inliers5_15f = getInliners(inliers1_15, inliers5_15)


# find matches common between first, second, and third images 
common_inliers5_15, common_inliers1_15, common_worldpoints_15, new_inliers5_15f, new_inliers1_15f = find_common(inliers1_15f, ref_inliers1234, inliers5_15f, World_points1234)
end = time.time()
print('total time', end - start)

# 8) Linear, RANSAC
PnP_Undecomposed_5, PnP_Matrix_5 = LinearPnP(common_worldpoints_15, common_inliers5_15, K)

PnP_RANSAC_Matrix_5 = PnPRANSAC(common_worldpoints_15, common_inliers5_15, K)
PnP_RANSAC_Matrix_isolated_5 = np.dot(np.linalg.inv(K), PnP_RANSAC_Matrix_5) 


I = np.identity(3)

#plotting reprojections
plot_images(5, PnP_Undecomposed_5, I, common_inliers5_15, common_worldpoints_15, 'img5 undecomposed PnP Linear')
plot_images(5, PnP_RANSAC_Matrix_5, I, common_inliers5_15, common_worldpoints_15, 'img5 undecomposed PnP RANSAC')


x_proj = np.dot(PnP_RANSAC_Matrix_5, common_worldpoints_15[0])
x_proj = x_proj/x_proj[2]

x_proj1 = np.dot(np.dot(K,PnP_RANSAC_Matrix_5),common_worldpoints_15[0])
x_proj1 = x_proj1/x_proj1[2]


nonlinear_PnP_Undecomposed_5, decomposed_non_linear_PnP_5 = NonlinearPnP(PnP_RANSAC_Matrix_5, common_worldpoints_15, common_inliers5_15, K)
plot_images(5, nonlinear_PnP_Undecomposed_5, I, common_inliers5_15, common_worldpoints_15, 'img5 undecomposed PnP Non Linear')

nonlinear_PnP_Undecomposed_isolated_5 = np.dot(np.linalg.inv(K), nonlinear_PnP_Undecomposed_5)

new_world_points_15 = LinearTriangulation(new_inliers1_15f, new_inliers5_15f, P, nonlinear_PnP_Undecomposed_isolated_5, K)
plot_images(5, nonlinear_PnP_Undecomposed_5, I, new_inliers5_15f, new_world_points_15, 'img5 new world points linear')

#optimized_world_points_13 = NonlinearTriangulation(P, nonlinear_PnP_Undecomposed_isolated, new_inliers1_13f, new_inliers3_13f, new_world_points_13, K)


count1 = 0
new_non_linear_15 = []

for i in range(len(new_world_points_15)):
    optimized_list15 = scipy.optimize.least_squares(error_func, new_world_points_15[i], ftol = 0.001, xtol=0.001 ,args = (new_inliers1_15f[i], new_inliers5_15f[i], P, nonlinear_PnP_Undecomposed_isolated_5, K))
    optim = optimized_list15.x
    optim = optim/ optim[3]
    new_non_linear_15.append(optim)


World_points12345 = World_points1234.copy()
Linear_World_Points12345 = Linear_World_Points1234.copy()
for i in range(len(new_world_points_15)):
    Linear_World_Points12345.append(new_world_points_15[i])
    World_points1234.append(new_non_linear_15[i])
    

World15_x = np.zeros([len(World_points12345),1])
World15_z = np.zeros([len(World_points12345),1])

Linear_World15_x = np.zeros([len(World_points12345),1])
Linear_World15_z = np.zeros([len(World_points12345),1])


for i in range(len(World_points12345)):
    World15_x[i] = World_points12345[i][0]
    World15_z[i] = World_points12345[i][2]
    Linear_World15_x[i] = Linear_World_Points12345[i][0]
    Linear_World15_z[i] = Linear_World_Points12345[i][2]


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



plt.scatter(World15_x, World15_z, s = 1)
plt.scatter(Linear_World15_x, Linear_World15_z, s = 1)
plt.scatter(P1_x, P1_z, s = 60, marker="s")
plt.scatter(P2_x, P2_z, s = 60, marker="^")
plt.scatter(P3_x, P3_z, s = 60, marker="<")
plt.scatter(P4_x, P4_z, s = 60, marker=">")
plt.scatter(P5_x, P5_z, s = 60, marker="v")


plt.title('World points including image 1, 2 ,3, 4,  5')
plt.axis('off')
plt.xlim([-15, 15])
plt.ylim([-5, 25])
plt.show()
