#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 13:20:37 2022

@author: dushyant
"""
import numpy as np

def EssentialMatrixFromFundamentalMatrix(F, K):
    K1 = K.copy()
    K_tranpose = np.transpose(K1)
    E = np.dot(np.dot(K_tranpose, F), K)
    U_ess, S_ess, V_ess = np.linalg.svd(E)
    print('singular values of E',S_ess)
    S_ess[0] = 1
    S_ess[1] = 1
    S_ess[2] = 0
    S_ess2 = np.diag(S_ess)
    E_new = np.dot(np.dot(U_ess, S_ess2), V_ess)
    #print('rank of E', np.linalg.matrix_rank(E_new))
    return E_new
