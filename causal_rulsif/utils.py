# FILENAME: utils.py
# Author: Shanyun Gao
# Paper: https://openreview.net/forum?id=3tV5AtAXk0
# Note that this code utilizes the old verision of tigramite package(v5.1.0), with a significant portion aligning with the logic present in functions from the tigramite package.
# The current version of CPD algorithm is designed to handle only one change point. For multiple change points, divide the data manually into segments with one change point each, run the algorithm on each segment with the same parameters, and then merge the results.

import sys
sys.path.insert(0,'/home/gao565/PCMCI_CP') 

import os
from tqdm import tqdm
import time
import itertools
from cmath import sqrt
from pickle import TRUE
import numpy as np
from numpy import sum as sum
import matplotlib
from matplotlib import pyplot as plt
import statsmodels.formula.api as sm
import sklearn
from matplotlib.backends.backend_pdf import PdfPages
from numpy import nan
from math import isnan
from copy import deepcopy
from numpy.random.mtrand import random_integers
import pandas as pd
import random
import os
import tigramite
import scipy
import math
from tigramite import data_processing as pp
from tigramite.toymodels import structural_causal_processes as toys
from tigramite import independence_tests
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramite.models import LinearMediation, Prediction
import random
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from numpy import nan
from math import isnan
from copy import deepcopy
import os
import math
import cvxpy as cp
from scipy.special import xlogy
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pickle
from math import copysign


def divide(numerator, denominator):
    if denominator == 0.0:
        return copysign(float('inf'), denominator)
    return numerator / denominator

def findPeaks_inIterval_n(arr, n) :

  peak=[]
  arr_peak = []
  # check for every other element
  for i in range(n, len(arr)) :

      # check if the neighbors are smaller
      if (arr[i] >= np.max(arr[i - n:i+n])) :
        peak.append(i)
        arr_peak.append(arr[i])
  arr_peak=np.array(arr_peak)
  peak=np.array(peak)
  arr_peakinds = arr_peak.argsort()
  # print(arr_peak)
  # print(arr_peakinds)
  sorted_peak1 = peak[arr_peakinds[::-1]]
  return np.array(sorted_peak1),arr_peak

def sliding_window(X, window_size, step):
    if len(X.shape)==1:
      X=X.reshape(1,X.shape[0])

    num_dims, num_samples = X.shape
    # print(math.floor(num_samples/window_size/step))
    windows = np.zeros((num_dims * window_size * step,  int(math.ceil((num_samples-window_size*step+1)/step))))
    # print(windows.shape)
    for i in range(0, num_samples, step):
        offset = window_size * step #10*1=10
        if i + offset > num_samples: #i+10>2002, so that i>1992 break so i=1993 break.
            break
        w = X[:, i:i + offset].T
        windows[:, math.ceil(i/step)] = w.ravel("F")

    return windows

def comp_dist(x, y):
    if len(x.shape)==1:
      x=x.reshape(1,x.shape[0])
    if len(y.shape)==1:
      y=y.reshape(1,y.shape[0])

    # print(x.shape)
    d, nx = x.shape
    d, ny = y.shape

    G_x = np.sum(x * x, axis=0) # G_x.shape=(nx,)
    # print(G_x)
    T = np.tile(G_x, (ny, 1)) #
    # print(T) T.shape=(ny,nx)
    G_y = np.sum(y * y, axis=0)
    R = np.tile(G_y, (nx, 1))
    #print(R) R.shape=(nx,ny)
    dist2 = T.T + R - 2 * np.dot(x.T, y) # is a nx*ny dimension matrix, no matter what is dx and dy.
            #(nx,ny)+(nx,ny)-2*(nx,d)*(d,ny)
    return dist2


def comp_med(x):
    if len(x.shape)==1:
        x=x.reshape(1,x.shape[0])

    d, n = x.shape
    G = np.sum(x * x, axis=0)
    T = np.tile(G, (n, 1))
    dist2 = T - 2 * np.dot(x.T, x) + T.T
    dist2 -= np.tril(dist2)
    R = dist2.ravel()
    med = np.sqrt(0.5 * np.median(R[R > 0]))
    return med

def kernel_gau(dist2, sigma):
    k = np.exp(-dist2 / (2 * sigma**2))
    return k

######################################## Original ReluLSIF ###################################
def RelULSIF(x_de, x_nu, x_re=None, x_ce=None, alpha=0.5, fold=5):
    np.random.seed(3)  # Reset random seed to default

    if fold is None:
        fold = 5

    if len(x_nu.shape)==1:
      x_nu=x_nu.reshape(1,x_nu.shape[0])
    if len(x_de.shape)==1:
      x_de=x_de.reshape(1,x_de.shape[0])

    _, n_nu = x_nu.shape
    _, n_de = x_de.shape

    # Parameter Initialization Section
    if x_ce is None:
        b = min(100, n_nu)
        idx = np.random.permutation(n_nu)
        # idx=[5,8,2,9,6,7,3,0,1,4]
        # print(idx)
        x_ce = x_nu[:, idx[:b]]

    if alpha is None:
        alpha = 0.5

    # Construct Gaussian centers
    if len(x_ce.shape)==1:
      x_ce=x_ce.reshape(1,x_ce.shape[0])

    _, n_ce = x_ce.shape

    # Get sigma candidates
    x = np.concatenate((x_nu, x_de), axis=1)
    med = comp_med(x)
    sigma_list = med * np.array([0.6, 0.8, 1.0, 1.2, 1.4])
    # print(sigma_list)
    # Get lambda candidates
    lambda_list = 10.0**np.arange(-3, 2)
    # print(lambda_list)
    dist2_de = comp_dist(x_de, x_ce)
    dist2_nu = comp_dist(x_nu, x_ce)
    # print(dist2_de)
    # print(dist2_nu)
    # Cross-validation Section Begins
    score = np.zeros((len(sigma_list), len(lambda_list)))

    for i in range(len(sigma_list)):
        k_de = kernel_gau(dist2_de, sigma_list[i])
        # print(k_de.shape) nx*ny
        k_nu = kernel_gau(dist2_nu, sigma_list[i])

        for j in range(len(lambda_list)):
            cv_index_nu = np.random.permutation(n_nu)
            # cv_index_nu = np.array([8,9,7,4,5,0,3,6,1,2])
            cv_split_nu = np.floor(np.arange(n_nu) * fold / n_nu).astype(int) + 1
            cv_index_de = np.random.permutation(n_de)
            # cv_index_de = np.array([8,9,7,4,5,0,3,6,1,2])
            cv_split_de = np.floor(np.arange(n_de) * fold / n_de).astype(int) + 1

            total_sum = 0

            for k in range(1, fold + 1):
                k_de_k = k_de[cv_index_de[cv_split_de != k], :].T
                # print(k_de_k.shape)
                k_nu_k = k_nu[cv_index_nu[cv_split_nu != k],:].T
                H_k = ((1 - alpha) / k_de_k.shape[1]) * np.dot(k_de_k, k_de_k.T) + \
                      (alpha / k_nu_k.shape[1]) * np.dot(k_nu_k, k_nu_k.T)
                h_k = np.mean(k_nu_k, axis=1)
                theta = np.linalg.solve(H_k + np.eye(n_ce) * lambda_list[j], h_k)
                k_de_test = k_de[cv_index_de[cv_split_de == k],: ].T
                k_nu_test = k_nu[cv_index_nu[cv_split_nu == k],:].T
                J = alpha / 2 * np.mean((np.dot(theta.T, k_nu_test))**2) + \
                    (1 - alpha) / 2 * np.mean((np.dot(theta.T, k_de_test))**2) - \
                    np.mean(np.dot(theta.T, k_nu_test))
                total_sum += J
            score[i, j] = total_sum / fold

    # Find the chosen sigma and lambda
    i_min, j_min = np.unravel_index(np.argmin(score), score.shape)
    sigma_chosen = sigma_list[i_min]
    lambda_chosen = lambda_list[j_min]

    # Compute the final result
    k_de = kernel_gau(dist2_de.T, sigma_chosen)
    k_nu = kernel_gau(dist2_nu.T, sigma_chosen)
    H = ((1 - alpha) / n_de) * np.dot(k_de, k_de.T) + \
        (alpha / n_nu) * np.dot(k_nu, k_nu.T)
    h = np.mean(k_nu, axis=1)
    theta = np.linalg.solve(H + np.eye(n_ce) * lambda_chosen, h)
    g_nu = np.dot(theta.T, k_nu)
    g_de = np.dot(theta.T, k_de)
    g_re = []

    if x_re is not None:
        dist2_re = comp_dist(x_re, x_ce)
        k_re = kernel_gau(dist2_re.T, sigma_chosen)
        g_re = np.dot(theta.T, k_re)

    rPE = np.mean(g_nu) - 1 / 2 * (alpha * np.mean(g_nu**2) +
                                   (1 - alpha) * np.mean(g_de**2)) - 1 / 2

    return rPE, g_nu, g_de, sigma_chosen, lambda_chosen

def change_detection(X, n, k, alpha, fold):
    SCORE = []
    WIN = sliding_window(X, k, 1)
    nSamples = WIN.shape[1]
    t = n
    sigma_track = []
    lambda_track = []

    while (t + n <= nSamples):
        Y = WIN[:, t - n:t + n]
        # print(Y)
        # print(Y.shape) #(k,2*n)
        # print(np.std(Y, axis=1))
        # print(np.tile(np.std(Y, axis=1).reshape((Y.shape[0],1)), (1, 2 * n)))
        Y = Y / np.tile(np.std(Y, axis=1).reshape((Y.shape[0],1)), (1, 2 * n))
        # YRef = Y[:, :n]
        # YTest = Y[:, n:]
        YRef = Y[:, n:]
        YTest = Y[:, :n]
        s, _, _, sig, lam = RelULSIF(YRef, YTest, None, None, alpha, fold)
        sigma_track.append(sig)
        lambda_track.append(lam)

        # Print out the progress
        if (t % 20 == 0):
            print(f". {t}")

        SCORE.append(s)
        t = t + 1

    return SCORE, sigma_track, lambda_track


def RelULSIF_fix(x_de, x_nu, x_re=None, x_ce=None, alpha=0.5, fold=5):
    np.random.seed(3)  # Reset random seed to default

    if fold is None:
        fold = 5

    if len(x_nu.shape)==1:
      x_nu=x_nu.reshape(1,x_nu.shape[0])
    if len(x_de.shape)==1:
      x_de=x_de.reshape(1,x_de.shape[0])

    _, n_nu = x_nu.shape
    _, n_de = x_de.shape

    # Parameter Initialization Section
    if x_ce is None:
        b = min(100, n_nu)
        idx = np.random.permutation(n_nu)
        # idx=[5,8,2,9,6,7,3,0,1,4]
        # print(idx)
        x_ce = x_nu[:, idx[:b]]

    if alpha is None:
        alpha = 0.5

    # Construct Gaussian centers
    if len(x_ce.shape)==1:
      x_ce=x_ce.reshape(1,x_ce.shape[0])

    _, n_ce = x_ce.shape

    # Get sigma candidates
    x = np.concatenate((x_nu, x_de), axis=1)
    med = comp_med(x)
    sigma_list = med * np.array([0.6, 0.8, 1.0, 1.2, 1.4])
    # print(sigma_list)
    # Get lambda candidates
    lambda_list = 10.0**np.arange(-3, 2)
    # print(lambda_list)
    dist2_nu = comp_dist(x_nu, x_ce)
    dist2_de = comp_dist(x_de, x_ce)
    # print(dist2_de)
    # print(dist2_nu)
    # Cross-validation Section Begins
    score = np.zeros((len(sigma_list), len(lambda_list)))

    # for i in range(len(sigma_list)):
    #     k_de = kernel_gau(dist2_de, sigma_list[i])
    #     # print(k_de.shape) nx*ny
    #     k_nu = kernel_gau(dist2_nu, sigma_list[i])

    #     for j in range(len(lambda_list)):
    #         cv_index_nu = np.random.permutation(n_nu)
    #         # cv_index_nu = np.array([8,9,7,4,5,0,3,6,1,2])
    #         cv_split_nu = np.floor(np.arange(n_nu) * fold / n_nu).astype(int) + 1
    #         cv_index_de = np.random.permutation(n_de)
    #         # cv_index_de = np.array([8,9,7,4,5,0,3,6,1,2])
    #         cv_split_de = np.floor(np.arange(n_de) * fold / n_de).astype(int) + 1

    #         total_sum = 0

    #         for k in range(1, fold + 1):
    #             k_de_k = k_de[cv_index_de[cv_split_de != k], :].T
    #             # print(k_de_k.shape)
    #             k_nu_k = k_nu[cv_index_nu[cv_split_nu != k],:].T
    #             H_k = ((1 - alpha) / k_de_k.shape[1]) * np.dot(k_de_k, k_de_k.T) + \
    #                   (alpha / k_nu_k.shape[1]) * np.dot(k_nu_k, k_nu_k.T)
    #             h_k = np.mean(k_nu_k, axis=1)
    #             theta = np.linalg.solve(H_k + np.eye(n_ce) * lambda_list[j], h_k)
    #             k_de_test = k_de[cv_index_de[cv_split_de == k],: ].T
    #             k_nu_test = k_nu[cv_index_nu[cv_split_nu == k],:].T
    #             J = alpha / 2 * np.mean((np.dot(theta.T, k_nu_test))**2) + \
    #                 (1 - alpha) / 2 * np.mean((np.dot(theta.T, k_de_test))**2) - \
    #                 np.mean(np.dot(theta.T, k_nu_test))
    #             total_sum += J
    #         score[i, j] = total_sum / fold

    # # Find the chosen sigma and lambda
    # i_min, j_min = np.unravel_index(np.argmin(score), score.shape)
    # sigma_chosen = sigma_list[i_min]
    # lambda_chosen = lambda_list[j_min]
    ################ de is the second window, nu is the first window. ################
    ################ de is the second window, nu is the first window. ################
    ################ de is the second window, nu is the first window. ################
    # Compute the final result
    k_de = kernel_gau(dist2_de.T, sigma_list[2])
    k_nu = kernel_gau(dist2_nu.T, sigma_list[2])
    H = ((1 - alpha) / n_de) * np.dot(k_de, k_de.T) + \
        (alpha / n_nu) * np.dot(k_nu, k_nu.T)
    h = np.mean(k_nu, axis=1)
    theta = np.linalg.solve(H + np.eye(n_ce) * lambda_list[2], h)
    g_nu = np.dot(theta.T, k_nu) #first
    g_de = np.dot(theta.T, k_de) #second
    g_re = []

    if x_re is not None:
        dist2_re = comp_dist(x_re, x_ce)
        k_re = kernel_gau(dist2_re.T, sigma_list[2])
        g_re = np.dot(theta.T, k_re)

    rPE = np.mean(g_nu) - 1 / 2 * (alpha * np.mean(g_nu**2) +
                                   (1 - alpha) * np.mean(g_de**2)) - 1 / 2

    return rPE, g_nu, g_de

def change_detection_fix(X, n, k, alpha, fold):
    SCORE = []
    WIN = sliding_window(X, k, 1)
    nSamples = WIN.shape[1]
    t = n
    while (t + n <= nSamples):
        Y = WIN[:, t - n:t + n]
        # Y[:,:n] = Y_w1_fixed
        # Y[:,:n] = Y_w1_fixed
        # print(Y)
        # print(Y.shape) #(k,2*n)
        # print(np.std(Y, axis=1))
        # print(np.tile(np.std(Y, axis=1).reshape((Y.shape[0],1)), (1, 2 * n)))
        # Y = Y / np.tile(np.std(Y, axis=1).reshape((Y.shape[0],1)), (1, 2 * n))
        # YTest = WIN[:, t:t + n]
        Y = Y / np.tile(np.std(Y, axis=1).reshape((Y.shape[0],1)), (1, 2 * n))
        # YRef = Y[:, :n]
        # YTest = Y[:, n:]
        YRef = Y[:, n:] #de: the second window
        YTest = Y[:, :n] #nu: the first window
        s, _, _ = RelULSIF_fix(YRef, YTest, None, None, alpha, fold)
        # sigma_track.append(sig)
        # lambda_track.append(lam)

        # Print out the progress
        # if (t % 20 == 0):
        #     print(f". {t}")

        SCORE.append(s)
        t = t + 1

    return SCORE

def change_detection_onlycp_fix(X, n, Tc_most_close, k, alpha, fold):

  SCORE = []
  WIN = sliding_window(X, k, 1)
  if Tc_most_close-n<n:
    t=int(n)
  else:
    t = int(Tc_most_close-n)
  if Tc_most_close>WIN.shape[1]-2*n:
    nSamples =WIN.shape[1]
  else:
    nSamples=int(Tc_most_close+2*n)
  while (t + n < nSamples):
      Y = WIN[:, t - n:t + n]
      # Y[:,:n] = Y_w1_fixed
      # Y[:,:n] = Y_w1_fixed
      # print(Y)
      # print(Y.shape) #(k,2*n)
      # print(np.std(Y, axis=1))
      # print(np.tile(np.std(Y, axis=1).reshape((Y.shape[0],1)), (1, 2 * n)))
      # Y = Y / np.tile(np.std(Y, axis=1).reshape((Y.shape[0],1)), (1, 2 * n))
      # YTest = WIN[:, t:t + n]
      Y = Y / np.tile(np.std(Y, axis=1).reshape((Y.shape[0],1)), (1, 2 * n))
      # YRef = Y[:, :n]
      # YTest = Y[:, n:]
      YRef = Y[:, n:]
      YTest = Y[:, :n]
      s, _, _ = RelULSIF_fix(YRef, YTest, None, None, alpha, fold)
      # sigma_track.append(sig)
      # lambda_track.append(lam)

      # Print out the progress
      # if (t % 20 == 0):
      #     print(f". {t}")

      SCORE.append(s)
      t = t + 1

  return SCORE

def change_detection_onlycp(X, n, Tc_most_close, k, alpha, fold):

  SCORE = []
  WIN = sliding_window(X, k, 1)
  sigma_track = []
  lambda_track = []
  if Tc_most_close-n<n:
    t=int(n)
  else:
    t = int(Tc_most_close-n)
  if Tc_most_close>WIN.shape[1]-2*n:
    nSamples = WIN.shape[1]
  else:
    nSamples=int(Tc_most_close+2*n)
  while (t + n <= nSamples):
      Y = WIN[:, t - n:t + n]
      # Y[:,:n] = Y_w1_fixed
      # Y[:,:n] = Y_w1_fixed
      # print(Y)
      # print(Y.shape) #(k,2*n)
      # print(np.std(Y, axis=1))
      # print(np.tile(np.std(Y, axis=1).reshape((Y.shape[0],1)), (1, 2 * n)))
      # Y = Y / np.tile(np.std(Y, axis=1).reshape((Y.shape[0],1)), (1, 2 * n))
      # YTest = WIN[:, t:t + n]
      Y = Y / np.tile(np.std(Y, axis=1).reshape((Y.shape[0],1)), (1, 2 * n))
    #   YRef = Y[:, :n]
    #   YTest = Y[:, n:]
      YRef = Y[:, n:]
      YTest = Y[:, :n]
      s, _, _, sig, lam = RelULSIF(YRef, YTest, None, None, alpha, fold)
      sigma_track.append(sig)
      lambda_track.append(lam)

      # Print out the progress
      # if (t % 20 == 0):
      #     print(f". {t}")

      SCORE.append(s)
      t = t + 1

  return SCORE,sigma_track, lambda_track

def super_parent(data,T,N):
  datatime = np.arange(T-N)
  data_run = np.array(data[N:T,:])
  var_names=[]
  for i in range(N):
    var_names.append("$X^{}$".format(i))
  dataframe = pp.DataFrame(data_run,
                            datatime = {0:np.arange(len(data_run))},
                            var_names=var_names)
  #parcorr = ParCorr(significance='analytic')
  cmi_symb = CMIsymb(significance='shuffle_test')
  pcmci_cmi_symb = PCMCI(
      dataframe=dataframe,
      cond_ind_test=cmi_symb,
      verbosity=1)
  pcmci_cmi_symb.verbosity = 0
  results = pcmci_cmi_symb.run_pcmci(tau_min=1,tau_max=tau_max, pc_alpha=0.2, alpha_level=0.1)
  superset_bool=np.array(results["p_matrix"]<0.1, dtype=int)
  return superset_bool

def specific_configuration_data_generator(superset_bool,data,domain_of_var_vector,N,tau_max,k): # k=0 means the target variable is X_1
  # T=np.shape(data)[0]
  # print(T)
  superset_dict_est = {}
  n_value = len(domain_of_var_vector)
  for i in range(N):
      superset_dict_est[i] = []
  for h in range(N):
    for i in range(N):
      for j in range(len(superset_bool[0,0,:])):
        if superset_bool[i,h,j]==1:
          #  key='{}'.format(k)
          superset_dict_est[h].append((i,-j))
  # print(superset_dict_est)
  # if k==0:
  combine=np.zeros(np.power(n_value,sum(superset_bool[:,k,:])))
  # print(superset_bool[:,k,:])
  count_1=np.zeros(np.power(n_value,sum(superset_bool[:,k,:])))
  combine_len=len(combine)
  # print(combine_len)
  # print(combine_len)
  lst = list(itertools.product(domain_of_var_vector, repeat=sum(superset_bool[:,k,:])))
  count_index_1 = {}
  all_configuration = {}
  for g in range(combine_len):
    count_index_1[g] = []
    all_configuration[g] = []
  for g in range(combine_len):
    for i in range(tau_max,T):
      parent_index=np.zeros(sum(superset_bool[:,k,:]))
      for j in range(len(superset_dict_est[k])): #the order of superset_dict_est is: first parent variable index, then time lag.
        if data[i+superset_dict_est[k][j][1],superset_dict_est[k][j][0]]==lst[g][j]:
          parent_index[j]=1
      # print(parent_index)
      if sum(parent_index)==sum(superset_bool[:,k,:]):
        count_1[g]=count_1[g]+1
        count_index_1[g].append(i)
        # all_configuration[g].append(data[i,:].tolist())
        all_configuration[g].append(data[i,k].tolist())
  return count_1,count_index_1,all_configuration

######################################## ReluLSIF without Cross Validation + two estimators ###################################
def change_detection_two_esti(X, n, k, alpha, fold):
    SCORE = []
    WIN = sliding_window(X, k, 1)
    # print(WIN.shape)
    nSamples = WIN.shape[1]
    t = n
    sigma_track = []
    lambda_track = []
    Y_w1_fixed = WIN[:, 0:n]
    YRef = Y_w1_fixed
    while (t + n <= nSamples):
        # Y = WIN[:, t - n:t + n]
        # Y[:,:n] = Y_w1_fixed
        # Y[:,:n] = Y_w1_fixed
        # print(Y)
        # print(Y.shape) #(k,2*n)
        # print(np.std(Y, axis=1))
        # print(np.tile(np.std(Y, axis=1).reshape((Y.shape[0],1)), (1, 2 * n)))
        # Y = Y / np.tile(np.std(Y, axis=1).reshape((Y.shape[0],1)), (1, 2 * n))
        YTest = WIN[:, t:t + n]

        s, _, _ = RelULSIF_fix(YRef, YTest, None, None, alpha, fold)
        # sigma_track.append(sig)
        # lambda_track.append(lam)

        # Print out the progress
        if (t % 20 == 0):
            print(f". {t}")

        SCORE.append(s)
        t = t + k

    return SCORE

def change_detection_given_alpha_more_efficient(X,SCORE_list,esti_Tc,number_windows,n, k, fold):
  esti_alpha=Tc_determin_alpha_update(esti_Tc,window_size,number_windows)
  SCORE_list=np.array(SCORE_list)
  SCORE = np.zeros(len(esti_alpha))+300000
  if esti_Tc>=window_size:
    SCORE[esti_Tc-window_size:esti_Tc]=SCORE_list[0:window_size]
    SCORE[0:esti_Tc-window_size]=SCORE_list[0]
    SCORE[esti_Tc:number_windows]=SCORE_list[window_size]
  else:
    SCORE[0:esti_Tc]=SCORE[window_size-esti_Tc:window_size]
    SCORE[esti_Tc:number_windows]=SCORE_list[window_size]
  return SCORE

def Tc_determin_alpha_update(esti_Tc,window_size,number_windows):
  esti_alpha=Tc_determin_alpha(window_size,window_size,number_windows)
  esti_alpha_update=esti_alpha
  # print("esti_Tc"+str(esti_Tc))
  if esti_Tc>=window_size:
    # print("first len"+str(len(esti_alpha_update[esti_Tc-window_size:esti_Tc])))
    # print("second len"+str(len(esti_alpha[0:window_size])))
    esti_alpha_update[esti_Tc-window_size:esti_Tc]=esti_alpha[0:window_size]
    esti_alpha_update[0:esti_Tc-window_size]=1
    esti_alpha_update[esti_Tc:number_windows]=0
  else:
    esti_alpha_update[0:esti_Tc]=esti_alpha[window_size-esti_Tc:window_size]
    esti_alpha_update[esti_Tc:number_windows]=0
  return esti_alpha_update

def Tc_determin_alpha(esti_Tc,window_size,number_windows):
  esti_alpha= np.zeros(number_windows)+3000
  for i in range(number_windows):
    if i+window_size <= esti_Tc:
      # print(i)
      esti_alpha[i] = 1
    elif i >= esti_Tc:
      esti_alpha[i] = 0
    else:
      esti_alpha[i] = -1/window_size * i + esti_Tc/window_size
  return esti_alpha

def score_dic(X,number_windows,n, k, fold):
  SCORE_list = []
  WIN = sliding_window(X, k, 1)
  # print(WIN.shape)
  nSamples = WIN.shape[1]
  t = n
  sigma_track = []
  lambda_track = []
  Y_w1_fixed = WIN[:, 0:n]
  YRef = Y_w1_fixed
  YTest = WIN[:, nSamples-n:nSamples ]
  alpha_list=Tc_determin_alpha(window_size,window_size,number_windows)
  for i in range(int(window_size+1)):
      alpha=alpha_list[i]
      # Y = WIN[:, t - n:t + n]
      # Y[:,:n] = Y_w1_fixed
      # Y[:,:n] = Y_w1_fixed
      # print(Y)
      # print(Y.shape) #(k,2*n)
      # print(np.std(Y, axis=1))
      # print(np.tile(np.std(Y, axis=1).reshape((Y.shape[0],1)), (1, 2 * n)))
      # Y = Y / np.tile(np.std(Y, axis=1).reshape((Y.shape[0],1)), (1, 2 * n))

      s, _, _= RelULSIF_fix(YRef, YTest, None, None, alpha, fold)
      # sigma_track.append(sig)
      # lambda_track.append(lam)

      # Print out the progress
      # print(f". {alpha}")

      SCORE_list.append(s)
  return SCORE_list

def change_detection_onlycp_fix(X, n, Tc_most_close, k, alpha, fold):

  SCORE = []
  WIN = sliding_window(X, k, 1)
  if Tc_most_close-n<n:
    t=int(n)
  else:
    t = int(Tc_most_close-n)
  if Tc_most_close>WIN.shape[1]-2*n:
    nSamples =WIN.shape[1]
  else:
    nSamples=int(Tc_most_close+2*n)
  while (t + n < nSamples):
      Y = WIN[:, t - n:t + n]
      # Y[:,:n] = Y_w1_fixed
      # Y[:,:n] = Y_w1_fixed
      # print(Y)
      # print(Y.shape) #(k,2*n)
      # print(np.std(Y, axis=1))
      # print(np.tile(np.std(Y, axis=1).reshape((Y.shape[0],1)), (1, 2 * n)))
      # Y = Y / np.tile(np.std(Y, axis=1).reshape((Y.shape[0],1)), (1, 2 * n))
      # YTest = WIN[:, t:t + n]
      Y = Y / np.tile(np.std(Y, axis=1).reshape((Y.shape[0],1)), (1, 2 * n))
      # YRef = Y[:, :n]
      # YTest = Y[:, n:]
      YRef = Y[:, n:]
      YTest = Y[:, :n]
      s, _, _ = RelULSIF_fix(YRef, YTest, None, None, alpha, fold)
      # sigma_track.append(sig)
      # lambda_track.append(lam)

      # Print out the progress
      # if (t % 20 == 0):
      #     print(f". {t}")

      SCORE.append(s)
      t = t + 1

  return SCORE
######################################## Discrete Data Generation ###################################
######################################## Discrete Data Generation ###################################
######################################## Discrete Data Generation ###################################
######################################## Discrete Data Generation ###################################
######################################## Discrete Data Generation ###################################
# constraints: each time series should have the same number of change point
def discrete_data_generation_cp_model(T,N,tau_max,Tc_matrix,alpha_matrix,domain_of_var_vector,complexity,flag_tau_equal_1):
  if alpha_matrix.shape[1]!= len(domain_of_var_vector):
    raise ValueError("The dim of alpha_matirx is not consistent with the dim of domain_of_var_vector")
  for k in range(N):
    Tc_matrix[k]=np.array(Tc_matrix[k])
  if (Tc_matrix[0]==None).all():
    n_cp = 0
  else:
    n_cp = len(Tc_matrix[0]) # since every time series have the same number of change point, hence Tc_matric[0]=Tc_matric[1]=...Tc_matric[N]
  n_regime = n_cp+1
  # Generate an edge matrix with shape (N,n_cp,N,tau_max+1):
  edge_matrix = np.zeros((N,n_regime,N,tau_max+1))
  for i in range(N): # the parent var index
    for j in range(n_regime): # the regime index
      for k in range(N):  # the target var index
        if (flag_tau_equal_1==1) & (i==k): # if flag_tau_equal_1 = 1 meaning the edge between X^j_{t-1} and X^j_{t} must exist.
            edge_matrix[i,j,k,1] = 1
        else: # if flag_tau_equal_1 = 0 meaning the edge between X^j_{t-1} and X^j_{t} could not exist.
          for tau in range(1,tau_max+1): # time lag index
            edge_flag = np.random.binomial(1, complexity, size=None) # the complexity denote the edge_matrix complexity.
            edge_matrix[i,j,k,tau] = edge_flag
  # Based on the edge_matrix, we know the parent set for each variable at each regime.
  # Now needs to generate the conditional distribution based on dirichlet distributions (given alpha_vector and domain_of_var_vector)
  # alpha_matrix: shape is [N,len(domain_of_var_vector)]. Refer to https://stats.stackexchange.com/questions/244917/what-exactly-is-the-alpha-in-the-dirichlet-distribution for explaination
  #               Each row is for each time series X^j, j\in[N], and for each X^j, the conditional distribution of X^j given its parent relization have the same alpha_vetor value but
  #               the order could be different. For instance, if for X^0, alpha = [1,0.9], then given the parent relization 1, the alpha is [1,0.9], for relization 2, the alpha could be [1,0.9] or [0.9,1]
  # domain_of_var_vecotr: the value that variable can take. For binary variable, it is [0,1]

  # Generate the conditional distribution table given
  key_target = np.arange(N)
  var_dic={}
  for j in range(n_regime):
    var_dic[j]={}
  for j in range(n_regime):
    for k in range(N): # target
      var_dic[j][key_target[k]] = []
  for j in range(n_regime):# index of regime
    for k in range(N): # target variable (children)
      for i in range(N): # parent index
        for tau in range(tau_max+1):
          if edge_matrix[i,j,k,tau]==1:
            var_dic[j][key_target[k]].append((key_target[i],-tau))
   #In the var, the first dic key is the index of regime, in each regime, the dic key is the target variable and the pair in the [] is (parent_index, time_lag).
  # alpha_matrix=np.ones(shape=(N,len(domain_of_var_vector)))
  n_value = len(domain_of_var_vector)
  n_par = np.zeros(shape=(n_regime,N))
  conditional_table={}
  for j in range(n_regime):
    conditional_table[j]={}
  for j in range(n_regime):
    for k in range(N): # target
      conditional_table[j][key_target[k]] = []
  for j in range(n_regime):
    for k in range(N):
      n_par[j,k] = sum(edge_matrix[:,j,k,:])
      temp_n_par_relization = pow(n_value,n_par[j,k])
      conditional_table[j][k] = stats.dirichlet.rvs(np.roll(alpha_matrix[k],k),size=int(temp_n_par_relization))
  #For conditional table, {0: {0: array}, first 0 is the regime index, second 0 is the target variable index, the array [2^{number of parent}, n_value] is the conditional table.
  #Next step: generate time series data according to the conditional table
  data=np.zeros(shape=(T,N))
  for k in range(N):
    for t in range(tau_max):
      data[t,k] =np.random.choice(domain_of_var_vector, size=1, replace = True, p=[1/len(domain_of_var_vector)]*len(domain_of_var_vector))[0]
  # print(data[0:tau_max,:])
  ####### The above is for the starting point##################
  # print(time_point_array)
  for k in range(N):
    time_point_array = np.zeros(n_cp+2)
    time_point_array[0]=tau_max
    time_point_array[-1]=T
    if (Tc_matrix[k] != None).all():
      time_point_array[1:len(time_point_array)-1]=Tc_matrix[k]
    for j in range(n_regime):
      for t in range(int(time_point_array[j]),int(time_point_array[j+1])):
      # print(t,"t")
      # print(int(time_point_array[j]),int(time_point_array[j+1]))
        n_par = sum(edge_matrix[:,j,k,:]) # jth regime and k th variable; n_par: how many parents the kth variable in jth regime have
        n_par=int(n_par)
        parent_configuration_number = int(pow(n_value,sum(edge_matrix[:,j,k,:]))) # how many configurations: n_value ^ n_par
        # print(parent_configuration_number,"parent_configuration_number")
        parent_configuration_array = list(itertools.product(domain_of_var_vector, repeat=n_par)) # the list of all configurations.
        # print(parent_configuration_array,"parent_configuration_array",n_par,"n_par")
        parent_list=[]
        for n_par_index in range(n_par):
          # print(n_par,"n_par")
          # print(j,"j",k,"k")
          parent_list.append(data[t+var_dic[j][k][n_par_index][1],var_dic[j][k][n_par_index][0]])
        parent_list = np.array(parent_list)
        # print(parent_list," parent_list")
        for configuration_index in range(parent_configuration_number):
          # print(configuration_index)
          if (parent_list == parent_configuration_array[configuration_index]).all():
            temp_conditional_dis = conditional_table[j][k][configuration_index]
            break
        data[t,k] = np.random.choice(domain_of_var_vector, size=1, replace = True, p=temp_conditional_dis)[0]
  return data, conditional_table,edge_matrix,var_dic

############################### Data generation given skeleton and CPT #############################
def discrete_data_generation_given_CPT(T,N,tau_max,Tc_matrix,domain_of_var_vector,conditional_table,edge_matrix,var_dic):
  # Based on the edge_matrix, we know the parent set for each variable at each regime.
  # Now needs to generate the conditional distribution based on dirichlet distributions (given alpha_vector and domain_of_var_vector)
  # alpha_matrix: shape is [N,len(domain_of_var_vector)]. Refer to https://stats.stackexchange.com/questions/244917/what-exactly-is-the-alpha-in-the-dirichlet-distribution for explaination
  #               Each row is for each time series X^j, j\in[N], and for each X^j, the conditional distribution of X^j given its parent relization have the same alpha_vetor value but
  #               the order could be different. For instance, if for X^0, alpha = [1,0.9], then given the parent relization 1, the alpha is [1,0.9], for relization 2, the alpha could be [1,0.9] or [0.9,1]
  # domain_of_var_vecotr: the value that variable can take. For binary variable, it is [0,1]

  # Generate the conditional distribution table given

  #For conditional table, {0: {0: array}, first 0 is the regime index, second 0 is the target variable index, the array [2^{number of parent}, n_value] is the conditional table.
  #Next step: generate time series data according to the conditional table
  n_value = len(domain_of_var_vector)
  for k in range(N):
    Tc_matrix[k]=np.array(Tc_matrix[k])
  if (Tc_matrix[0]==None).all():
    n_cp = 0
  else:
    n_cp = len(Tc_matrix[0]) # since every time series have the same number of change point, hence Tc_matric[0]=Tc_matric[1]=...Tc_matric[N]
  n_regime = n_cp+1
  data=np.zeros(shape=(T,N))
  for k in range(N):
    for t in range(tau_max):
      data[t,k] =np.random.choice(domain_of_var_vector, size=1, replace = True, p=[1/len(domain_of_var_vector)]*len(domain_of_var_vector))[0]
  # print(data[0:tau_max,:])
  ####### The above is for the starting point##################
  # print(time_point_array)
  for k in range(N):
    time_point_array = np.zeros(n_cp+2)
    time_point_array[0]=tau_max
    time_point_array[-1]=T
    if (Tc_matrix[k] != None).all():
      time_point_array[1:len(time_point_array)-1]=Tc_matrix[k]
    for j in range(n_regime):
      for t in range(int(time_point_array[j]),int(time_point_array[j+1])):
      # print(t,"t")
      # print(int(time_point_array[j]),int(time_point_array[j+1]))
        n_par = sum(edge_matrix[:,j,k,:]) # jth regime and k th variable; n_par: how many parents the kth variable in jth regime have
        n_par=int(n_par)
        parent_configuration_number = int(pow(n_value,sum(edge_matrix[:,j,k,:]))) # how many configurations: n_value ^ n_par
        # print(parent_configuration_number,"parent_configuration_number")
        parent_configuration_array = list(itertools.product(domain_of_var_vector, repeat=n_par)) # the list of all configurations.
        # print(parent_configuration_array,"parent_configuration_array",n_par,"n_par")
        parent_list=[]
        for n_par_index in range(n_par):
          # print(n_par,"n_par")
          # print(j,"j",k,"k")
          parent_list.append(data[t+var_dic[j][k][n_par_index][1],var_dic[j][k][n_par_index][0]])
        parent_list = np.array(parent_list)
        # print(parent_list," parent_list")
        for configuration_index in range(parent_configuration_number):
          # print(configuration_index)
          if (parent_list == parent_configuration_array[configuration_index]).all():
            temp_conditional_dis = conditional_table[j][k][configuration_index]
            # print(temp_conditional_dis,configuration_index,t)
            break
        data[t,k] = np.random.choice(domain_of_var_vector, size=1, replace = True, p=temp_conditional_dis)[0]
  return data,time_point_array

######################################## Change Point Generation ###################################
######################################## Change Point Generation ###################################

def generate_Tc(scope_Tc,T,n,num_cp): #scope=1,2,3,4,5,6,7,8
  if scope_Tc==1:
    scope_Tc1=int((scope_Tc-1)/8*T)+n
  else:
    scope_Tc1=int((scope_Tc-1)/8*T)
  if scope_Tc==8:
    scope_Tc2=int(scope_Tc/8*T)-n
  else:
    scope_Tc2=int(scope_Tc/8*T)
  Tc = np.random.randint(scope_Tc1,scope_Tc2+1,size=num_cp)
  return Tc

# def generate_Tc(scale_Tc,T,n): #scope=1,2,3,4,5
#   Tc = np.random.randint(4*n,T-4*n,size=3)
#   Tc.sort()
#   if Tc[2]>Tc[0]+scale_Tc*T/8:
#     Tc[2] = Tc[0]+scale_Tc*T/8
#     Tc[1] = np.random.randint(Tc[0],Tc[2],size=1)[0]
#   random.shuffle(Tc)
#   return Tc
def discrete_data_generation_a_specific_cp_model_soft_interven(T,N,Tc_matrix,alpha_matrix,domain_of_var_vector):
  tau_max=2
  if alpha_matrix.shape[1]!= len(domain_of_var_vector):
    raise ValueError("The dim of alpha_matirx is not consistent with the dim of domain_of_var_vector")
  for k in range(N):
    Tc_matrix[k]=np.array(Tc_matrix[k])
  if (Tc_matrix[0]==None).all():
    n_cp = 0
  else:
    n_cp = len(Tc_matrix[0]) # since every time series have the same number of change point, hence Tc_matric[0]=Tc_matric[1]=...Tc_matric[N]
  n_regime = n_cp+1
  # Generate an edge matrix with shape (N,n_cp,N,tau_max+1):
  edge_matrix = np.zeros((N,n_regime,N,tau_max+1))

  ######################################## A specfic skeleton is generated ########################################
  ######################################## A specfic skeleton is generated ########################################
  ######################################## A specfic skeleton is generated ########################################

  for i in range(N): # the parent var index
    j=0
    # for j in range(n_regime): # the regime index
    for k in range(N):  # the target var index
      if i==k: # if flag_tau_equal_1 = 1 meaning the edge between X^j_{t-1} and X^j_{t} must exist.
        edge_matrix[i,j,k,1] = 1
      else:
        edge_matrix[i,j,k,1] = 1
        edge_matrix[i,j,k,2] = 1

  # print(edge_matrix[:,0,:,:])

  edge_matrix[:,1,:,:]=edge_matrix[:,0,:,:]

  # print(edge_matrix)
  # Based on the edge_matrix, we know the parent set for each variable at each regime.
  # Now needs to generate the conditional distribution based on dirichlet distributions (given alpha_vector and domain_of_var_vector)
  # alpha_matrix: shape is [N,len(domain_of_var_vector)]. Refer to https://stats.stackexchange.com/questions/244917/what-exactly-is-the-alpha-in-the-dirichlet-distribution for explaination
  #               Each row is for each time series X^j, j\in[N], and for each X^j, the conditional distribution of X^j given its parent relization have the same alpha_vetor value but
  #               the order could be different. For instance, if for X^0, alpha = [1,0.9], then given the parent relization 1, the alpha is [1,0.9], for relization 2, the alpha could be [1,0.9] or [0.9,1]
  # domain_of_var_vecotr: the value that variable can take. For binary variable, it is [0,1]

  # Generate the conditional distribution table given
  key_target = np.arange(N)
  var_dic={}
  for j in range(n_regime):
    var_dic[j]={}
  for j in range(n_regime):
    for k in range(N): # target
      var_dic[j][key_target[k]] = []
  for j in range(n_regime):# index of regime
    for k in range(N): # target variable (children)
      for i in range(N): # parent index
        for tau in range(tau_max+1):
          if edge_matrix[i,j,k,tau]==1:
            var_dic[j][key_target[k]].append((key_target[i],-tau))
   #In the var, the first dic key is the index of regime, in each regime, the dic key is the target variable and the pair in the [] is (parent_index, time_lag).
  # alpha_matrix=np.ones(shape=(N,len(domain_of_var_vector)))
  n_value = len(domain_of_var_vector)
  n_par = np.zeros(shape=(n_regime,N))
  conditional_table={}
  for j in range(n_regime):
    conditional_table[j]={}
  for j in range(n_regime):
    for k in range(N): # target
      conditional_table[j][key_target[k]] = []
  for j in range(n_regime):
    for k in range(N):
      n_par[j,k] = sum(edge_matrix[:,j,k,:])
      temp_n_par_relization = pow(n_value,n_par[j,k])
      conditional_table[j][k] = stats.dirichlet.rvs(np.roll(alpha_matrix[k],k),size=int(temp_n_par_relization))
  #For conditional table, {0: {0: array}, first 0 is the regime index, second 0 is the target variable index, the array [2^{number of parent}, n_value] is the conditional table.
  #Next step: generate time series data according to the conditional table
  data=np.zeros(shape=(T,N))
  for k in range(N):
    for t in range(tau_max):
      data[t,k] =np.random.choice(domain_of_var_vector, size=1, replace = True, p=[1/len(domain_of_var_vector)]*len(domain_of_var_vector))[0]
  # print(data[0:tau_max,:])
  ####### The above is for the starting point##################
  # print(time_point_array)
  time_point_array = np.zeros(shape=(N,n_cp+2))
  for k in range(N):
    time_point_array[k][0]=tau_max
    time_point_array[k][-1]=T
    if (Tc_matrix[k] != None).all():
      time_point_array[k][1:time_point_array.shape[1]-1]=Tc_matrix[k]
  # print(time_point_array)

  for t in range(tau_max,T):
    for k in range(N):
      for j in range(n_regime):
        if t in range(int(time_point_array[k][j]),int(time_point_array[k][j+1])):
          # print(t,"t",k,"k"). #######Check!!!!
    # print(int(time_point_array[j]),int(time_point_array[j+1])). #######Check!!!!
          n_par = sum(edge_matrix[:,j,k,:]) # jth regime and k th variable; n_par: how many parents the kth variable in jth regime have
          n_par=int(n_par)
          parent_configuration_number = int(pow(n_value,sum(edge_matrix[:,j,k,:]))) # how many configurations: n_value ^ n_par
          # print(parent_configuration_number,"parent_configuration_number")
          parent_configuration_array = list(itertools.product(domain_of_var_vector, repeat=n_par)) # the list of all configurations.
          # print(parent_configuration_array,"parent_configuration_array",n_par,"n_par")
          parent_list=[]
          for n_par_index in range(n_par):
            # print(n_par,"n_par")
            # print(j,"j",k,"k")
            parent_list.append(data[t+var_dic[j][k][n_par_index][1],var_dic[j][k][n_par_index][0]])
          parent_list = np.array(parent_list)
          # print(parent_list," parent_list"). #######Check!!!!
          for configuration_index in range(parent_configuration_number):
            # print(configuration_index)
            temp_conditional_dis = []
            if (parent_list == parent_configuration_array[configuration_index]).all():
              temp_conditional_dis = conditional_table[j][k][configuration_index]
              # print(temp_conditional_dis).  #######Check!!!!
              break
          data[t,k] = np.random.choice(domain_of_var_vector, size=1, replace = True, p=temp_conditional_dis)[0]
        else:
          continue
  return data, conditional_table,edge_matrix,var_dic

def super_parent(data,T,N):
  datatime = np.arange(T-N)
  data_run = np.array(data[N:T,:])
  var_names=[]
  for i in range(N):
    var_names.append("$X^{}$".format(i))
  dataframe = pp.DataFrame(data_run,
                            datatime = {0:np.arange(len(data_run))},
                            var_names=var_names)
  #parcorr = ParCorr(significance='analytic')
  cmi_symb = CMIsymb(significance='shuffle_test')
  pcmci_cmi_symb = PCMCI(
      dataframe=dataframe,
      cond_ind_test=cmi_symb,
      verbosity=1)
  pcmci_cmi_symb.verbosity = 0
  results = pcmci_cmi_symb.run_pcmci(tau_min=1,tau_max=tau_max, pc_alpha=0.2, alpha_level=0.1)
  superset_bool=np.array(results["p_matrix"]<0.01, dtype=int)
  return superset_bool
def true_PE(n_regime,domain_of_var_vector,conditional_table,var_dic,edge_matrix,alpha_0,alpha_i):
  union_of_true_parent_set = np.zeros(shape=(N,N,tau_max+1)).astype(int)
  for i in range(N):
    list_union=np.zeros(shape=(N,tau_max+1))
    for j in range(n_regime):
      list_union+=edge_matrix[:,j,i,:] # i is the target value, j is the regime index
      union_of_true_parent_set[:,i,:]=np.array(list_union)

  for i in range(N):
    for j in range(N):
      for k in range(tau_max+1):
        if union_of_true_parent_set[j,i,k] >1:
          union_of_true_parent_set[j,i,k]=1

  union_of_true_parent_set_str = {}
  n_value = len(domain_of_var_vector)
  for i in range(N):
    union_of_true_parent_set_str [i] = []
  for h in range(N):
    for i in range(N):
      for j in range(len(union_of_true_parent_set[0,0,:])):
        if union_of_true_parent_set[i,h,j]==1:
          #  key='{}'.format(k)
          union_of_true_parent_set_str[h].append((i,-j))
  # print("var_dic[0][0]",var_dic[0][0])
  # print("var_dic[1][0]",var_dic[1][0])
  # print("union_of_true_parent_set_str[0]",union_of_true_parent_set_str[0])
  # print("conditional_table[0][0]",conditional_table[0][0])
  # print("conditional_table[1][0]",conditional_table[1][0])

  index_parent_configuration={}
  value_parent_configuration_permutation={}
  value_parent_permutation = {}
  value_var_dic = {}
  repeat_index_dic = {}
  expand_conditional_table={}
  for j in range(n_regime):
    index_parent_configuration[j]={}
    value_parent_configuration_permutation[j]=[]
    value_parent_permutation[j]={}
    value_var_dic[j]={}
    repeat_index_dic[j]={}
    expand_conditional_table[j]={}
  for j in range(n_regime):
    for i in range(N):
      index_parent_configuration[j][i]=[]
      value_parent_permutation[j][i]=[]
      value_var_dic[j][i]=[]
      repeat_index_dic[j][i]=[]
      expand_conditional_table[j][i]=[]
  for s in range(N):
    for j in range(n_regime):
      temp=[]
      for i in range(len(union_of_true_parent_set_str[s])):
        for k in range(len(var_dic[j][s])):
          if var_dic[j][s][k]==union_of_true_parent_set_str[s][i]:
            temp.append(i)
      index_parent_configuration[j][s]=temp
      value_parent_configuration_permutation[s]=list(itertools.product(domain_of_var_vector, repeat=len(union_of_true_parent_set_str[s])))
  # print(index_parent_configuration) # step 1: find the pair index (par,lag) in var_dic from union_of_true_parent_set_str
  # print(value_parent_configuration_permutation) # step 2: find out all the permutation of the union_of_true_parent_set_str, the order is the same as the order of the sub-timeseries

  for s in range(N):
    for j in range(n_regime):
      temp1=[]
      for l in range(len(value_parent_configuration_permutation[s])): # 8
        temp2=[]
        for i in range(len(index_parent_configuration[j][s])): # 2
          temp2.append(value_parent_configuration_permutation[s][l][index_parent_configuration[j][s][i]])
        temp1.append(temp2)
        # print(temp1)
      value_parent_permutation[j][s]=temp1

  # print(value_parent_permutation)# step 3: the long dic-dic array with values preparing for the expand conditional table

  for i in range(N):
    for j in range(n_regime):
      value_var_dic[j][i]=np.array(list(itertools.product(domain_of_var_vector, repeat=len(var_dic[j][i]))))
  # print(value_var_dic) # step 4: the short dic dic array with values preparing for how the expand conditional table repeats

  for j in range(n_regime):
    for i in range(N):
      temp=[]
      for s in range(len(value_parent_permutation[j][i])):
        for k in range(len(value_var_dic[j][i])):
          if (value_var_dic[j][i][k]==value_parent_permutation[j][i][s]).all():
            temp.append(k)
      repeat_index_dic[j][i]=temp


  for j in range(n_regime):
    for i in range(N):
      expand_conditional_table[j][i] = conditional_table[j][i][repeat_index_dic[j][i]] # step 5: according to the repeat index, expand the conditional table.
  # value_parent_configuration=index_parent_configuration
  # for i in range(N):
  #   for j in range(n_regime):
  #      for k in range(len(index_parent_configuration[j][i])):
  #       print(index_parent_configuration[j][i][k])
  #       print(index_parent_configuration[i][j][k])
  #       value_parent_configuration[i][j][k]=domain_of_var_vector[index_parent_configuration[i][j][k]]
  # alpha_0 = 0.95
  # alpha_i = np.arange(0,1.1,0.1)
  # true_PE = np.zeros(shape=(len(alpha_0),len(alpha_i)))
  n_parent=np.zeros(shape=(n_regime,N))
  n_parent_configuration=np.zeros(shape=(n_regime,N))
  repeat_time=np.zeros(shape=(n_regime,N))
  # var_dic
  # union_of_true_parent_set_str

  true_PE_dic={}
  true_PE_reverse_dic={}
  true_PE_sum_dic={}
  kl_dic={}

  max_r_dic={}
  max_r_reverse_dic={}
  c_dic={}
  c_reverse_dic={}
  for i in range(N):
    true_PE_dic[i]={}
    true_PE_reverse_dic[i]={}
    true_PE_sum_dic[i]={}
    kl_dic[i]={}
    max_r_dic[i]={}
    max_r_reverse_dic[i]={}
    c_dic[i]={}
    c_reverse_dic[i]={}
  for i in range(N):
    for j in range(len(expand_conditional_table[0][i])):
      true_PE_dic[i][j]=[]
      true_PE_reverse_dic[i][j]=[]
      true_PE_sum_dic[i][j]=[]
      kl_dic[i][j]=[]
      max_r_dic[i][j]=[]
      max_r_reverse_dic[i][j]=[]
      c_dic[i][j]=[]
      c_reverse_dic[i][j]=[]

  for i in range(N): # target variable
    for j in range(len(expand_conditional_table[0][i])): #
      temp=0
      temp2=0 # the row number of expand_conditional_table; that is, the number of sub-series
      temp3=0
      temp_list=[]
      temp2_list=[]
      for k in range(len(expand_conditional_table[0][i][j])):
        for p in range(10,11):
          temp+=expand_conditional_table[0][i][j][k]/((1-alpha_0*alpha_i)+alpha_0*alpha_i*(expand_conditional_table[1][i][j][k]/expand_conditional_table[0][i][j][k]))
          temp2+=expand_conditional_table[1][i][j][k]/((1-alpha_0*alpha_i)+alpha_0*alpha_i*(expand_conditional_table[0][i][j][k]/expand_conditional_table[1][i][j][k]))
          temp3+=expand_conditional_table[0][i][j][k]*np.log(expand_conditional_table[0][i][j][k]/expand_conditional_table[1][i][j][k])
          temp_list.append(expand_conditional_table[0][i][j][k]/((1-alpha_0*alpha_i)+alpha_0*alpha_i*(expand_conditional_table[1][i][j][k]/expand_conditional_table[0][i][j][k])))
          temp2_list.append(expand_conditional_table[1][i][j][k]/((1-alpha_0*alpha_i)+alpha_0*alpha_i*(expand_conditional_table[0][i][j][k]/expand_conditional_table[1][i][j][k])))
      max_r = max(temp_list)
      max_r_reverse = max(temp2_list)
      # c =max(temp_list)**2/(4*(n**(1/6)))+(1-alpha_0*alpha_i)**2*max(temp_list)**4/(16*(n**(1/6)))+(alpha_0*alpha_i)**2*max(temp_list)**4/(16*(n**(1/6)))
      # c_reverse = max(temp2_list)**2/(4*(n**(1/6)))+(1-alpha_0*alpha_i)**2*max(temp2_list)**4/(16*(n**(1/6)))+(alpha_0*alpha_i)**2*max(temp2_list)**4/(16*(n**(1/6)))
      c =max(temp_list)**2/(4*n)+(1-alpha_0*alpha_i)**2*max(temp_list)**4/(16*n)+(alpha_0*alpha_i)**2*max(temp_list)**4/(16*n)
      c_reverse = max(temp2_list)**2/(4*n)+(1-alpha_0*alpha_i)**2*max(temp2_list)**4/(16*n)+(alpha_0*alpha_i)**2*max(temp2_list)**4/(16*n)

      kl_dic[i][j]=temp3
      true_PE_dic[i][j] =1/2*temp-1/2
      true_PE_reverse_dic[i][j]=1/2*temp2-1/2
      true_PE_sum_dic[i][j]=true_PE_dic[i][j]+true_PE_reverse_dic[i][j]
      max_r_dic[i][j]=max_r
      max_r_reverse_dic[i][j]=max_r_reverse
      c_dic[i][j]=c
      c_reverse_dic[i][j]= c_reverse
      # true_PE_dic[i][j] =1/2*temp-1/2
  # print(union_of_true_parent_set_str)  # the order is, first
  return true_PE_dic,kl_dic,union_of_true_parent_set,true_PE_reverse_dic,true_PE_sum_dic,expand_conditional_table,max_r_dic,max_r_reverse_dic,c_dic,c_reverse_dic

def KL(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))

def super_parent(data,T,N,tau_max):
  datatime = np.arange(T-N)
  data_run = np.array(data[N:T,:])
  var_names=[]
  for i in range(N):
    var_names.append("$X^{}$".format(i))
  dataframe = pp.DataFrame(data_run,
                            datatime = {0:np.arange(len(data_run))},
                            var_names=var_names)
  #parcorr = ParCorr(significance='analytic')
  cmi_symb = CMIsymb(significance='shuffle_test')
  pcmci_cmi_symb = PCMCI(
      dataframe=dataframe,
      cond_ind_test=cmi_symb,
      verbosity=1)
  pcmci_cmi_symb.verbosity = 0
  results = pcmci_cmi_symb.run_pcmci(tau_min=1,tau_max=tau_max, pc_alpha=0.2, alpha_level=0.01)
  superset_bool=np.array(results["p_matrix"]<0.01, dtype=int)
  val_matrix=results['val_matrix']
  return superset_bool,val_matrix
def subsuperset_parent(data,T,N,tau_max,superset_dict_est):
  datatime = np.arange(T-N)
  data_run = np.array(data[N:T,:])
  var_names=[]
  for i in range(N):
    var_names.append("$X^{}$".format(i))

  dataframe = pp.DataFrame(data_run,
                            datatime = {0:np.arange(len(data_run))},
                            var_names=var_names)
  #parcorr = ParCorr(significance='analytic')
  cmi_symb = CMIsymb(significance='shuffle_test')
  pcmci_cmi_symb = PCMCI(
      dataframe=dataframe,
      cond_ind_test=cmi_symb,
      verbosity=1)
  pcmci_cmi_symb.verbosity = 0
  results = pcmci_cmi_symb.run_pcmci(selected_links=superset_dict_est,tau_max=tau_max, pc_alpha=0.2, alpha_level=0.1)
  superset_bool=np.array(results["p_matrix"]<0.01, dtype=int)
  return superset_bool
# superset_dict_est = {}
# for i in range(N):
#     superset_dict_est[i] = []
# for k in range(N):
#   for i in range(N):
#     for j in range(len(superset_bool[0,0,:])):
#       if superset_bool[i,k,j]==1:
#         #  key='{}'.format(k)
#         superset_dict_est[k].append((i,-j))
def dataset_with_cp(data,T,N,Tc_esti_interval_sort_top1):
  subdataset_with_cp = {}
  cp = np.zeros(N)
  cp_array=np.zeros((N,3)).astype(int)
  for tar in range(N):
    subdataset_with_cp[tar]={}
    for i in range(2):
      subdataset_with_cp[tar][i]=[]
  for tar in range(N):
    cp[tar] = (Tc_esti_interval_sort_top1[tar][0]+Tc_esti_interval_sort_top1[tar][1])/2
    cp_array[tar,0]=int(tau_max)
    cp_array[tar,2]=int(T)
    cp_array[tar,1]=int(np.floor(cp[tar]))
    for i in range(2):
      subdataset_with_cp[tar][i]=data[cp_array[tar,i]:cp_array[tar,i+1],:]
  return subdataset_with_cp

def al_dataset_with_cp(data,T,N,tau_max,Tc_esti_interval_sort_top1,k):
  tar=k
  subdataset_with_cp = {}
  # cp = np.zeros(N)
  cp_array=np.zeros(3).astype(int)
  for i in range(2):
    subdataset_with_cp[i]=[]
  cp = (Tc_esti_interval_sort_top1[0]+Tc_esti_interval_sort_top1[1])/2
  cp_array[0]=int(tau_max)
  cp_array[2]=int(T)
  cp_array[1]=int(np.floor(cp))
  for i in range(2):
    subdataset_with_cp[i]=data[cp_array[i]:cp_array[i+1],:]
  return subdataset_with_cp
def super_parent_with_cp(subdataset_with_cp,T,N,tau_max,superset_dict_est):
  subsuperset_parent_dic={}
  superset_bool_update={}
  superset_bool_update = np.zeros((N,N,tau_max+1)).astype(int)
  for tar in range(N):
    subsuperset_parent_dic[tar]={}
    for i in range(2): # 2 regime
     subsuperset_parent_dic[tar][i]=np.zeros((N,tau_max+1))
  for tar in range(N):
    for i in range(2): # 2 regime
      data_temp = subdataset_with_cp[tar][i]
      subsuperset_parent_dic[tar][i]=subsuperset_parent(data_temp,T,N,tau_max,superset_dict_est)[:,tar,:]
      superset_bool_update[:,tar,:]+=subsuperset_parent_dic[tar][i]
  superset_bool_update[superset_bool_update>=1]=1
  return superset_bool_update,subsuperset_parent_dic

def al_super_parent_with_cp(subdataset_with_cp,T,N,tau_max,superset_dict_est,k,n_regime=2):
  tar=k
  subsuperset_parent_dic={}
  superset_bool_update={}
  superset_bool_update = np.zeros((N,N,tau_max+1)).astype(int)
  superset_bool_BA_Tc = np.zeros((N,n_regime,N,tau_max+1)).astype(int)
  # for tar in range(N):
  #   subsuperset_parent_dic[tar]={}
  for i in range(2): # 2 regime
    subsuperset_parent_dic[i]=np.zeros((N,tau_max+1))
  # for tar in range(N):
  for i in range(2): # 2 regime
    data_temp = subdataset_with_cp[i]
    subsuperset_parent_dic[i]=subsuperset_parent(data_temp,T,N,tau_max,superset_dict_est)[:,tar,:]
    superset_bool_BA_Tc[:,i,tar,:]=subsuperset_parent_dic[i]
    superset_bool_update[:,tar,:]+=subsuperset_parent_dic[i]
  superset_bool_update[superset_bool_update>=1]=1
  return superset_bool_update,subsuperset_parent_dic,superset_bool_BA_Tc
def al_specific_configuration_data_generator(T,superset_bool_all,data,domain_of_var_vector,N,tau_max,k): # k=0 means the target variable is X_1
  # T=np.shape(data)[0]
  # print(T)
  superset_bool=superset_bool_all[:,k,:]
  n_value = len(domain_of_var_vector)
  superset_dict_est={}
  for h in range(N):
    superset_dict_est[h]=[]
  for h in range(N):
    for i in range(N):
      for j in range(len(superset_bool_all[0,0,:])):
        if superset_bool_all[i,h,j]==1:
        #  key='{}'.format(k)
          superset_dict_est[h].append((i,-j))
  # print(superset_dict_est)
  # if k==0:
  combine=np.zeros(np.power(n_value,sum(superset_bool)))
  # print(superset_bool[:,k,:])
  count_1=np.zeros(np.power(n_value,sum(superset_bool)))
  combine_len=len(combine)
  # print(combine_len)
  # print(combine_len)
  lst = list(itertools.product(domain_of_var_vector, repeat=sum(superset_bool)))
  count_index_1 = {}
  all_configuration = {}
  for g in range(combine_len):
    count_index_1[g] = []
    all_configuration[g] = []
  for g in range(combine_len):
    for i in range(tau_max,T):
      parent_index=np.zeros(sum(superset_bool))
      for j in range(len(superset_dict_est[k])): #the order of superset_dict_est is: first parent variable index, then time lag.
        if data[i+superset_dict_est[k][j][1],superset_dict_est[k][j][0]]==lst[g][j]:
          parent_index[j]=1
      # print(parent_index)
      if sum(parent_index)==sum(superset_bool):
        count_1[g]=count_1[g]+1
        count_index_1[g].append(i)
        # all_configuration[g].append(data[i,:].tolist())
        all_configuration[g].append(data[i,k].tolist())
  return count_1,count_index_1,all_configuration,superset_dict_est

def CP_algorithm(data,T,N,s_bound,n,domain_of_var_vector,tau_max,edge_matrix=None, cv_flag=1,Topk=None,alpha_i=1,alpha_0=0.5,k_dim_condi=1,n_regime=2):
  st = time.time()
  alpha=1-alpha_0
  whole_data=data
  score1_dic={}
#   score2_dic={}
#   combined_score_dic={}
  n_value = len(domain_of_var_vector)


  max_value_score1_index = np.zeros(N)
#   max_value_score2_index = np.zeros(N)
#   max_value_combined_score_index = np.zeros(N)

  max_value_score1_array = {}
#   max_value_score2_array= {}
#   max_value_combined_score_array = {}

  max_value_score1_dic={}
#   max_value_score2_dic={}
#   max_value_combined_score_dic={}

  Tc_score1_dic={}
#   Tc_score2_dic={}
#   Tc_combined_score_dic={}

  whole_data_int=np.array(data).astype(int)
  superset_bool,val_matrix = super_parent(whole_data_int,T,N,tau_max)

    ######################Topk Operation Starts##########################
    ######################Topk Operation Starts##########################
    ######################Topk Operation Starts##########################
  # Get the indices of the top-k values along the first dimension (D1)

  if Topk!=None:
    # Initialize a matrix with the same shape as val_matrix, filled with zeros
    top_k_matrix = np.zeros_like(val_matrix)
    val_matrix_significant = np.zeros_like(val_matrix)
    for col in range(val_matrix.shape[1]):
        val_matrix_significant[:,col,:] = val_matrix[:,col,:]*superset_bool[:,col,:]
    # Iterate over the third dimension (columns)
    for col in range(val_matrix_significant.shape[1]):  # Iterate over the third dimension
        # Get the top-k indices for the current column
        top_k_indices = np.argsort(val_matrix_significant[:,col,:], axis=None)  # Sort along the first dimension (depth)
        top_k_indices_insignificant= np.argsort(val_matrix[:,col,:], axis=None) 
        for i in range(Topk):
            i=i+1
            i=int(i)
            max_index = top_k_indices[-i]  # Get the index of the second maximum value
            max_location = np.unravel_index(max_index, val_matrix_significant[:,col,:].shape)  # Convert to multi-dimensional index
            
            max_top1_index=top_k_indices_insignificant[-1]
            max_top1_location=np.unravel_index(max_top1_index, val_matrix_significant[:,col,:].shape)
        # Set the corresponding entries in top_k_matrix to 1 for the top-k indices
        # Iterate over the top-k indices for the current column
            if superset_bool[max_location[0],col,max_location[1]]==1:
                top_k_matrix[max_location[0],col,max_location[1]] = 1  # Set the entire row for the top-k depth to 1

    # print("Topk",Topk+1)
    for i in range(val_matrix.shape[1]):
            # print("val_matrix:\n", val_matrix[:,i,:])
            # print("val_matrix_significant:\n",val_matrix_significant[:,i,:])
            # print("top_k_matrix:\n", top_k_matrix[:,i,:])
            # print("superset_bool_before",'var',i,superset_bool[:,i,:])
            superset_bool[:,i,:]=top_k_matrix.astype(int)[:,i,:]
            # print("superset_bool_after",'var',i,superset_bool[:,i,:])

    ######################Topk Operation Ends##########################
    ######################Topk Operation Ends##########################
    ######################Topk Operation Ends##########################
  n_value = len(domain_of_var_vector)
  split_sample = np.zeros(N)
  superset_bool_BA_Tc=np.zeros((N,n_regime,N,tau_max+1))

  max_value_score1 = np.zeros((s_bound,N))
#     max_value_score2 = np.zeros((s_bound,N))
#     max_value_combined_score = np.zeros((s_bound,N))

  Tc_esti_interval_sort_top1_dic={}

  if edge_matrix is not None:
    TP_array={}
    FN_array={}
    TN_array={}
    FP_array={}
    total_est_edge={}
    precision={}
    recall={}
    F_1={}

    max_sample_subtimeseries={}
    average_sample_subtimeseries={}


    TP_regime_array={}
    FN_regime_array={}
    TN_regime_array={}
    FP_regime_array={}
    total_est_regime_edge={}

    precision_regime={}
    recall_regime={}
    F_1_regime={}
  
    green_flag=np.zeros(N)
    superset_dict_est_final={}
    for i in range(N):
        superset_dict_est_final[i] = []
    for tar in range(N):
      TP_regime_array[tar]={}
      FN_regime_array[tar]={}
      TN_regime_array[tar]={}
      FP_regime_array[tar]={}
      total_est_regime_edge[tar]={}

      precision_regime[tar]={}
      recall_regime[tar]={}
      F_1_regime[tar]={}

      for regime in range(n_regime):

        superset_bool_BA_Tc[:,regime,:,:]=superset_bool

        TP_regime_array[tar][regime]=[]
        FN_regime_array[tar][regime]=[]
        TN_regime_array[tar][regime]=[]
        FP_regime_array[tar][regime]=[]
        total_est_regime_edge[tar][regime]=[]

        precision_regime[tar][regime]=[]
        recall_regime[tar][regime]=[]
        F_1_regime[tar][regime]=[]

      Tc_esti_interval_sort_top1_dic[tar]=[]

      TP_array[tar]=[]
      FN_array[tar]=[]
      TN_array[tar]=[]
      FP_array[tar]=[]
      total_est_edge[tar]=[]

      precision[tar]=[]
      recall[tar]=[]
      F_1[tar]=[]

      max_sample_subtimeseries[tar]=[]
      average_sample_subtimeseries[tar]=[]

    for tar in range(N):
      TP=0
      FN=0
      TN=0
      FP=0
      for r in range(n_regime):
        TP_regime=0
        FN_regime=0
        TN_regime=0
        FP_regime=0
        for k in range(N): # parent index
          for j in range(tau_max+1):
            if (edge_matrix[k,r,tar,j]==1) & (edge_matrix[k,r,tar,j] == superset_bool_BA_Tc[k,r,tar,j]):
              TP+=1
              TP_regime+=1
            if (edge_matrix[k,r,tar,j]==1) & (edge_matrix[k,r,tar,j] != superset_bool_BA_Tc[k,r,tar,j]):
              FN+=1
              FN_regime+=1
            if (edge_matrix[k,r,tar,j]==0) & (edge_matrix[k,r,tar,j] == superset_bool_BA_Tc[k,r,tar,j]):
              TN+=1
              TN_regime+=1
            if (edge_matrix[k,r,tar,j]==0) & (edge_matrix[k,r,tar,j] != superset_bool_BA_Tc[k,r,tar,j]):
              FP+=1
              FP_regime+=1
        # print(TP_regime_array)
        TP_regime_array[tar][r].append(TP_regime)
        FN_regime_array[tar][r].append(FN_regime)
        TN_regime_array[tar][r].append(TN_regime)
        FP_regime_array[tar][r].append(FP_regime)

        total_est_regime_edge[tar][r].append(TP_regime+FP_regime)

        precision_regime[tar][r].append(divide(TP_regime,TP_regime+FP_regime))
        recall_regime[tar][r].append(divide(TP_regime,TP_regime+FN_regime))
        F_1_regime[tar][r].append(divide(TP_regime,TP_regime+1/2*(FN_regime+FP_regime)))

      TP_array[tar].append(TP)
      FN_array[tar].append(FN)
      TN_array[tar].append(TN)
      FP_array[tar].append(FP)

      total_edge = TP+FP
      total_est_edge[tar].append(total_edge)

      precision[tar].append(divide(TP,TP+FP))
      recall[tar].append(divide(TP,TP+FN))
      F_1[tar].append(divide(TP,TP+1/2*(FP+FN)))

    for tar in range(N):
      superset_bool_option = superset_bool
      max_value_score1[-1,tar]=-0.1
      s=0
    #   print("#################var##############",tar)
      superset_bool_pre=superset_bool_option+1
      green_flag[tar]=0
      max_sample_subtimeseries[tar].append(int(0))
      average_sample_subtimeseries[tar].append(int(0))
      # while (max_value_score1[s,tar]>max_value_score1[s-1,tar]):
      while(sum(superset_bool_pre[:,tar,:]!=superset_bool_option[:,tar,:])!=0 or T/2**sum(superset_bool_option[:,tar,:])<2*n):
        if sum(superset_bool_pre[:,tar,:]!=superset_bool_option[:,tar,:])==0:
          while T/2**sum(superset_bool_option[:,tar,:])<2*n:
            val_matrix_temp = np.multiply(val_matrix[:,tar,:],superset_bool_option[:,tar,:])
            val_matrix_temp[val_matrix_temp==0]=np.inf
            p_matrix_temp = np.multiply(1-np.matrix(val_matrix_temp==val_matrix_temp.min()),superset_bool_option[:,tar,:])
            superset_bool_option[:,tar,:]=p_matrix_temp
            specific=al_specific_configuration_data_generator(T,superset_bool_option,data,domain_of_var_vector,N,tau_max,k=tar)
            green_flag[tar] = 1 # green_flag =1 means if the average length of sub time series is smaller than 2*n, then we need to delete some edges with the smallest statistics.
        s=s+1 #s=1, 2, 3
        if s>=s_bound:
          break
        split_sample[tar] = np.power(n_value,sum(superset_bool_option[:,tar,:]))
        split_sample = np.array(split_sample).astype(int)
        Tc_esti_interval = np.zeros((max(split_sample),2))
        Tc_esti_interval_peak = np.zeros((max(split_sample),2))


        for i in range(split_sample[tar]):

          score1_dic[i]=[]
          # score2_dic[i]=[]
          # combined_score_dic[i]=[]
          max_value_score1_dic[i]={}
          # max_value_score2_dic[i]={}
          # max_value_combined_score_dic[i]={}
          Tc_score1_dic[i]={}
          # Tc_score2_dic[i]={}
          # Tc_combined_score_dic[i]={}
        specific=al_specific_configuration_data_generator(T,superset_bool_option,data,domain_of_var_vector,N,tau_max,k=tar)
        max_sample_subtimeseries[tar].append(int(max(specific[0])))
        average_sample_subtimeseries[tar].append(np.average(specific[0]))
        # print(specific[0])
        # print(specific[1])
        # print(specific[2])
        while (max(specific[0])<=2*n):
          val_matrix_temp = np.multiply(val_matrix[:,tar,:],superset_bool_option[:,tar,:])
          val_matrix_temp[val_matrix_temp==0]=np.inf
          p_matrix_temp = np.multiply(1-np.matrix(val_matrix_temp==val_matrix_temp.min()),superset_bool_option[:,tar,:])
          superset_bool_option[:,tar,:]=p_matrix_temp
          specific=al_specific_configuration_data_generator(T,superset_bool_option,data,domain_of_var_vector,N,tau_max,k=tar)
        #   print(specific[0])
        #   print(specific[1])
        #   print(specific[2])
          green_flag[tar] = 2
       ################################################################# 
       ################################################################# 
       ################################################################# 
       ################################################################# 
        superset_dict_est=specific[3]
        for i in range(len(specific[0])):
          if len(specific[2][i])<2*n:
            continue
          data1= np.array(specific[2][i])
          y= data1.transpose()
        #   print("y.shape",y.shape)
          # Perform change detection in both directions
          if cv_flag==1:
            score1, sigma_track1, lambda_track1 = change_detection(y, n, k_dim_condi, alpha, 5)
            reversed_y = y[::-1]
          #   score2, sigma_track2, lambda_track2 = change_detection(reversed_y, n, k_dim_condi, alpha, 5)
          else:
            score1= change_detection_fix(y, n,k_dim_condi, alpha, 5)
            reversed_y = y[::-1]
          #   score2 = change_detection_fix(reversed_y, n, k_dim_condi, alpha, 5)
    
        #   print(score1)

          # score2 = score2[::-1]
          score1_dic[i]=np.array(score1)
          # score2_dic[i]=np.array(score2)
          # combined_score_dic[i] = score1_dic[i] + score2_dic[i]
          # combined_score_dic[i]=np.array(combined_score_dic[i])
          # print(combined_score.shape)
          # print(combined_score)
          if len(score1_dic[i])==0:
              max_value_score1_dic[i]=nan
              Tc_score1_dic[i]=nan
          else:
              max_value_score1_dic[i]=max(score1_dic[i])
              Tc_score1_dic[i]=np.where(score1_dic[i]==np.nanmax(score1_dic[i]))[0][0]
          # if len(score2_dic[i])==0:
          #     max_value_score2_dic[i]=nan
          #     Tc_score2_dic[i]=nan
          # else:
          #     max_value_score2_dic[i]=max(score2_dic[i])
          #     Tc_score2_dic[i]=np.where(score2_dic[i]==np.nanmax(score2_dic[i]))[0][0]
          # if len(combined_score_dic[i])==0:
          #     max_value_combined_score_dic[i]=nan
          #     Tc_combined_score_dic[i]=nan
          # else:
          #     max_value_combined_score_dic[i]=max(combined_score_dic[i])
          #     Tc_combined_score_dic[i]=np.where(combined_score_dic[i]==np.nanmax(combined_score_dic[i]))[0][0]

          if len(score1_dic[i])==0: ######### Only base on score1_dic
            continue
          esti_Tc_sub =np.where(score1_dic[i]==np.nanmax(score1_dic[i]))[0][0]+n-1

          Tc_esti_interval[i][0]=specific[1][i][esti_Tc_sub]
          Tc_esti_interval[i][1]=specific[1][i][esti_Tc_sub+1]
        #   print("Tc_esti_interval[i][0]",Tc_esti_interval[i][0],"Tc_esti_interval[i][1]",Tc_esti_interval[i][1])


        temp=[]
        for j in range(split_sample[tar]):
          temp.append(max_value_score1_dic[j])
        max_value_score1_array = np.array(temp)

    #     temp=[]
    #     for j in range(split_sample[tar]):
    #       temp.append(max_value_score2_dic[j])
    #     max_value_score2_array = np.array(temp)

    #     temp=[]
    #     for j in range(split_sample[tar]):
    #       temp.append(max_value_combined_score_dic[j])
    #     max_value_combined_score_array = np.array(temp)

        for j in range(len(max_value_score1_array)):
          if max_value_score1_array[j]=={}:
            max_value_score1_array[j]=0
          # if max_value_score2_array[j]=={}:
          #   max_value_score2_array[j]=0
          # if max_value_combined_score_array[j]=={}:
          #   max_value_combined_score_array[j]=0


        max_value_score1_index = np.where(max_value_score1_array==np.nanmax(max_value_score1_array))[0][0]
    #     max_value_score2_index = np.where(max_value_score2_array==np.nanmax(max_value_score2_array))[0][0]
    #     max_value_combined_score_index= np.where(max_value_combined_score_array==np.nanmax(max_value_combined_score_array))[0][0]

        max_value_score1[s,tar]=np.nanmax(max_value_score1_array)
    #     max_value_score2[s,tar]=np.nanmax(max_value_score2_array)
    #     max_value_combined_score[s,tar]=np.nanmax(max_value_combined_score_array)

        max_value_score1_array_temp=np.array(max_value_score1_array).astype(float)
        max_value_score1_array_temp=np.nan_to_num(max_value_score1_array_temp,nan=0)

      
        Tc_esti_interval_sort = np.sort(Tc_esti_interval[max_value_score1_array_temp.argsort()])[::-1]
        # print("Tc_esti_interval_sort",Tc_esti_interval_sort)
        # print("max_value_score1_array_temp",max_value_score1_array_temp)
        # print("Tc_esti_interval",Tc_esti_interval)
        Tc_esti_interval_sort_top1 = Tc_esti_interval_sort[0]

        # print("s",s)
        # print("Tc_esti_interval_sort_top1",Tc_esti_interval_sort_top1)
        Tc_esti_interval_sort_top1_dic[tar].append(Tc_esti_interval_sort_top1)
        et = time.time()
        subdataset_with_cp =  al_dataset_with_cp(whole_data_int,T,N,tau_max,Tc_esti_interval_sort_top1,tar)
        superset_bool_update,subsuperset_parent_dic,superset_bool_BA_Tc = al_super_parent_with_cp(subdataset_with_cp,T,N,tau_max,superset_dict_est,tar)
        superset_bool_pre = superset_bool_option
        superset_bool_option = superset_bool_update

        TP=0
        FN=0
        TN=0
        FP=0
        for r in range(n_regime):
          TP_regime=0
          FN_regime=0
          TN_regime=0
          FP_regime=0
          for k in range(N): # parent index
            for j in range(tau_max+1):
              if (edge_matrix[k,r,tar,j]==1) & (edge_matrix[k,r,tar,j] == superset_bool_BA_Tc[k,r,tar,j]):
                TP+=1
                TP_regime+=1
              if (edge_matrix[k,r,tar,j]==1) & (edge_matrix[k,r,tar,j] != superset_bool_BA_Tc[k,r,tar,j]):
                FN+=1
                FN_regime+=1
              if (edge_matrix[k,r,tar,j]==0) & (edge_matrix[k,r,tar,j] == superset_bool_BA_Tc[k,r,tar,j]):
                TN+=1
                TN_regime+=1
              if (edge_matrix[k,r,tar,j]==0) & (edge_matrix[k,r,tar,j] != superset_bool_BA_Tc[k,r,tar,j]):
                FP+=1
                FP_regime+=1
        #   print(TP_regime_array)
          TP_regime_array[tar][r].append(TP_regime)
          FN_regime_array[tar][r].append(FN_regime)
          TN_regime_array[tar][r].append(TN_regime)
          FP_regime_array[tar][r].append(FP_regime)

          total_est_regime_edge[tar][r].append(TP_regime+FP_regime)

          precision_regime[tar][r].append(divide(TP_regime,TP_regime+FP_regime))
          recall_regime[tar][r].append(divide(TP_regime,TP_regime+FN_regime))
          F_1_regime[tar][r].append(divide(TP_regime,TP_regime+1/2*(FN_regime+FP_regime)))

        TP_array[tar].append(TP)
        FN_array[tar].append(FN)
        TN_array[tar].append(TN)
        FP_array[tar].append(FP)

        total_edge = TP+FP
        total_est_edge[tar].append(total_edge)

        precision[tar].append(divide(TP,TP+FP))
        recall[tar].append(divide(TP,TP+FN))
        F_1[tar].append(divide(TP,TP+1/2*(FP+FN)))
        # print("superset_bool_BA_Tc_regime0_var0",superset_bool_BA_Tc[:,0,0,:])
        # print("superset_bool_BA_Tc_regime1_var0",superset_bool_BA_Tc[:,1,0,:])
        # print("superset_bool_BA_Tc_regime0_var1",superset_bool_BA_Tc[:,0,1,:])
        # print("superset_bool_BA_Tc_regime1_var1",superset_bool_BA_Tc[:,1,1,:])
        # print("superset_bool_pre", superset_bool_pre )
        # print("superset_bool_option",superset_bool_option)
        # print("s",s)

      if s!=0:
        for i in range(N):
          for j in range(len(superset_bool_update[0,tar,:])):
            if superset_bool_update[i,tar,j]==1:
            #  key='{}'.format(k)
              superset_dict_est_final[tar].append((i,-j))
    elapsed_time = et - st
    return Tc_esti_interval_sort_top1_dic,s,max_value_score1,superset_dict_est_final,TP_array,FN_array,TN_array,FP_array,precision,recall,F_1,total_est_edge,TP_regime_array,FN_regime_array,TN_regime_array,FP_regime_array,total_est_regime_edge,precision_regime,recall_regime,F_1_regime,superset_bool,elapsed_time,green_flag,max_sample_subtimeseries,average_sample_subtimeseries

  else: ### if edge_matrix=None
    total_est_edge={}
    max_sample_subtimeseries={}
    average_sample_subtimeseries={}
    total_est_regime_edge={}
    green_flag=np.zeros(N)
    superset_dict_est_final={}
    for i in range(N):
        superset_dict_est_final[i] = []
    for tar in range(N):
      total_est_regime_edge[tar]={}
      Tc_esti_interval_sort_top1_dic[tar]=[]
      total_est_edge[tar]=[]
      max_sample_subtimeseries[tar]=[]
      average_sample_subtimeseries[tar]=[]

      for regime in range(n_regime):

        superset_bool_BA_Tc[:,regime,:,:]=superset_bool
        total_est_regime_edge[tar][regime]=[]

    for tar in range(N):
      total_est_edge[tar].append(np.sum(superset_bool_BA_Tc[:,:,tar,:]))
      for r in range(n_regime):
        total_est_regime_edge[tar][r]=np.sum(superset_bool_BA_Tc[:,r,tar,:])


    for tar in range(N):
      superset_bool_option = superset_bool
      max_value_score1[-1,tar]=-0.1
      s=0
    #   print("#################var##############",tar)
      superset_bool_pre=superset_bool_option+1
      green_flag[tar]=0
      max_sample_subtimeseries[tar].append(int(0))
      average_sample_subtimeseries[tar].append(int(0))
      # while (max_value_score1[s,tar]>max_value_score1[s-1,tar]):
      while(sum(superset_bool_pre[:,tar,:]!=superset_bool_option[:,tar,:])!=0 or T/2**sum(superset_bool_option[:,tar,:])<2*n):
        if sum(superset_bool_pre[:,tar,:]!=superset_bool_option[:,tar,:])==0:
          while T/2**sum(superset_bool_option[:,tar,:])<2*n:
            val_matrix_temp = np.multiply(val_matrix[:,tar,:],superset_bool_option[:,tar,:])
            val_matrix_temp[val_matrix_temp==0]=np.inf
            p_matrix_temp = np.multiply(1-np.matrix(val_matrix_temp==val_matrix_temp.min()),superset_bool_option[:,tar,:])
            superset_bool_option[:,tar,:]=p_matrix_temp
            specific=al_specific_configuration_data_generator(T,superset_bool_option,data,domain_of_var_vector,N,tau_max,k=tar)
            green_flag[tar] = 1 # green_flag =1 means if the average length of sub time series is smaller than 2*n, then we need to delete some edges with the smallest statistics.
        s=s+1 #s=1, 2, 3
        if s>=s_bound:
          break
        split_sample[tar] = np.power(n_value,sum(superset_bool_option[:,tar,:]))
        split_sample = np.array(split_sample).astype(int)
        Tc_esti_interval = np.zeros((max(split_sample),2))
        Tc_esti_interval_peak = np.zeros((max(split_sample),2))


        for i in range(split_sample[tar]):

          score1_dic[i]=[]
          # score2_dic[i]=[]
          # combined_score_dic[i]=[]
          max_value_score1_dic[i]={}
          # max_value_score2_dic[i]={}
          # max_value_combined_score_dic[i]={}
          Tc_score1_dic[i]={}
          # Tc_score2_dic[i]={}
          # Tc_combined_score_dic[i]={}
        specific=al_specific_configuration_data_generator(T,superset_bool_option,data,domain_of_var_vector,N,tau_max,k=tar)
        max_sample_subtimeseries[tar].append(int(max(specific[0])))
        average_sample_subtimeseries[tar].append(np.average(specific[0]))
        # print(specific[0])
        # print(specific[1])
        # print(specific[2])
        while (max(specific[0])<=2*n):
          val_matrix_temp = np.multiply(val_matrix[:,tar,:],superset_bool_option[:,tar,:])
          val_matrix_temp[val_matrix_temp==0]=np.inf
          p_matrix_temp = np.multiply(1-np.matrix(val_matrix_temp==val_matrix_temp.min()),superset_bool_option[:,tar,:])
          superset_bool_option[:,tar,:]=p_matrix_temp
          specific=al_specific_configuration_data_generator(T,superset_bool_option,data,domain_of_var_vector,N,tau_max,k=tar)
        #   print(specific[0])
        #   print(specific[1])
        #   print(specific[2])
          green_flag[tar] = 2
       ################################################################# 
       ################################################################# 
       ################################################################# 
       ################################################################# 
        superset_dict_est=specific[3]
        for i in range(len(specific[0])):
          if len(specific[2][i])<2*n:
            continue
          data1= np.array(specific[2][i])
          y= data1.transpose()
        #   print("y.shape",y.shape)
          # Perform change detection in both directions
          if cv_flag==1:
            score1, sigma_track1, lambda_track1 = change_detection(y, n, k_dim_condi, alpha, 5)
            reversed_y = y[::-1]
          #   score2, sigma_track2, lambda_track2 = change_detection(reversed_y, n, k_dim_condi, alpha, 5)
          else:
            score1= change_detection_fix(y, n,k_dim_condi, alpha, 5)
            reversed_y = y[::-1]
          #   score2 = change_detection_fix(reversed_y, n, k_dim_condi, alpha, 5)
    
        #   print(score1)

          # score2 = score2[::-1]
          score1_dic[i]=np.array(score1)
          # score2_dic[i]=np.array(score2)
          # combined_score_dic[i] = score1_dic[i] + score2_dic[i]
          # combined_score_dic[i]=np.array(combined_score_dic[i])
          # print(combined_score.shape)
          # print(combined_score)
          if len(score1_dic[i])==0:
              max_value_score1_dic[i]=nan
              Tc_score1_dic[i]=nan
          else:
              max_value_score1_dic[i]=max(score1_dic[i])
              Tc_score1_dic[i]=np.where(score1_dic[i]==np.nanmax(score1_dic[i]))[0][0]
          # if len(score2_dic[i])==0:
          #     max_value_score2_dic[i]=nan
          #     Tc_score2_dic[i]=nan
          # else:
          #     max_value_score2_dic[i]=max(score2_dic[i])
          #     Tc_score2_dic[i]=np.where(score2_dic[i]==np.nanmax(score2_dic[i]))[0][0]
          # if len(combined_score_dic[i])==0:
          #     max_value_combined_score_dic[i]=nan
          #     Tc_combined_score_dic[i]=nan
          # else:
          #     max_value_combined_score_dic[i]=max(combined_score_dic[i])
          #     Tc_combined_score_dic[i]=np.where(combined_score_dic[i]==np.nanmax(combined_score_dic[i]))[0][0]

          if len(score1_dic[i])==0: ######### Only base on score1_dic
            continue
          esti_Tc_sub =np.where(score1_dic[i]==np.nanmax(score1_dic[i]))[0][0]+n-1

          Tc_esti_interval[i][0]=specific[1][i][esti_Tc_sub]
          Tc_esti_interval[i][1]=specific[1][i][esti_Tc_sub+1]
        #   print("Tc_esti_interval[i][0]",Tc_esti_interval[i][0],"Tc_esti_interval[i][1]",Tc_esti_interval[i][1])


        temp=[]
        for j in range(split_sample[tar]):
          temp.append(max_value_score1_dic[j])
        max_value_score1_array = np.array(temp)

    #     temp=[]
    #     for j in range(split_sample[tar]):
    #       temp.append(max_value_score2_dic[j])
    #     max_value_score2_array = np.array(temp)

    #     temp=[]
    #     for j in range(split_sample[tar]):
    #       temp.append(max_value_combined_score_dic[j])
    #     max_value_combined_score_array = np.array(temp)

        for j in range(len(max_value_score1_array)):
          if max_value_score1_array[j]=={}:
            max_value_score1_array[j]=0
          # if max_value_score2_array[j]=={}:
          #   max_value_score2_array[j]=0
          # if max_value_combined_score_array[j]=={}:
          #   max_value_combined_score_array[j]=0


        max_value_score1_index = np.where(max_value_score1_array==np.nanmax(max_value_score1_array))[0][0]
    #     max_value_score2_index = np.where(max_value_score2_array==np.nanmax(max_value_score2_array))[0][0]
    #     max_value_combined_score_index= np.where(max_value_combined_score_array==np.nanmax(max_value_combined_score_array))[0][0]

        max_value_score1[s,tar]=np.nanmax(max_value_score1_array)
    #     max_value_score2[s,tar]=np.nanmax(max_value_score2_array)
    #     max_value_combined_score[s,tar]=np.nanmax(max_value_combined_score_array)

        max_value_score1_array_temp=np.array(max_value_score1_array).astype(float)
        max_value_score1_array_temp=np.nan_to_num(max_value_score1_array_temp,nan=0)

      
        Tc_esti_interval_sort = np.sort(Tc_esti_interval[max_value_score1_array_temp.argsort()])[::-1]
        # print("Tc_esti_interval_sort",Tc_esti_interval_sort)
        # print("max_value_score1_array_temp",max_value_score1_array_temp)
        # print("Tc_esti_interval",Tc_esti_interval)
        Tc_esti_interval_sort_top1 = Tc_esti_interval_sort[0]

        # print("s",s)
        # print("Tc_esti_interval_sort_top1",Tc_esti_interval_sort_top1)
        Tc_esti_interval_sort_top1_dic[tar].append(Tc_esti_interval_sort_top1)
        et = time.time()
        subdataset_with_cp =  al_dataset_with_cp(whole_data_int,T,N,tau_max,Tc_esti_interval_sort_top1,tar)
        superset_bool_update,subsuperset_parent_dic,superset_bool_BA_Tc = al_super_parent_with_cp(subdataset_with_cp,T,N,tau_max,superset_dict_est,tar)
        superset_bool_pre = superset_bool_option
        superset_bool_option = superset_bool_update

        for tar in range(N):
          total_est_edge[tar].append(np.sum(superset_bool_BA_Tc[:,:,tar,:]))
        for r in range(n_regime):
          total_est_regime_edge[tar][r]=np.sum(superset_bool_BA_Tc[:,r,tar,:])

        # print("superset_bool_BA_Tc_regime0_var0",superset_bool_BA_Tc[:,0,0,:])
        # print("superset_bool_BA_Tc_regime1_var0",superset_bool_BA_Tc[:,1,0,:])
        # print("superset_bool_BA_Tc_regime0_var1",superset_bool_BA_Tc[:,0,1,:])
        # print("superset_bool_BA_Tc_regime1_var1",superset_bool_BA_Tc[:,1,1,:])
        # print("superset_bool_pre", superset_bool_pre )
        # print("superset_bool_option",superset_bool_option)
        # print("s",s)

      if s!=0:
        for i in range(N):
          for j in range(len(superset_bool_update[0,tar,:])):
            if superset_bool_update[i,tar,j]==1:
            #  key='{}'.format(k)
              superset_dict_est_final[tar].append((i,-j))
  # get the execution time
    elapsed_time = et - st
    return Tc_esti_interval_sort_top1_dic,s,max_value_score1,superset_dict_est_final,total_est_edge,total_est_regime_edge,superset_bool,elapsed_time,green_flag,max_sample_subtimeseries,average_sample_subtimeseries

def CP_algorithm_track_with_true_edge_matrix(data,T,N,s_bound,n,domain_of_var_vector,tau_max,edge_matrix, cv_flag,Topk=None,alpha_i=1,alpha_0=0.5,k_dim_condi=1,n_regime=2):
  st = time.time()
  alpha=1-alpha_0
  whole_data=data
  score1_dic={}
#   score2_dic={}
#   combined_score_dic={}
  n_value = len(domain_of_var_vector)


  max_value_score1_index = np.zeros(N)
#   max_value_score2_index = np.zeros(N)
#   max_value_combined_score_index = np.zeros(N)

  max_value_score1_array = {}
#   max_value_score2_array= {}
#   max_value_combined_score_array = {}

  max_value_score1_dic={}
#   max_value_score2_dic={}
#   max_value_combined_score_dic={}

  Tc_score1_dic={}
#   Tc_score2_dic={}
#   Tc_combined_score_dic={}

  whole_data_int=np.array(data).astype(int)
  superset_bool,val_matrix = super_parent(whole_data_int,T,N,tau_max)

    ######################Topk Operation Starts##########################
    ######################Topk Operation Starts##########################
    ######################Topk Operation Starts##########################
  # Get the indices of the top-k values along the first dimension (D1)

  if Topk!=None:
    # Initialize a matrix with the same shape as val_matrix, filled with zeros
    top_k_matrix = np.zeros_like(val_matrix)
    val_matrix_significant = np.zeros_like(val_matrix)
    for col in range(val_matrix.shape[1]):
        val_matrix_significant[:,col,:] = val_matrix[:,col,:]*superset_bool[:,col,:]
    # Iterate over the third dimension (columns)
    for col in range(val_matrix_significant.shape[1]):  # Iterate over the third dimension
        # Get the top-k indices for the current column
        top_k_indices = np.argsort(val_matrix_significant[:,col,:], axis=None)  # Sort along the first dimension (depth)
        top_k_indices_insignificant= np.argsort(val_matrix[:,col,:], axis=None) 
        for i in range(Topk):
            i=i+1
            i=int(i)
            max_index = top_k_indices[-i]  # Get the index of the second maximum value
            max_location = np.unravel_index(max_index, val_matrix_significant[:,col,:].shape)  # Convert to multi-dimensional index
            
            max_top1_index=top_k_indices_insignificant[-1]
            max_top1_location=np.unravel_index(max_top1_index, val_matrix_significant[:,col,:].shape)
        # Set the corresponding entries in top_k_matrix to 1 for the top-k indices
        # Iterate over the top-k indices for the current column
            if superset_bool[max_location[0],col,max_location[1]]==1:
                top_k_matrix[max_location[0],col,max_location[1]] = 1  # Set the entire row for the top-k depth to 1

    # print("Topk",Topk+1)
    for i in range(val_matrix.shape[1]):
            # print("val_matrix:\n", val_matrix[:,i,:])
            # print("val_matrix_significant:\n",val_matrix_significant[:,i,:])
            # print("top_k_matrix:\n", top_k_matrix[:,i,:])
            # print("superset_bool_before",'var',i,superset_bool[:,i,:])
            superset_bool[:,i,:]=top_k_matrix.astype(int)[:,i,:]
            # print("superset_bool_after",'var',i,superset_bool[:,i,:])

    ######################Topk Operation Ends##########################
    ######################Topk Operation Ends##########################
    ######################Topk Operation Ends##########################
  n_value = len(domain_of_var_vector)
  split_sample = np.zeros(N)
  superset_bool_BA_Tc=np.zeros((N,n_regime,N,tau_max+1))

  max_value_score1 = np.zeros((s_bound,N))
#   max_value_score2 = np.zeros((s_bound,N))
#   max_value_combined_score = np.zeros((s_bound,N))

  Tc_esti_interval_sort_top1_dic={}

  TP_array={}
  FN_array={}
  TN_array={}
  FP_array={}
  total_est_edge={}
  precision={}
  recall={}
  F_1={}

  max_sample_subtimeseries={}
  average_sample_subtimeseries={}


  TP_regime_array={}
  FN_regime_array={}
  TN_regime_array={}
  FP_regime_array={}
  total_est_regime_edge={}

  precision_regime={}
  recall_regime={}
  F_1_regime={}
  
  green_flag=np.zeros(N)
  superset_dict_est_final={}
  for i in range(N):
      superset_dict_est_final[i] = []
  for tar in range(N):
    TP_regime_array[tar]={}
    FN_regime_array[tar]={}
    TN_regime_array[tar]={}
    FP_regime_array[tar]={}
    total_est_regime_edge[tar]={}

    precision_regime[tar]={}
    recall_regime[tar]={}
    F_1_regime[tar]={}

    for regime in range(n_regime):

      superset_bool_BA_Tc[:,regime,:,:]=superset_bool

      TP_regime_array[tar][regime]=[]
      FN_regime_array[tar][regime]=[]
      TN_regime_array[tar][regime]=[]
      FP_regime_array[tar][regime]=[]
      total_est_regime_edge[tar][regime]=[]

      precision_regime[tar][regime]=[]
      recall_regime[tar][regime]=[]
      F_1_regime[tar][regime]=[]

    Tc_esti_interval_sort_top1_dic[tar]=[]

    TP_array[tar]=[]
    FN_array[tar]=[]
    TN_array[tar]=[]
    FP_array[tar]=[]
    total_est_edge[tar]=[]

    precision[tar]=[]
    recall[tar]=[]
    F_1[tar]=[]

    max_sample_subtimeseries[tar]=[]
    average_sample_subtimeseries[tar]=[]

  for tar in range(N):
    TP=0
    FN=0
    TN=0
    FP=0
    for r in range(n_regime):
      TP_regime=0
      FN_regime=0
      TN_regime=0
      FP_regime=0
      for k in range(N): # parent index
        for j in range(tau_max+1):
          if (edge_matrix[k,r,tar,j]==1) & (edge_matrix[k,r,tar,j] == superset_bool_BA_Tc[k,r,tar,j]):
            TP+=1
            TP_regime+=1
          if (edge_matrix[k,r,tar,j]==1) & (edge_matrix[k,r,tar,j] != superset_bool_BA_Tc[k,r,tar,j]):
            FN+=1
            FN_regime+=1
          if (edge_matrix[k,r,tar,j]==0) & (edge_matrix[k,r,tar,j] == superset_bool_BA_Tc[k,r,tar,j]):
            TN+=1
            TN_regime+=1
          if (edge_matrix[k,r,tar,j]==0) & (edge_matrix[k,r,tar,j] != superset_bool_BA_Tc[k,r,tar,j]):
            FP+=1
            FP_regime+=1
    #   print(TP_regime_array)
      TP_regime_array[tar][r].append(TP_regime)
      FN_regime_array[tar][r].append(FN_regime)
      TN_regime_array[tar][r].append(TN_regime)
      FP_regime_array[tar][r].append(FP_regime)

      total_est_regime_edge[tar][r].append(TP_regime+FP_regime)

      precision_regime[tar][r].append(divide(TP_regime,TP_regime+FP_regime))
      recall_regime[tar][r].append(divide(TP_regime,TP_regime+FN_regime))
      F_1_regime[tar][r].append(divide(TP_regime,TP_regime+1/2*(FN_regime+FP_regime)))

    TP_array[tar].append(TP)
    FN_array[tar].append(FN)
    TN_array[tar].append(TN)
    FP_array[tar].append(FP)

    total_edge = TP+FP
    total_est_edge[tar].append(total_edge)

    precision[tar].append(divide(TP,TP+FP))
    recall[tar].append(divide(TP,TP+FN))
    F_1[tar].append(divide(TP,TP+1/2*(FP+FN)))

  for tar in range(N):
    superset_bool_option = superset_bool
    max_value_score1[-1,tar]=-0.1
    s=0
    # print("#################var##############",tar)
    superset_bool_pre=superset_bool_option+1
    green_flag[tar]=0
    max_sample_subtimeseries[tar].append(int(0))
    average_sample_subtimeseries[tar].append(int(0))
    # while (max_value_score1[s,tar]>max_value_score1[s-1,tar]):
    while(sum(superset_bool_pre[:,tar,:]!=superset_bool_option[:,tar,:])!=0 or T/2**sum(superset_bool_option[:,tar,:])<2*n):
      if sum(superset_bool_pre[:,tar,:]!=superset_bool_option[:,tar,:])==0:
        while T/2**sum(superset_bool_option[:,tar,:])<2*n:
          val_matrix_temp = np.multiply(val_matrix[:,tar,:],superset_bool_option[:,tar,:])
          val_matrix_temp[val_matrix_temp==0]=np.inf
          p_matrix_temp = np.multiply(1-np.matrix(val_matrix_temp==val_matrix_temp.min()),superset_bool_option[:,tar,:])
          superset_bool_option[:,tar,:]=p_matrix_temp
          specific=al_specific_configuration_data_generator(T,superset_bool_option,data,domain_of_var_vector,N,tau_max,k=tar)
          green_flag[tar] = 1 # green_flag =1 means if the average length of sub time series is smaller than 2*n, then we need to delete some edges with the smallest statistics.
      s=s+1 #s=1, 2, 3
      if s>=s_bound:
        break
      split_sample[tar] = np.power(n_value,sum(superset_bool_option[:,tar,:]))
      split_sample = np.array(split_sample).astype(int)
      Tc_esti_interval = np.zeros((max(split_sample),2))
      Tc_esti_interval_peak = np.zeros((max(split_sample),2))


      for i in range(split_sample[tar]):

        score1_dic[i]=[]
        # score2_dic[i]=[]
        # combined_score_dic[i]=[]
        max_value_score1_dic[i]={}
        # max_value_score2_dic[i]={}
        # max_value_combined_score_dic[i]={}
        Tc_score1_dic[i]={}
        # Tc_score2_dic[i]={}
        # Tc_combined_score_dic[i]={}
      specific=al_specific_configuration_data_generator(T,superset_bool_option,data,domain_of_var_vector,N,tau_max,k=tar)
      max_sample_subtimeseries[tar].append(int(max(specific[0])))
      average_sample_subtimeseries[tar].append(np.average(specific[0]))
    #   print(specific[0])
    #   print(specific[1])
    #   print(specific[2])
      while (max(specific[0])<=2*n):
        val_matrix_temp = np.multiply(val_matrix[:,tar,:],superset_bool_option[:,tar,:])
        val_matrix_temp[val_matrix_temp==0]=np.inf
        p_matrix_temp = np.multiply(1-np.matrix(val_matrix_temp==val_matrix_temp.min()),superset_bool_option[:,tar,:])
        superset_bool_option[:,tar,:]=p_matrix_temp
        specific=al_specific_configuration_data_generator(T,superset_bool_option,data,domain_of_var_vector,N,tau_max,k=tar)
        # print(specific[0])
        # print(specific[1])
        # print(specific[2])
        green_flag[tar] = 2
       ################################################################# 
       ################################################################# 
       ################################################################# 
       ################################################################# 
      superset_dict_est=specific[3]
      for i in range(len(specific[0])):
        if len(specific[2][i])<2*n:
          continue
        data1= np.array(specific[2][i])
        y= data1.transpose()
        # print("y.shape",y.shape)
        # Perform change detection in both directions
        if cv_flag==1:
          score1, sigma_track1, lambda_track1 = change_detection(y, n, k_dim_condi, alpha, 5)
          reversed_y = y[::-1]
        #   score2, sigma_track2, lambda_track2 = change_detection(reversed_y, n, k_dim_condi, alpha, 5)
        else:
          score1= change_detection_fix(y, n,k_dim_condi, alpha, 5)
          reversed_y = y[::-1]
        #   score2 = change_detection_fix(reversed_y, n, k_dim_condi, alpha, 5)
    
        # print(score1)

        # score2 = score2[::-1]
        score1_dic[i]=np.array(score1)
        # score2_dic[i]=np.array(score2)
        # combined_score_dic[i] = score1_dic[i] + score2_dic[i]
        # combined_score_dic[i]=np.array(combined_score_dic[i])
        # print(combined_score.shape)
        # print(combined_score)
        if len(score1_dic[i])==0:
            max_value_score1_dic[i]=nan
            Tc_score1_dic[i]=nan
        else:
            max_value_score1_dic[i]=max(score1_dic[i])
            Tc_score1_dic[i]=np.where(score1_dic[i]==np.nanmax(score1_dic[i]))[0][0]
        # if len(score2_dic[i])==0:
        #     max_value_score2_dic[i]=nan
        #     Tc_score2_dic[i]=nan
        # else:
        #     max_value_score2_dic[i]=max(score2_dic[i])
        #     Tc_score2_dic[i]=np.where(score2_dic[i]==np.nanmax(score2_dic[i]))[0][0]
        # if len(combined_score_dic[i])==0:
        #     max_value_combined_score_dic[i]=nan
        #     Tc_combined_score_dic[i]=nan
        # else:
        #     max_value_combined_score_dic[i]=max(combined_score_dic[i])
        #     Tc_combined_score_dic[i]=np.where(combined_score_dic[i]==np.nanmax(combined_score_dic[i]))[0][0]

        if len(score1_dic[i])==0: ######### Only base on score1_dic
          continue
        esti_Tc_sub =np.where(score1_dic[i]==np.nanmax(score1_dic[i]))[0][0]+n-1

        Tc_esti_interval[i][0]=specific[1][i][esti_Tc_sub]
        Tc_esti_interval[i][1]=specific[1][i][esti_Tc_sub+1]
        # print("Tc_esti_interval[i][0]",Tc_esti_interval[i][0],"Tc_esti_interval[i][1]",Tc_esti_interval[i][1])


      temp=[]
      for j in range(split_sample[tar]):
        temp.append(max_value_score1_dic[j])
      max_value_score1_array = np.array(temp)

    #   temp=[]
    #   for j in range(split_sample[tar]):
    #     temp.append(max_value_score2_dic[j])
    #   max_value_score2_array = np.array(temp)

    #   temp=[]
    #   for j in range(split_sample[tar]):
    #     temp.append(max_value_combined_score_dic[j])
    #   max_value_combined_score_array = np.array(temp)

      for j in range(len(max_value_score1_array)):
        if max_value_score1_array[j]=={}:
          max_value_score1_array[j]=0
        # if max_value_score2_array[j]=={}:
        #   max_value_score2_array[j]=0
        # if max_value_combined_score_array[j]=={}:
        #   max_value_combined_score_array[j]=0


      max_value_score1_index = np.where(max_value_score1_array==np.nanmax(max_value_score1_array))[0][0]
    #   max_value_score2_index = np.where(max_value_score2_array==np.nanmax(max_value_score2_array))[0][0]
    #   max_value_combined_score_index= np.where(max_value_combined_score_array==np.nanmax(max_value_combined_score_array))[0][0]

      max_value_score1[s,tar]=np.nanmax(max_value_score1_array)
    #   max_value_score2[s,tar]=np.nanmax(max_value_score2_array)
    #   max_value_combined_score[s,tar]=np.nanmax(max_value_combined_score_array)

      max_value_score1_array_temp=np.array(max_value_score1_array).astype(float)
      max_value_score1_array_temp=np.nan_to_num(max_value_score1_array_temp,nan=0)

      
      Tc_esti_interval_sort = np.sort(Tc_esti_interval[max_value_score1_array_temp.argsort()])[::-1]
    #   print("Tc_esti_interval_sort",Tc_esti_interval_sort)
    #   print("max_value_score1_array_temp",max_value_score1_array_temp)
    #   print("Tc_esti_interval",Tc_esti_interval)
      Tc_esti_interval_sort_top1 = Tc_esti_interval_sort[0]

    #   print("s",s)
    #   print("Tc_esti_interval_sort_top1",Tc_esti_interval_sort_top1)
      Tc_esti_interval_sort_top1_dic[tar].append(Tc_esti_interval_sort_top1)
      et = time.time()
      subdataset_with_cp =  al_dataset_with_cp(whole_data_int,T,N,tau_max,Tc_esti_interval_sort_top1,tar)
      superset_bool_update,subsuperset_parent_dic,superset_bool_BA_Tc = al_super_parent_with_cp(subdataset_with_cp,T,N,tau_max,superset_dict_est,tar)
      superset_bool_pre = superset_bool_option
      superset_bool_option = superset_bool_update

      TP=0
      FN=0
      TN=0
      FP=0
      for r in range(n_regime):
        TP_regime=0
        FN_regime=0
        TN_regime=0
        FP_regime=0
        for k in range(N): # parent index
          for j in range(tau_max+1):
            if (edge_matrix[k,r,tar,j]==1) & (edge_matrix[k,r,tar,j] == superset_bool_BA_Tc[k,r,tar,j]):
              TP+=1
              TP_regime+=1
            if (edge_matrix[k,r,tar,j]==1) & (edge_matrix[k,r,tar,j] != superset_bool_BA_Tc[k,r,tar,j]):
              FN+=1
              FN_regime+=1
            if (edge_matrix[k,r,tar,j]==0) & (edge_matrix[k,r,tar,j] == superset_bool_BA_Tc[k,r,tar,j]):
              TN+=1
              TN_regime+=1
            if (edge_matrix[k,r,tar,j]==0) & (edge_matrix[k,r,tar,j] != superset_bool_BA_Tc[k,r,tar,j]):
              FP+=1
              FP_regime+=1
        # print(TP_regime_array)
        TP_regime_array[tar][r].append(TP_regime)
        FN_regime_array[tar][r].append(FN_regime)
        TN_regime_array[tar][r].append(TN_regime)
        FP_regime_array[tar][r].append(FP_regime)

        total_est_regime_edge[tar][r].append(TP_regime+FP_regime)

        precision_regime[tar][r].append(divide(TP_regime,TP_regime+FP_regime))
        recall_regime[tar][r].append(divide(TP_regime,TP_regime+FN_regime))
        F_1_regime[tar][r].append(divide(TP_regime,TP_regime+1/2*(FN_regime+FP_regime)))

      TP_array[tar].append(TP)
      FN_array[tar].append(FN)
      TN_array[tar].append(TN)
      FP_array[tar].append(FP)

      total_edge = TP+FP
      total_est_edge[tar].append(total_edge)

      precision[tar].append(divide(TP,TP+FP))
      recall[tar].append(divide(TP,TP+FN))
      F_1[tar].append(divide(TP,TP+1/2*(FP+FN)))
    #   print("superset_bool_BA_Tc_regime0_var0",superset_bool_BA_Tc[:,0,0,:])
    #   print("superset_bool_BA_Tc_regime1_var0",superset_bool_BA_Tc[:,1,0,:])
    #   print("superset_bool_BA_Tc_regime0_var1",superset_bool_BA_Tc[:,0,1,:])
    #   print("superset_bool_BA_Tc_regime1_var1",superset_bool_BA_Tc[:,1,1,:])
    #   print("superset_bool_pre", superset_bool_pre )
    #   print("superset_bool_option",superset_bool_option)
    #   print("s",s)

    if s!=0:
      for i in range(N):
        for j in range(len(superset_bool_update[0,tar,:])):
          if superset_bool_update[i,tar,j]==1:
            #  key='{}'.format(k)
            superset_dict_est_final[tar].append((i,-j))
  # get the execution time
  elapsed_time = et - st
  return Tc_esti_interval_sort_top1_dic,s,max_value_score1,superset_dict_est_final,TP_array,FN_array,TN_array,FP_array,precision,recall,F_1,total_est_edge,TP_regime_array,FN_regime_array,TN_regime_array,FP_regime_array,total_est_regime_edge,precision_regime,recall_regime,F_1_regime,superset_bool,elapsed_time,green_flag,max_sample_subtimeseries,average_sample_subtimeseries

def final_result(N,max_value_score1,Tc_esti_interval_sort_top1_dic,TP_array=None,FN_array=None,TN_array=None,FP_array=None):
  iteration=np.zeros(N)
  max_value_score1_array_temp_dic={}
  last_Tc_esti_interval={}


  if TP_array is not None:
    last_Tc_PRF={}
    for k in range(3): # 3: precision, recall, F1_score
      last_Tc_PRF[k]=[]

    TP_temp=np.zeros(N)
    FP_temp=np.zeros(N)
    TN_temp=np.zeros(N)
    FN_temp=np.zeros(N)
    for i in range(N):
      max_value_score1_array_temp_dic[i]=[]
      last_Tc_esti_interval[i]=[]

    for i in range(N):
      iteration[i]=len(Tc_esti_interval_sort_top1_dic[i])
    for i in range(N):
      for j in range(int(iteration[i])):
        max_value_score1_array_temp_dic[i].append(max_value_score1[int(j+1)][i])
    for i in range(N):
      last_Tc_esti_interval[i]=Tc_esti_interval_sort_top1_dic[i][-1]

      TP_temp[i]=TP_array[i][-1]
      FP_temp[i]=FP_array[i][-1]
      TN_temp[i]=TN_array[i][-1]
      FN_temp[i]=FN_array[i][-1]

      last_Tc_PRF[0]=sum(TP_temp)/(sum(TP_temp)+sum(FP_temp))
      last_Tc_PRF[1]=sum(TP_temp)/(sum(TP_temp)+sum(FN_temp))
      last_Tc_PRF[2]=sum(TP_temp)/(sum(TP_temp)+1/2*(sum(FP_temp)+sum(FN_temp)))

    return last_Tc_esti_interval,last_Tc_PRF
  else:
    for i in range(N):
      max_value_score1_array_temp_dic[i]=[]
      last_Tc_esti_interval[i]=[]

    for i in range(N):
      iteration[i]=len(Tc_esti_interval_sort_top1_dic[i])
    for i in range(N):
      for j in range(int(iteration[i])):
        max_value_score1_array_temp_dic[i].append(max_value_score1[int(j+1)][i])
    for i in range(N):
      last_Tc_esti_interval[i]=Tc_esti_interval_sort_top1_dic[i][-1]

    return last_Tc_esti_interval


def discrete_data_generation_cp_model_soft_interven(T,N,tau_max,Tc_matrix,alpha_matrix,domain_of_var_vector,complexity,flag_tau_equal_1,edge_max_pervar):
  if alpha_matrix.shape[1]!= len(domain_of_var_vector):
    raise ValueError("The dim of alpha_matirx is not consistent with the dim of domain_of_var_vector")
  for k in range(N):
    Tc_matrix[k]=np.array(Tc_matrix[k])
  if (Tc_matrix[0]==None).all():
    n_cp = 0
  else:
    n_cp = len(Tc_matrix[0]) # since every time series have the same number of change point, hence Tc_matric[0]=Tc_matric[1]=...Tc_matric[N]
  n_regime = n_cp+1
  # Generate an edge matrix with shape (N,n_cp,N,tau_max+1):
  edge_matrix = np.zeros((N,n_regime,N,tau_max+1))
  # while sum(edge_matrix[:,0,:,:])<6:
  # while sum(edge_matrix[:,0,:,:])<4 or sum(edge_matrix[:,0,:,:])>6:
  edge_from_other_var=np.zeros(N) # total edges across all the regime
  numedge_of_each_var=np.zeros((n_regime,N))
  numedge_of_each_var_max=np.ones(N)+edge_max_pervar
  if edge_max_pervar==1:
    for i in range(N): # the parent var index
      j=0
    # for j in range(n_regime): # the regime index
      for k in range(N):  # the target var index
        if (flag_tau_equal_1==1) & (i==k): # if flag_tau_equal_1 = 1 meaning the edge between X^j_{t-1} and X^j_{t} must exist.
            edge_matrix[i,j,k,1] = 1
    edge_matrix[:,1,:,:]=edge_matrix[:,0,:,:]
    numedge_of_each_var_max=np.zeros(N)+edge_max_pervar
    # print("numedge_of_each_var_max", numedge_of_each_var_max)
  if N >1 and edge_max_pervar>1:
    while int(sum(edge_from_other_var>0))<N or int(sum(numedge_of_each_var_max==edge_max_pervar))!=N: # if there is no edge across different vars or if max for one var is larger than edge_max_pervar, then rerun in the while loop.
    #   print("generate data")
      edge_matrix = np.zeros((N,n_regime,N,tau_max+1))
      edge_from_other_var=np.zeros(N)
      numedge_of_each_var=np.zeros((n_regime,N))
      numedge_of_each_var_max=np.zeros(N)
      for k in range(N):  # the target var index
        j=0
        # for j in range(n_regime): # the regime index
        for i in range(N): # the parent var index
          if (flag_tau_equal_1==1) & (i==k): # if flag_tau_equal_1 = 1 meaning the edge between X^j_{t-1} and X^j_{t} must exist.
              edge_matrix[i,j,k,1] = 1
          # else: # if flag_tau_equal_1 = 0 meaning the edge between X^j_{t-1} and X^j_{t} could not exist.
          #   for tau in range(1,tau_max+1): # time lag index
          #     edge_flag = np.random.binomial(1, complexity, size=None) # the complexity denote the edge_matrix complexity.
          #     edge_matrix[i,j,k,tau] = edge_flag

        # Generate 5 unique random positions
        num_ones = int(edge_max_pervar-1)
        positions = np.random.choice(N * tau_max, num_ones, replace=False)

        # Set the selected positions to 1
        for pos in positions:
            row, col = divmod(pos, tau_max)  # Convert flat index to 2D index
            edge_matrix[row,0,k,int(col+1)] = 1

        if sum(edge_matrix[:,0,k,:])<edge_max_pervar:
            row_that_one = np.random.choice(N,1,replace=False)
            col_that_one = np.random.choice(int(tau_max-1),1,replace=False)
            edge_matrix[row_that_one,0,k,int(col_that_one+2)] = 1
    
        edge_matrix[:,1,:,:]=edge_matrix[:,0,:,:]

      for k in range(N):  # the target var index
        for i in range(N): # the parent var index
          if i==k:
            continue
          else:
            edge_from_other_var[k]+=sum(edge_matrix[i,:,k,:]) # this is to make sure that there must be an edge across different vars.
            # print("edge_from_other_var",edge_from_other_var,"sum(edge_matrix[i,:,k,:]",sum(edge_matrix[i,:,k,:]))
      # print(edge_matrix[:,0,:,:])
      for k in range(N): # the target var index
        for j in range(n_regime):
          numedge_of_each_var[j,k]=sum(edge_matrix[:,j,k,:])
        numedge_of_each_var_max[k]=max(numedge_of_each_var[:,k])
  elif edge_max_pervar>1:
    while int(sum(numedge_of_each_var_max==edge_max_pervar))!=N: # if there is no edge across different vars or if max for one var is larger than edge_max_pervar, then rerun in the while loop.
      edge_from_other_var=np.zeros(N)
      numedge_of_each_var=np.zeros((n_regime,N))
      numedge_of_each_var_max=np.zeros(N)
      for i in range(N): # the parent var index
        j=0
        # for j in range(n_regime): # the regime index
        for k in range(N):  # the target var index
           # if flag_tau_equal_1 = 0 meaning the edge between X^j_{t-1} and X^j_{t} could not exist.
          for tau in range(1,tau_max+1): # time lag index
            edge_flag = np.random.binomial(1, complexity, size=None) # the complexity denote the edge_matrix complexity.
            edge_matrix[i,j,k,tau] = edge_flag
      edge_matrix[:,1,:,:]=edge_matrix[:,0,:,:]
      for k in range(N):  # the target var index
        for i in range(N): # the parent var index
          if i==k:
            continue
          else:
            edge_from_other_var[k]+=sum(edge_matrix[i,:,k,:]) # this is to make sure that there must be an edge across different vars.
            # print("edge_from_other_var",edge_from_other_var,"sum(edge_matrix[i,:,k,:]",sum(edge_matrix[i,:,k,:]))
      # print(edge_matrix[:,0,:,:])
      for k in range(N): # the target var index
        for j in range(n_regime):
          numedge_of_each_var[j,k]=sum(edge_matrix[:,j,k,:])
        numedge_of_each_var_max[k]=max(numedge_of_each_var[:,k])
#         print("edge_from_other_var",edge_from_other_var,"numedge_of_each_var_max",numedge_of_each_var_max)
#   print("edge_from_other_var",edge_from_other_var,"numedge_of_each_var_max",numedge_of_each_var_max)
  # print(edge_matrix)
  # Based on the edge_matrix, we know the parent set for each variable at each regime.
  # Now needs to generate the conditional distribution based on dirichlet distributions (given alpha_vector and domain_of_var_vector)
  # alpha_matrix: shape is [N,len(domain_of_var_vector)]. Refer to https://stats.stackexchange.com/questions/244917/what-exactly-is-the-alpha-in-the-dirichlet-distribution for explaination
  #               Each row is for each time series X^j, j\in[N], and for each X^j, the conditional distribution of X^j given its parent relization have the same alpha_vetor value but
  #               the order could be different. For instance, if for X^0, alpha = [1,0.9], then given the parent relization 1, the alpha is [1,0.9], for relization 2, the alpha could be [1,0.9] or [0.9,1]
  # domain_of_var_vecotr: the value that variable can take. For binary variable, it is [0,1]

  # Generate the conditional distribution table given
  key_target = np.arange(N)
  var_dic={}
  for j in range(n_regime):
    var_dic[j]={}
  for j in range(n_regime):
    for k in range(N): # target
      var_dic[j][key_target[k]] = []
  for j in range(n_regime):# index of regime
    for k in range(N): # target variable (children)
      for i in range(N): # parent index
        for tau in range(tau_max+1):
          if edge_matrix[i,j,k,tau]==1:
            var_dic[j][key_target[k]].append((key_target[i],-tau))
   #In the var, the first dic key is the index of regime, in each regime, the dic key is the target variable and the pair in the [] is (parent_index, time_lag).
  # alpha_matrix=np.ones(shape=(N,len(domain_of_var_vector)))
  n_value = len(domain_of_var_vector)
  n_par = np.zeros(shape=(n_regime,N))
  conditional_table={}
  for j in range(n_regime):
    conditional_table[j]={}
  for j in range(n_regime):
    for k in range(N): # target
      conditional_table[j][key_target[k]] = []
  for j in range(n_regime):
    for k in range(N):
      n_par[j,k] = sum(edge_matrix[:,j,k,:])
      temp_n_par_relization = pow(n_value,n_par[j,k])
      conditional_table[j][k] = stats.dirichlet.rvs(np.roll(alpha_matrix[k],k),size=int(temp_n_par_relization))
  #For conditional table, {0: {0: array}, first 0 is the regime index, second 0 is the target variable index, the array [2^{number of parent}, n_value] is the conditional table.
  #Next step: generate time series data according to the conditional table
  data=np.zeros(shape=(T,N))
  for k in range(N):
    for t in range(tau_max):
      data[t,k] =np.random.choice(domain_of_var_vector, size=1, replace = True, p=[1/len(domain_of_var_vector)]*len(domain_of_var_vector))[0]
  # print(data[0:tau_max,:])
  ####### The above is for the starting point##################
  # print(time_point_array)
  # print(time_point_array)
  time_point_array = np.zeros(shape=(N,n_cp+2))
  for k in range(N):
    time_point_array[k][0]=tau_max
    time_point_array[k][-1]=T
    if (Tc_matrix[k] != None).all():
      time_point_array[k][1:time_point_array.shape[1]-1]=Tc_matrix[k]
#   print(time_point_array)

  for t in range(tau_max,T):
    for k in range(N):
      for j in range(n_regime):
        if t in range(int(time_point_array[k][j]),int(time_point_array[k][j+1])):
          # print(int(time_point_array[k][j]),int(time_point_array[k][j+1]),"t",t,"var",k,'regime',j) #######Check!!!!
          n_par = sum(edge_matrix[:,j,k,:]) # jth regime and k th variable; n_par: how many parents the kth variable in jth regime have
          n_par=int(n_par)
          parent_configuration_number = int(pow(n_value,sum(edge_matrix[:,j,k,:]))) # how many configurations: n_value ^ n_par
          # print(parent_configuration_number,"parent_configuration_number")
          parent_configuration_array = list(itertools.product(domain_of_var_vector, repeat=n_par)) # the list of all configurations.
          # print(parent_configuration_array,"parent_configuration_array",n_par,"n_par")
          parent_list=[]
          for n_par_index in range(n_par):
            # print(n_par,"n_par")
            # print(j,"j",k,"k")
            parent_list.append(data[t+var_dic[j][k][n_par_index][1],var_dic[j][k][n_par_index][0]])
          parent_list = np.array(parent_list)
          # print(parent_list," parent_list") #######Check!!!!
          for configuration_index in range(parent_configuration_number):
            # print(configuration_index)
            temp_conditional_dis = []
            if (parent_list == parent_configuration_array[configuration_index]).all():
              temp_conditional_dis = conditional_table[j][k][configuration_index]
              # print(temp_conditional_dis)  #######Check!!!!
              break
          data[t,k] = np.random.choice(domain_of_var_vector, size=1, replace = True, p=temp_conditional_dis)[0]
        else:
          continue
  return data, conditional_table,edge_matrix,var_dic

def discrete_data_generation_cp_model_hard_interven(T,N,tau_max,Tc_matrix,alpha_matrix,domain_of_var_vector,complexity,flag_tau_equal_1,numunion_edge):
  if alpha_matrix.shape[1]!= len(domain_of_var_vector):
    raise ValueError("The dim of alpha_matirx is not consistent with the dim of domain_of_var_vector")
  for k in range(N):
    Tc_matrix[k]=np.array(Tc_matrix[k])
  if (Tc_matrix[0]==None).all():
    n_cp = 0
  else:
    n_cp = len(Tc_matrix[0]) # since every time series have the same number of change point, hence Tc_matric[0]=Tc_matric[1]=...Tc_matric[N]
  n_regime = n_cp+1
  # Generate an edge matrix with shape (N,n_cp,N,tau_max+1):
  edge_matrix = np.zeros((N,n_regime,N,tau_max+1))
  # while sum(edge_matrix[:,0,:,:])<6:
  # while sum(edge_matrix[:,0,:,:])<4 or sum(edge_matrix[:,0,:,:])>6:
  edge_from_other_var=np.zeros(N) # total edges across all the regime
  numedge_of_each_var=np.zeros((n_regime,N))
  num_union_edge_matrix=np.ones(N)+numunion_edge
  if N >1 and numunion_edge>1:
    while int(sum(edge_from_other_var>0))<N or int(sum(num_union_edge_matrix==numunion_edge))!=N: # if there is no edge across different vars or if max for one var is larger than edge_max_pervar, then rerun in the while loop.
      edge_from_other_var=np.zeros(N)
      numedge_of_each_var=np.zeros((n_regime,N))
      numedge_of_each_var_max=np.zeros(N)
      num_union_edge_matrix=np.zeros(N)
      union_edge_matrix_temp=np.zeros((N,N,tau_max+1))
      for i in range(N): # the parent var index
        j=0
        # for j in range(n_regime): # the regime index
        for k in range(N):  # the target var index
          if (flag_tau_equal_1==1) & (i==k): # if flag_tau_equal_1 = 1 meaning the edge between X^j_{t-1} and X^j_{t} must exist.
              edge_matrix[i,j,k,1] = 1
          else: # if flag_tau_equal_1 = 0 meaning the edge between X^j_{t-1} and X^j_{t} could not exist.
            for tau in range(1,tau_max+1): # time lag index
              edge_flag = np.random.binomial(1, complexity, size=None) # the complexity denote the edge_matrix complexity.
              edge_matrix[i,j,k,tau] = edge_flag
      for i in range(N): # the parent var index
        j=1
        # for j in range(n_regime): # the regime index
        for k in range(N):  # the target var index
          if (flag_tau_equal_1==1) & (i==k): # if flag_tau_equal_1 = 1 meaning the edge between X^j_{t-1} and X^j_{t} must exist.
              edge_matrix[i,j,k,1] = 1
          else: # if flag_tau_equal_1 = 0 meaning the edge between X^j_{t-1} and X^j_{t} could not exist.
            for tau in range(1,tau_max+1): # time lag index
              edge_flag = np.random.binomial(1, complexity, size=None) # the complexity denote the edge_matrix complexity.
              edge_matrix[i,j,k,tau] = edge_flag
      for k in range(N):  # the target var index
        for i in range(N): # the parent var index
          if i==k:
            continue
          else:
            edge_from_other_var[k]+=sum(edge_matrix[i,:,k,:]) # this is to make sure that there must be an edge across different vars.
            # print("edge_from_other_var",edge_from_other_var,"sum(edge_matrix[i,:,k,:]",sum(edge_matrix[i,:,k,:]))
      # print(edge_matrix[:,0,:,:])
      for k in range(N):
        for i in range(N):
          for q in range(tau_max+1):
            union_edge_matrix_temp[:,k,:]=edge_matrix[:,0,k,:]+edge_matrix[:,1,k,:]
      for k in range(N):
        for i in range(N):
          for q in range(tau_max+1):     
            if union_edge_matrix_temp[i,k,q]>=1:
              union_edge_matrix_temp[i,k,q]=1
        num_union_edge_matrix[k]+=sum(union_edge_matrix_temp[:,k,:])
  elif numunion_edge>1:
    while int(sum(edge_from_other_var>0))<N or int(sum(num_union_edge_matrix==numunion_edge))!=N: # if there is no edge across different vars or if max for one var is larger than edge_max_pervar, then rerun in the while loop.
      edge_from_other_var=np.zeros(N)
      numedge_of_each_var=np.zeros((n_regime,N))
      numedge_of_each_var_max=np.zeros(N)
      num_union_edge_matrix=np.zeros(N)
      union_edge_matrix_temp=np.zeros((N,N,tau_max+1))
      for i in range(N): # the parent var index
        j=0
        # for j in range(n_regime): # the regime index
        for k in range(N):  # the target var index
          if (flag_tau_equal_1==1) & (i==k): # if flag_tau_equal_1 = 1 meaning the edge between X^j_{t-1} and X^j_{t} must exist.
              edge_matrix[i,j,k,1] = 1
          else: # if flag_tau_equal_1 = 0 meaning the edge between X^j_{t-1} and X^j_{t} could not exist.
            for tau in range(1,tau_max+1): # time lag index
              edge_flag = np.random.binomial(1, complexity, size=None) # the complexity denote the edge_matrix complexity.
              edge_matrix[i,j,k,tau] = edge_flag
      for i in range(N): # the parent var index
        j=1
        # for j in range(n_regime): # the regime index
        for k in range(N):  # the target var index
          if (flag_tau_equal_1==1) & (i==k): # if flag_tau_equal_1 = 1 meaning the edge between X^j_{t-1} and X^j_{t} must exist.
              edge_matrix[i,j,k,1] = 1
          else: # if flag_tau_equal_1 = 0 meaning the edge between X^j_{t-1} and X^j_{t} could not exist.
            for tau in range(1,tau_max+1): # time lag index
              edge_flag = np.random.binomial(1, complexity, size=None) # the complexity denote the edge_matrix complexity.
              edge_matrix[i,j,k,tau] = edge_flag
      for k in range(N):  # the target var index
        for i in range(N): # the parent var index
          if i==k:
            continue
          else:
            edge_from_other_var[k]+=sum(edge_matrix[i,:,k,:]) # this is to make sure that there must be an edge across different vars.
            # print("edge_from_other_var",edge_from_other_var,"sum(edge_matrix[i,:,k,:]",sum(edge_matrix[i,:,k,:]))
      # print(edge_matrix[:,0,:,:])
      for k in range(N): # the target var index
        for j in range(n_regime):
          numedge_of_each_var[j,k]=sum(edge_matrix[:,j,k,:])
        # numedge_of_each_var_max[k]=max(numedge_of_each_var[:,k])
      for k in range(N):
        for i in range(N):
          for q in range(tau_max+1):
            union_edge_matrix_temp[:,k,:]=edge_matrix[:,0,k,:]+edge_matrix[:,1,k,:]
      for k in range(N):
        for i in range(N):
          for q in range(tau_max+1):     
            if union_edge_matrix_temp[i,k,q]>=1:
              union_edge_matrix_temp[i,k,q]=1
        num_union_edge_matrix[k]+=sum(union_edge_matrix_temp[:,k,:])
        # print("edge_from_other_var",edge_from_other_var,"numedge_of_each_var_max",numedge_of_each_var_max)
#   print("edge_from_other_var",edge_from_other_var," union_edge_matrix_temp",num_union_edge_matrix)
  # print(edge_matrix)
  # Based on the edge_matrix, we know the parent set for each variable at each regime.
  # Now needs to generate the conditional distribution based on dirichlet distributions (given alpha_vector and domain_of_var_vector)
  # alpha_matrix: shape is [N,len(domain_of_var_vector)]. Refer to https://stats.stackexchange.com/questions/244917/what-exactly-is-the-alpha-in-the-dirichlet-distribution for explaination
  #               Each row is for each time series X^j, j\in[N], and for each X^j, the conditional distribution of X^j given its parent relization have the same alpha_vetor value but
  #               the order could be different. For instance, if for X^0, alpha = [1,0.9], then given the parent relization 1, the alpha is [1,0.9], for relization 2, the alpha could be [1,0.9] or [0.9,1]
  # domain_of_var_vecotr: the value that variable can take. For binary variable, it is [0,1]

  # Generate the conditional distribution table given
  key_target = np.arange(N)
  var_dic={}
  for j in range(n_regime):
    var_dic[j]={}
  for j in range(n_regime):
    for k in range(N): # target
      var_dic[j][key_target[k]] = []
  for j in range(n_regime):# index of regime
    for k in range(N): # target variable (children)
      for i in range(N): # parent index
        for tau in range(tau_max+1):
          if edge_matrix[i,j,k,tau]==1:
            var_dic[j][key_target[k]].append((key_target[i],-tau))
   #In the var, the first dic key is the index of regime, in each regime, the dic key is the target variable and the pair in the [] is (parent_index, time_lag).
  # alpha_matrix=np.ones(shape=(N,len(domain_of_var_vector)))
  n_value = len(domain_of_var_vector)
  n_par = np.zeros(shape=(n_regime,N))
  conditional_table={}
  for j in range(n_regime):
    conditional_table[j]={}
  for j in range(n_regime):
    for k in range(N): # target
      conditional_table[j][key_target[k]] = []
  for j in range(n_regime):
    for k in range(N):
      n_par[j,k] = sum(edge_matrix[:,j,k,:])
      temp_n_par_relization = pow(n_value,n_par[j,k])
      conditional_table[j][k] = stats.dirichlet.rvs(np.roll(alpha_matrix[k],k),size=int(temp_n_par_relization))
  #For conditional table, {0: {0: array}, first 0 is the regime index, second 0 is the target variable index, the array [2^{number of parent}, n_value] is the conditional table.
  #Next step: generate time series data according to the conditional table
  data=np.zeros(shape=(T,N))
  for k in range(N):
    for t in range(tau_max):
      data[t,k] =np.random.choice(domain_of_var_vector, size=1, replace = True, p=[1/len(domain_of_var_vector)]*len(domain_of_var_vector))[0]
  # print(data[0:tau_max,:])
  ####### The above is for the starting point##################
  # print(time_point_array)
  # print(time_point_array)
  time_point_array = np.zeros(shape=(N,n_cp+2))
  for k in range(N):
    time_point_array[k][0]=tau_max
    time_point_array[k][-1]=T
    if (Tc_matrix[k] != None).all():
      time_point_array[k][1:time_point_array.shape[1]-1]=Tc_matrix[k]
#   print(time_point_array)

  for t in range(tau_max,T):
    for k in range(N):
      for j in range(n_regime):
        if t in range(int(time_point_array[k][j]),int(time_point_array[k][j+1])):
          # print(int(time_point_array[k][j]),int(time_point_array[k][j+1]),"t",t,"var",k,'regime',j) #######Check!!!!
          n_par = sum(edge_matrix[:,j,k,:]) # jth regime and k th variable; n_par: how many parents the kth variable in jth regime have
          n_par=int(n_par)
          parent_configuration_number = int(pow(n_value,sum(edge_matrix[:,j,k,:]))) # how many configurations: n_value ^ n_par
          # print(parent_configuration_number,"parent_configuration_number")
          parent_configuration_array = list(itertools.product(domain_of_var_vector, repeat=n_par)) # the list of all configurations.
          # print(parent_configuration_array,"parent_configuration_array",n_par,"n_par")
          parent_list=[]
          for n_par_index in range(n_par):
            # print(n_par,"n_par")
            # print(j,"j",k,"k")
            parent_list.append(data[t+var_dic[j][k][n_par_index][1],var_dic[j][k][n_par_index][0]])
          parent_list = np.array(parent_list)
          # print(parent_list," parent_list") #######Check!!!!
          for configuration_index in range(parent_configuration_number):
            # print(configuration_index)
            temp_conditional_dis = []
            if (parent_list == parent_configuration_array[configuration_index]).all():
              temp_conditional_dis = conditional_table[j][k][configuration_index]
              # print(temp_conditional_dis)  #######Check!!!!
              break
          data[t,k] = np.random.choice(domain_of_var_vector, size=1, replace = True, p=temp_conditional_dis)[0]
        else:
          continue
  return data, conditional_table,edge_matrix,var_dic
 
def discrete_data_generation_cp_model_hard_interven(T,N,tau_max,Tc_matrix,alpha_matrix,domain_of_var_vector,complexity,flag_tau_equal_1,numunion_edge):
  if alpha_matrix.shape[1]!= len(domain_of_var_vector):
    raise ValueError("The dim of alpha_matirx is not consistent with the dim of domain_of_var_vector")
  for k in range(N):
    Tc_matrix[k]=np.array(Tc_matrix[k])
  if (Tc_matrix[0]==None).all():
    n_cp = 0
  else:
    n_cp = len(Tc_matrix[0]) # since every time series have the same number of change point, hence Tc_matric[0]=Tc_matric[1]=...Tc_matric[N]
  n_regime = n_cp+1
  # Generate an edge matrix with shape (N,n_cp,N,tau_max+1):
  edge_matrix = np.zeros((N,n_regime,N,tau_max+1))
  # while sum(edge_matrix[:,0,:,:])<6:
  # while sum(edge_matrix[:,0,:,:])<4 or sum(edge_matrix[:,0,:,:])>6:
  edge_from_other_var=np.zeros(N) # total edges across all the regime
  numedge_of_each_var=np.zeros((n_regime,N))
  num_union_edge_matrix=np.ones(N)+numunion_edge
  if N >1 and numunion_edge>1:
    while int(sum(edge_from_other_var>0))<N or int(sum(num_union_edge_matrix==numunion_edge))!=N: # if there is no edge across different vars or if max for one var is larger than edge_max_pervar, then rerun in the while loop.
      edge_from_other_var=np.zeros(N)
      numedge_of_each_var=np.zeros((n_regime,N))
      numedge_of_each_var_max=np.zeros(N)
      num_union_edge_matrix=np.zeros(N)
      union_edge_matrix_temp=np.zeros((N,N,tau_max+1))
      for i in range(N): # the parent var index
        j=0
        # for j in range(n_regime): # the regime index
        for k in range(N):  # the target var index
          if (flag_tau_equal_1==1) & (i==k): # if flag_tau_equal_1 = 1 meaning the edge between X^j_{t-1} and X^j_{t} must exist.
              edge_matrix[i,j,k,1] = 1
          else: # if flag_tau_equal_1 = 0 meaning the edge between X^j_{t-1} and X^j_{t} could not exist.
            for tau in range(1,tau_max+1): # time lag index
              edge_flag = np.random.binomial(1, complexity, size=None) # the complexity denote the edge_matrix complexity.
              edge_matrix[i,j,k,tau] = edge_flag
      for i in range(N): # the parent var index
        j=1
        # for j in range(n_regime): # the regime index
        for k in range(N):  # the target var index
          if (flag_tau_equal_1==1) & (i==k): # if flag_tau_equal_1 = 1 meaning the edge between X^j_{t-1} and X^j_{t} must exist.
              edge_matrix[i,j,k,1] = 1
          else: # if flag_tau_equal_1 = 0 meaning the edge between X^j_{t-1} and X^j_{t} could not exist.
            for tau in range(1,tau_max+1): # time lag index
              edge_flag = np.random.binomial(1, complexity, size=None) # the complexity denote the edge_matrix complexity.
              edge_matrix[i,j,k,tau] = edge_flag
      for k in range(N):  # the target var index
        for i in range(N): # the parent var index
          if i==k:
            continue
          else:
            edge_from_other_var[k]+=sum(edge_matrix[i,:,k,:]) # this is to make sure that there must be an edge across different vars.
            # print("edge_from_other_var",edge_from_other_var,"sum(edge_matrix[i,:,k,:]",sum(edge_matrix[i,:,k,:]))
      # print(edge_matrix[:,0,:,:])
      for k in range(N):
        for i in range(N):
          for q in range(tau_max+1):
            union_edge_matrix_temp[:,k,:]=edge_matrix[:,0,k,:]+edge_matrix[:,1,k,:]
      for k in range(N):
        for i in range(N):
          for q in range(tau_max+1):     
            if union_edge_matrix_temp[i,k,q]>=1:
              union_edge_matrix_temp[i,k,q]=1
        num_union_edge_matrix[k]+=sum(union_edge_matrix_temp[:,k,:])
  elif numunion_edge>1:
    while int(sum(edge_from_other_var>0))<N or int(sum(num_union_edge_matrix==numunion_edge))!=N: # if there is no edge across different vars or if max for one var is larger than edge_max_pervar, then rerun in the while loop.
      edge_from_other_var=np.zeros(N)
      numedge_of_each_var=np.zeros((n_regime,N))
      numedge_of_each_var_max=np.zeros(N)
      num_union_edge_matrix=np.zeros(N)
      union_edge_matrix_temp=np.zeros((N,N,tau_max+1))
      for i in range(N): # the parent var index
        j=0
        # for j in range(n_regime): # the regime index
        for k in range(N):  # the target var index
          if (flag_tau_equal_1==1) & (i==k): # if flag_tau_equal_1 = 1 meaning the edge between X^j_{t-1} and X^j_{t} must exist.
              edge_matrix[i,j,k,1] = 1
          else: # if flag_tau_equal_1 = 0 meaning the edge between X^j_{t-1} and X^j_{t} could not exist.
            for tau in range(1,tau_max+1): # time lag index
              edge_flag = np.random.binomial(1, complexity, size=None) # the complexity denote the edge_matrix complexity.
              edge_matrix[i,j,k,tau] = edge_flag
      for i in range(N): # the parent var index
        j=1
        # for j in range(n_regime): # the regime index
        for k in range(N):  # the target var index
          if (flag_tau_equal_1==1) & (i==k): # if flag_tau_equal_1 = 1 meaning the edge between X^j_{t-1} and X^j_{t} must exist.
              edge_matrix[i,j,k,1] = 1
          else: # if flag_tau_equal_1 = 0 meaning the edge between X^j_{t-1} and X^j_{t} could not exist.
            for tau in range(1,tau_max+1): # time lag index
              edge_flag = np.random.binomial(1, complexity, size=None) # the complexity denote the edge_matrix complexity.
              edge_matrix[i,j,k,tau] = edge_flag
      for k in range(N):  # the target var index
        for i in range(N): # the parent var index
          if i==k:
            continue
          else:
            edge_from_other_var[k]+=sum(edge_matrix[i,:,k,:]) # this is to make sure that there must be an edge across different vars.
            # print("edge_from_other_var",edge_from_other_var,"sum(edge_matrix[i,:,k,:]",sum(edge_matrix[i,:,k,:]))
      # print(edge_matrix[:,0,:,:])
      for k in range(N): # the target var index
        for j in range(n_regime):
          numedge_of_each_var[j,k]=sum(edge_matrix[:,j,k,:])
        # numedge_of_each_var_max[k]=max(numedge_of_each_var[:,k])
      for k in range(N):
        for i in range(N):
          for q in range(tau_max+1):
            union_edge_matrix_temp[:,k,:]=edge_matrix[:,0,k,:]+edge_matrix[:,1,k,:]
      for k in range(N):
        for i in range(N):
          for q in range(tau_max+1):     
            if union_edge_matrix_temp[i,k,q]>=1:
              union_edge_matrix_temp[i,k,q]=1
        num_union_edge_matrix[k]+=sum(union_edge_matrix_temp[:,k,:])
        # print("edge_from_other_var",edge_from_other_var,"numedge_of_each_var_max",numedge_of_each_var_max)
#   print("edge_from_other_var",edge_from_other_var," union_edge_matrix_temp",num_union_edge_matrix)
  # print(edge_matrix)
  # Based on the edge_matrix, we know the parent set for each variable at each regime.
  # Now needs to generate the conditional distribution based on dirichlet distributions (given alpha_vector and domain_of_var_vector)
  # alpha_matrix: shape is [N,len(domain_of_var_vector)]. Refer to https://stats.stackexchange.com/questions/244917/what-exactly-is-the-alpha-in-the-dirichlet-distribution for explaination
  #               Each row is for each time series X^j, j\in[N], and for each X^j, the conditional distribution of X^j given its parent relization have the same alpha_vetor value but
  #               the order could be different. For instance, if for X^0, alpha = [1,0.9], then given the parent relization 1, the alpha is [1,0.9], for relization 2, the alpha could be [1,0.9] or [0.9,1]
  # domain_of_var_vecotr: the value that variable can take. For binary variable, it is [0,1]

  # Generate the conditional distribution table given
  key_target = np.arange(N)
  var_dic={}
  for j in range(n_regime):
    var_dic[j]={}
  for j in range(n_regime):
    for k in range(N): # target
      var_dic[j][key_target[k]] = []
  for j in range(n_regime):# index of regime
    for k in range(N): # target variable (children)
      for i in range(N): # parent index
        for tau in range(tau_max+1):
          if edge_matrix[i,j,k,tau]==1:
            var_dic[j][key_target[k]].append((key_target[i],-tau))
   #In the var, the first dic key is the index of regime, in each regime, the dic key is the target variable and the pair in the [] is (parent_index, time_lag).
  # alpha_matrix=np.ones(shape=(N,len(domain_of_var_vector)))
  n_value = len(domain_of_var_vector)
  n_par = np.zeros(shape=(n_regime,N))
  conditional_table={}
  for j in range(n_regime):
    conditional_table[j]={}
  for j in range(n_regime):
    for k in range(N): # target
      conditional_table[j][key_target[k]] = []
  for j in range(n_regime):
    for k in range(N):
      n_par[j,k] = sum(edge_matrix[:,j,k,:])
      temp_n_par_relization = pow(n_value,n_par[j,k])
      conditional_table[j][k] = stats.dirichlet.rvs(np.roll(alpha_matrix[k],k),size=int(temp_n_par_relization))
  #For conditional table, {0: {0: array}, first 0 is the regime index, second 0 is the target variable index, the array [2^{number of parent}, n_value] is the conditional table.
  #Next step: generate time series data according to the conditional table
  data=np.zeros(shape=(T,N))
  for k in range(N):
    for t in range(tau_max):
      data[t,k] =np.random.choice(domain_of_var_vector, size=1, replace = True, p=[1/len(domain_of_var_vector)]*len(domain_of_var_vector))[0]
  # print(data[0:tau_max,:])
  ####### The above is for the starting point##################
  # print(time_point_array)
  # print(time_point_array)
  time_point_array = np.zeros(shape=(N,n_cp+2))
  for k in range(N):
    time_point_array[k][0]=tau_max
    time_point_array[k][-1]=T
    if (Tc_matrix[k] != None).all():
      time_point_array[k][1:time_point_array.shape[1]-1]=Tc_matrix[k]
#   print(time_point_array)

  for t in range(tau_max,T):
    for k in range(N):
      for j in range(n_regime):
        if t in range(int(time_point_array[k][j]),int(time_point_array[k][j+1])):
          # print(int(time_point_array[k][j]),int(time_point_array[k][j+1]),"t",t,"var",k,'regime',j) #######Check!!!!
          n_par = sum(edge_matrix[:,j,k,:]) # jth regime and k th variable; n_par: how many parents the kth variable in jth regime have
          n_par=int(n_par)
          parent_configuration_number = int(pow(n_value,sum(edge_matrix[:,j,k,:]))) # how many configurations: n_value ^ n_par
          # print(parent_configuration_number,"parent_configuration_number")
          parent_configuration_array = list(itertools.product(domain_of_var_vector, repeat=n_par)) # the list of all configurations.
          # print(parent_configuration_array,"parent_configuration_array",n_par,"n_par")
          parent_list=[]
          for n_par_index in range(n_par):
            # print(n_par,"n_par")
            # print(j,"j",k,"k")
            parent_list.append(data[t+var_dic[j][k][n_par_index][1],var_dic[j][k][n_par_index][0]])
          parent_list = np.array(parent_list)
          # print(parent_list," parent_list") #######Check!!!!
          for configuration_index in range(parent_configuration_number):
            # print(configuration_index)
            temp_conditional_dis = []
            if (parent_list == parent_configuration_array[configuration_index]).all():
              temp_conditional_dis = conditional_table[j][k][configuration_index]
              # print(temp_conditional_dis)  #######Check!!!!
              break
          data[t,k] = np.random.choice(domain_of_var_vector, size=1, replace = True, p=temp_conditional_dis)[0]
        else:
          continue
  return data, conditional_table,edge_matrix,var_dic
  

def est_summary_casaul_graph(ar,icml_of_true_and_est,omega,N):
  new_ar= np.zeros(shape=(N,icml_of_true_and_est,N,tau_max+1)) #if icml_of_ture_and_est=2, omega=2 new_ar=N,2,N,tau+1
  for i in range(N):
    omega_single=omega[i] #omega_single=2
    new_ar[i][0:omega_single]=ar[i][0:omega_single] #0:2=0:2
    replicate_num=int(icml_of_true_and_est/omega_single) #=1
    if replicate_num!=1:
      for j in range(replicate_num-1): #j=0
        new_ar[i][omega_single+j*omega_single:omega_single+(j+1)*omega_single]=ar[i][0:omega_single] #2:4=0:2
      #int(icml_of_true_and_est/omega_hat_single)
  return new_ar

def LCMofArray(a):
  lcm = a[0]
  for i in range(1,len(a)):
    lcm = lcm*a[i]//math.gcd(lcm, a[i])
  return lcm


def PCMCI_result(superset_bool,Omega,true_edge_array,tau_max_pcmci):
  PCMCI_result = np.zeros(shape=(N,N,tau_max_pcmci+1))
  tem_array2=deepcopy(superset_bool)
  for i in range(N):
    PCMCI_result[i]=tem_array2[:,i,:]
  PCMCI_result=np.expand_dims(PCMCI_result,axis=1)
  PCMCI_result.shape

  omega_PCMCI=np.array(np.zeros(shape=(N))+1,dtype=int)
  merge_omega_2=np.concatenate((omega_PCMCI,Omega))
  lcm_2=LCMofArray(merge_omega_2)

  est_summary_matrix=est_summary_casaul_graph(PCMCI_result,lcm_2,omega_PCMCI,N)
  summary_matrix=est_summary_casaul_graph(true_edge_array,lcm_2,Omega,N)

#   print(est_summary_matrix.shape)
#   print(summary_matrix.shape)
#   print(sum(est_summary_matrix))
#   print(sum(summary_matrix))
  metric_matrix=est_summary_matrix-summary_matrix
  False_Negative=sum(metric_matrix == -1) #False Negative
  False_Positive=sum(metric_matrix == 1)  #False Positive
  sum_true=sum(summary_matrix) # Total positive-False Negative=True Positive
  True_Positive = sum_true-False_Negative
  precision=True_Positive/(True_Positive+False_Positive)
  recall=True_Positive/(True_Positive+False_Negative)
  F1_score=True_Positive/(True_Positive+1/2*(False_Positive+False_Negative))
  # print("est_summary_matrix={}".format(est_summary_matrix))
  # print("sum_true={}".format(sum_true))
  # print("False_Negative={}".format(False_Negative))
  # print("False_Positive={}".format(False_Positive))
  # print("True_Positive={}".format(True_Positive))
  return precision,recall,F1_score
