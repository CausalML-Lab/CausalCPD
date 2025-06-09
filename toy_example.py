# FILENAME: toy_example.py
# Author: Shanyun Gao
# Paper: https://openreview.net/forum?id=3tV5AtAXk0
# Note that this code utilizes the old verision of tigramite package(v5.1.0), with a significant portion aligning with the logic present in functions from the tigramite package.
# The current version of CPD algorithm is designed to handle only one change point. For multiple change points, divide the data manually into segments with one change point each, run the algorithm on each segment with the same parameters, and then merge the results.

# Running time for this toy example: 104 seconds
import sys
# sys.path.insert(0,'/home/gao565/PCMCI_CP') 
import numpy as np
from causal_rulsif.utils import CP_algorithm, final_result
import pickle

n=50                        # 2*n is the window size; n is the half window size.
s_bound=8                   # The early stop threshold. It can be set to a large value, as the algorithm typically stops before reaching 4.
cv_flag=0                   # If set to 0, the algorithm will not use cross-validation. If set to 1, the algorithm will use cross-validation.        
topK=None                   # This means only the parents with the top K statistics from the PC algorithm are considered for each variable. Be default, we consider all parents.
                            # However, if the sample are not enough, set topK=1,2,...
alpha_i=1                   # By default, this is set to 1. No need to change this value.
alpha_0=0.5                 # By default, this is set to 0.5. No need to change this value.
k_dim_condi=1               # By default, this number is set to be 1.
tau_max=3                   # The maximum time lag to be considered.
data = np.loadtxt('test_data.csv', dtype=int, delimiter=',')    #The data is a numpy array with shape (T, N), where T is the number of time points and N is the number of variables.
T, N = data.shape
domain_size=2
domain_of_var_vector=np.arange(0,domain_size+1) # if domain_size=3, then domain_of_var_vector=[0,1,2]: this is the domain of each variable


######## If the true edge_matrix (representing the causal relations) is unknown, you can uncomment the following lines. ########
######## If the true edge_matrix (representing the causal relations) is unknown, you can uncomment the following lines. ########
######## If the true edge_matrix (representing the causal relations) is unknown, you can uncomment the following lines. ########

edge_matrix = None
Tc_esti_interval_sort_top1_dic,s,max_value_score1,superset_dict_est_final,total_est_edge,total_est_regime_edge,superset_bool,elapsed_time,green_flag,max_sample_subtimeseries,average_sample_subtimeseries = CP_algorithm(data,T,N,s_bound,n,domain_of_var_vector,tau_max,edge_matrix,cv_flag,topK,alpha_i,alpha_0,k_dim_condi)
last_Tc_esti_interval = final_result(N,max_value_score1,Tc_esti_interval_sort_top1_dic)
print("last_Tc_esti_interval",last_Tc_esti_interval)
## This is the true change point matrix, which is a numpy array with shape (N,). Each element is the index of the change point for the corresponding univariate time series.
## This is only used for evaluation purposes, and is not used in the algorithm.
with open(f'test_Tc_matrix.pkl', 'rb') as f_Tc_matrix:
    Tc_matrix = pickle.load(f_Tc_matrix)

for i in range(N):
  print("Estimated Change Point interval for "+str(i)+"th variable:" +str(last_Tc_esti_interval[i])+" vs true change point index:" +str(Tc_matrix[i]))


######### If you want to use the edge_matrix, you can uncomment the following lines. ########
######### If you want to use the edge_matrix, you can uncomment the following lines. ########
######### If you want to use the edge_matrix, you can uncomment the following lines. ########

# The edge_matrix should be in shape of (N,n_regime,N,tau_max+1) where the first N indicates the parent variable index, n_regime indicates the regime index, and the last N indicates the child variable index. The tau_max+1 indicates the time lag.
with open(f'test_edgematrix.pkl', 'rb') as f_edgematrix:
    edge_matrix = pickle.load(f_edgematrix)
# The var_dic is the dictionary representation of the edge_matrix; the first key is the index of regime, the second key is the target variable index, and the value is a list of tuples (parent_index, time_lag).
with open(f'test_saved_dictionary_vardic.pkl', 'rb') as f_vardic:
    var_dic = pickle.load(f_vardic)

Tc_esti_interval_sort_top1_dic,s,max_value_score1,superset_dict_est_final,TP_array,FN_array,TN_array,FP_array,precision,recall,F_1,total_est_edge,TP_regime_array,FN_regime_array,TN_regime_array,FP_regime_array,total_est_regime_edge,precision_regime,recall_regime,F_1_regime,superset_bool,elapsed_time,green_flag,max_sample_subtimeseries,average_sample_subtimeseries= CP_algorithm(data,T,N,s_bound,n,domain_of_var_vector,tau_max,edge_matrix,cv_flag,topK,alpha_i,alpha_0,k_dim_condi)
last_Tc_esti_interval, last_Tc_PRF = final_result(N,max_value_score1,Tc_esti_interval_sort_top1_dic,TP_array,FN_array,TN_array,FP_array)

print("last_Tc_esti_interval",last_Tc_esti_interval)
print("Precision", last_Tc_PRF[0])      # Adjacent Precision metric for the causal relations based on the estimat
print("Recall", last_Tc_PRF[1])         # Adjacent Recall metric for the causal relations based on the estimat
print("F1", last_Tc_PRF[2])             # Adjacent F_1 score for the causal relations based on the estimat
# print("elapsed_time",elapsed_time)
with open(f'test_Tc_matrix.pkl', 'rb') as f_Tc_matrix:
    Tc_matrix = pickle.load(f_Tc_matrix)
    
for i in range(N):
  print("Estimated Change Point interval for "+str(i)+"th variable:" +str(last_Tc_esti_interval[i])+" vs true change point index:" +str(Tc_matrix[i]))

