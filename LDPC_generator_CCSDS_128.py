import sys
import numpy as np
import scipy.io as scio
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
#a[r,c];


## LDPC Variables
mat_content = scio.loadmat('LDPC_CCSDS_128')
H = np.array(mat_content['H'])
G = np.array(mat_content['G'])

k = 128
n = 512

def LDPC_getK():
    return k

def LDPC_getN():
    return n

def LDPC_Encode(u):
    return np.mod(u@G,2)

def LDPC_ParityCheck(x):
    return np.sum(np.mod(x@(np.transpose(H)),2))

def LDPC_getCheckNodes(x):
    return np.mod(x@(np.transpose(H)),2)

def LDPC_getH():
    return H

def LDPC_getG():
    return G