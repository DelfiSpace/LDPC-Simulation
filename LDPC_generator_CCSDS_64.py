import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize, linewidth=sys.maxsize)
#a[r,c];

def PSI(order):
    Im = np.eye(M, dtype = int)
    return np.roll(Im, order, axis=1)
def XOR(a,b):
    return np.mod(a+b,2)


## LDPC Variables
k = 64 
n = 128
M = int(k/4)

# LDPC Parity Matrix
Im = np.eye(M, dtype=int)
Om = np.zeros((M, M), dtype=int)
H1 =  np.concatenate( (XOR(Im, PSI(7)), PSI(2), PSI(14), PSI(6), Om, Im, PSI(13), Im), axis = 1)
H2 =  np.concatenate( (PSI(6), XOR(Im, PSI(15)), PSI(0), PSI(1), Im, Om, Im, PSI(7)), axis = 1)
H3 =  np.concatenate( (PSI(4), PSI(1), XOR(Im, PSI(15)), PSI(14), PSI(11), Im, Om, PSI(3)), axis = 1)
H4 =  np.concatenate( (PSI(0), PSI(1), PSI(9), XOR(Im, PSI(13)), PSI(14), PSI(1), Im, Om), axis = 1)
H = np.concatenate((H1,H2,H3,H4), axis = 0)

# Calculate LDPC Generator
Q = H[0:4*M,0:4*M]
P = H[-4*M:,-4*M:]
P_inv = np.mod(np.round(np.linalg.inv(P)*np.linalg.det(P)),2)
W = np.mod(np.transpose(np.mod(np.round(np.linalg.inv(P)*np.linalg.det(P)),2)@Q),2).astype(int)
G = np.concatenate((np.eye(4*M, dtype=int), W), axis = 1)


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

def LDPC_getCheckNodeIndex(i):
    #get Index of the checks influenced by bit i
    tmp = np.array([], dtype = int)
    for j in range(0,H.shape[0]):
        if (H[j,i] == 1):
            tmp = np.append(tmp, j)

    return tmp

def LDPC_getH():
    return H

def LDPC_getG():
    return G