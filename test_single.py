import LDPC_generator_CCSDS_256 as LDPC
import LDPC_decoder as LDPCdecoder
import scipy.io as scio
import numpy as np
import multiprocessing

## LDPC Generator
u = np.random.randint(0,2,LDPC.LDPC_getK())
x = np.mod(u@LDPC.LDPC_getG(),2)

# for i in range(0,32):
#     k = 0
#     for p in range(0,8):
#         k = k ^ (u[8*i+p] << (7-p))
#     print("%02x " % k, end = '')
# print()

# for i in range(0,64):
#     k = 0
#     for p in range(0,8):
#         k = k ^ (x[8*i+p] << (7-p))
#     print("%02x " % k, end = '')
# print()

## Noise
EbN = 3

SNR_lin = 10**(EbN/10)
No = 1.0/SNR_lin
sigma = np.sqrt(No/2)

print(x)
r = 2*x - 1 
#print(r[0:5])
r = r + sigma*np.random.randn(r.size)
#print(r[0:5])

decoder = LDPCdecoder.decoder(LDPC.LDPC_getH())
decoder.setInputMSA(r, sigma)
# Get Hard-Bits
w0 = r
w0[w0 >= 0] = 1
w0[w0 < 0] = 0
w0 = np.array(w0, dtype = int)
ErrorUncoded = np.sum(w0 != x)
print("Amount of Bit Errors (uncoded) : %d " % ErrorUncoded)

#MSA algorithm
for n in range(0,200):
    
    decoded, y = decoder.iterateMinimumSumAlgorithm()
    ErrorSPA = np.sum(y != x)

    print("Amount of Bit Errors (SPA) : %d " % ErrorSPA)
    if(decoded):
        break


ErrorSPA = np.sum(y != x)
print("Iterations:  %d  |  Amount of Bit Errors (SPA) : %d " % (n, ErrorSPA))
print("he")
print(y)