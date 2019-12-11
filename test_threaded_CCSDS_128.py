import LDPC_generator_CCSDS_256 as LDPC
import LDPC_decoder as LDPCdecoder
import scipy.io as scio
import numpy as np
import multiprocessing

def init_worker(ErrorUncoded_shared,ErrorBF_shared,MC_shared,ArraySize_shared):

    global ErrorUncoded
    global ErrorBF
    global MC
    global ArraySize

    MC = MC_shared
    ArraySize = ArraySize_shared

    ErrorUncoded = np.frombuffer(ErrorUncoded_shared.get_obj(), dtype=np.int32).reshape((ArraySize, MC))
    ErrorBF = np.frombuffer(ErrorBF_shared.get_obj(), dtype=np.int32).reshape((ArraySize, MC))
    



def MonteCarloRun(args):
    (p, k, EbN, maxDecodeIter) = args
    
    debugString = ""
    u = np.random.randint(0,2,LDPC.LDPC_getK())
    x = LDPC.LDPC_Encode(u)
    debugString+=("##########################################\n")
    debugString+=("###  EbN0 = %.2f                                     Iteration = %4d    ###\n" % (EbN, k))
    debugString+=("##########################################\n")

    SNR_lin = 10**(EbN/10)
    No = 1.0/SNR_lin
    sigma = np.sqrt(No/2)

    r = 2*x - 1 
    r = r + sigma*np.random.randn(r.size)

    decoder = LDPCdecoder.decoder(LDPC.LDPC_getH())
    # get Hard-Bits
    w0 = r
    w0[w0 >= 0] = 1
    w0[w0 < 0] = 0
    w0 = np.array(w0, dtype = int)
    ErrorUncoded[p,k] = np.sum(w0 != x)
    debugString+=("##### Bit Errors in Test: %d \n" % ErrorUncoded[p,k])

    ##Hard Decoders, Based on Bit-Flipping:

    ## (Galager's) Bit Flipping
    debugString+=("##### Bit Flipping\n")

    #print("Threshold: %d" % T)
    #print("Maximum Iteration: %d" % lmax)

    v = np.copy(w0)
    for _ in range(1,maxDecodeIter+1):
        decoded, v = decoder.iterateBitFlip(v)
        if(decoded):
            break

    if(decoded):
        ErrorBF[p,k] = np.sum( v != x )
        #IterBF[p,k] = l
        debugString+=("SUCCESS  |  ERROR: %3d  |  ITER:    |\n" % ( ErrorBF[p,k]))
    else:
        ErrorBF[p,k] = np.sum( v != x )
        #IterBF[p,k] = l
        debugString+=("FAILURE  |  ERROR: %3d  |  ITER:    |\n" % ( ErrorBF[p,k]))

    ## Weighted Bit Flipping
        ## not allowed as it introduces Soft-decisions

    ## Modified Weighted Bit Flipping
        ## not allowed as it introduces Soft-decisions

    #print(chr(27) + "[2J")
    #print(debugString)
    print("THREAD %2.2f  |  Iteration: %3.2f" % (EbN, k))
    #print("THREAD %2.2f  |  PROGRESS: %3.2f  == DONE!" % (EbN, 100*k/MC))

if __name__ == "__main__":
    job_list = []
    
    MC = 10**4
    
    step = 0.5
    EbNs = np.arange(2,(11+step),step)
    ArraySize = EbNs.size

    Error_shape = (ArraySize, MC)
    ErrorUncoded_shared = multiprocessing.Array('i', Error_shape[0] * Error_shape[1],lock= True)
    ErrorUncoded = np.frombuffer(ErrorUncoded_shared.get_obj(), dtype=np.int32).reshape(Error_shape)
    np.copyto(ErrorUncoded, np.zeros(Error_shape, dtype = int))

    ErrorBF_shared = multiprocessing.Array('i', Error_shape[0] * Error_shape[1],lock= True)
    ErrorBF = np.frombuffer(ErrorBF_shared.get_obj(), dtype=np.int32).reshape(Error_shape)
    np.copyto(ErrorBF, np.zeros(Error_shape, dtype = int))

    IterBF = np.zeros(Error_shape, dtype = int)
    IterSBF = np.zeros(Error_shape, dtype = int)

    maxDecodeIter = 50

    for p, EbN in enumerate(EbNs):
        for k in range(0,MC):
            job_args = (p, k, EbN, maxDecodeIter)
            job_list.append(job_args)
    pool = multiprocessing.Pool(processes=8,initializer = init_worker, initargs = (ErrorUncoded_shared,ErrorBF_shared,MC,ArraySize))
    pool.map(MonteCarloRun, job_list)

    pool.close()
    pool.join()
    print("######## Threads finished! ########")

    K = LDPC.LDPC_getK()
    N = LDPC.LDPC_getN()
    
    scio.savemat("LDPC_out_128.mat" ,{
        'EbN0s': EbNs,
        'ErrorUncoded': ErrorUncoded,
        'ErrorBF' : ErrorBF,
        'MC'    : MC,
        'K'     : K,
        'N'     : N
    }) 