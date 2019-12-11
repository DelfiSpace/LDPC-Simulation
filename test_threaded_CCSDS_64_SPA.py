import LDPC_generator_CCSDS_64 as LDPC
import LDPC_decoder as LDPCdecoder
import scipy.io as scio
import numpy as np
import multiprocessing

def init_worker(ErrorUncoded_shared,ErrorSPA_shared,MC_shared,ArraySize_shared):
    
    global ErrorUncoded
    global ErrorSPA
    global MC
    global ArraySize

    MC = MC_shared
    ArraySize = ArraySize_shared

    ErrorUncoded = np.frombuffer(ErrorUncoded_shared.get_obj(), dtype=np.int32).reshape((ArraySize, MC))
    ErrorSPA = np.frombuffer(ErrorSPA_shared.get_obj(), dtype=np.int32).reshape((ArraySize, MC))
    



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
    decoder.setInputSPA(r,sigma)

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
        decoded, v = decoder.iterateSumProductAlgorithm2()
        if(decoded):
            break

    if(decoded):
        ErrorSPA[p,k] = np.sum( v != x )
        #IterBF[p,k] = l
        debugString+=("SUCCESS  |  ERROR: %3d  |  ITER:    |\n" % ( ErrorSPA[p,k]))
    else:
        ErrorSPA[p,k] = np.sum( v != x )
        #IterBF[p,k] = l
        debugString+=("FAILURE  |  ERROR: %3d  |  ITER:    |\n" % ( ErrorSPA[p,k]))

    #print(chr(27) + "[2J")
    #print(debugString)
    print("THREAD %2.2f  |  Initial Errors:  %2d  |   Errors: %2d  |    Iteration: %3.2f" % (EbN, ErrorUncoded[p,k], ErrorSPA[p,k], k))
    #print("THREAD %2.2f  |  PROGRESS: %3.2f  == DONE!" % (EbN, 100*k/MC))

if __name__ == "__main__":
    job_list = []
    
    MC = 10**5
    
    step = 0.5
    start = 0
    end = 2
    EbNs = np.arange(start,(end+step),step)
    ArraySize = EbNs.size

    Error_shape = (ArraySize, MC)
    ErrorUncoded_shared = multiprocessing.Array('i', Error_shape[0] * Error_shape[1],lock= True)
    ErrorUncoded = np.frombuffer(ErrorUncoded_shared.get_obj(), dtype=np.int32).reshape(Error_shape)
    np.copyto(ErrorUncoded, np.zeros(Error_shape, dtype = int))

    ErrorSPA_shared = multiprocessing.Array('i', Error_shape[0] * Error_shape[1],lock= True)
    ErrorSPA = np.frombuffer(ErrorSPA_shared.get_obj(), dtype=np.int32).reshape(Error_shape)
    np.copyto(ErrorSPA, np.zeros(Error_shape, dtype = int))

    IterBF = np.zeros(Error_shape, dtype = int)
    IterSBF = np.zeros(Error_shape, dtype = int)

    maxDecodeIter = 20

    for p, EbN in enumerate(EbNs):
        for k in range(0,MC):
            job_args = (p, k, EbN, maxDecodeIter)
            job_list.append(job_args)
    pool = multiprocessing.Pool(processes=8,initializer = init_worker, initargs = (ErrorUncoded_shared,ErrorSPA_shared,MC,ArraySize))
    pool.map(MonteCarloRun, job_list)

    pool.close()
    pool.join()
    print("######## Threads finished! ########")

    K = LDPC.LDPC_getK()
    N = LDPC.LDPC_getN()
    
    scio.savemat("LDPC_out_64_SPA.mat" ,{
        'EbN0s': EbNs,
        'ErrorUncoded': ErrorUncoded,
        'ErrorSPA' : ErrorSPA,
        'MC'    : MC,
        'K'     : K,
        'N'     : N
    }) 