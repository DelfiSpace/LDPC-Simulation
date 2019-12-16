#include "DSerial.h"
#include "BitArray.h"

#ifndef LDPCDECODER_H_
#define LDPCDECODER_H_

class LDPCDecoder
{
protected:
    static const int H1[2048];
    static uint8_t sn[64];
    static uint8_t en[512];

    static bool getParity(uint8_t input[]);
    static void getScore();


public:
    static bool iterateBitflip(uint8_t input[]);
};

#endif
