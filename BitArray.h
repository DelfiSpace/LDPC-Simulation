#include "DSerial.h"

#ifndef BITARRAY_H_
#define BITARRAY_H_

class BitArray
{
protected:

public:

    static uint8_t getBit(uint8_t byteArray[], int bitIndex);
    static void xorBit(uint8_t byteArray[], int bitIndex);
    static void setBit(uint8_t byteArray[], int bitIndex, bool state);

};

#endif
