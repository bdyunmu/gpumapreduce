#include "TeraInputFormat.h"

void TeraInputFormat::generateRecord(byte *recBuf,Unsigned16 rand,Unsigned16 recordNumber){
        int i = 0;
        while(i < 10){
        recBuf[i] = rand.getByte(i);
        i += 1;
        }

        recBuf[10] = 0x00;
        recBuf[11] = 0x11;

        i = 0;
        while(i < 32){
        recBuf[12+i] = recordNumber.getHexDigit(i);
        i += 1;
        }

        recBuf[44] = 0x88;
        recBuf[45] = 0x99;
        recBuf[46] = 0xAA;
        recBuf[47] = 0xBB;

        i = 0;
        while(i < 12) {
                byte v = rand.getHexDigit(20+i);
                recBuf[48+i*4] = v;
                recBuf[49+i*4] = v;
                recBuf[50+i*4] = v;
                recBuf[51+i*4] = v;
                i += 1;
        }//
        recBuf[96] = 0xCC;
        recBuf[97] = 0xDD;
        recBuf[98] = 0xEE;
        recBuf[99] = 0xFF;
}

void TeraInputFormat::copyByte(byte *input, byte *output, int start, int end){
        int i=0;
        for(i=start; i<end; i++)
                output[i-start] = input[start];
}
TeraInputFormat::TeraInputFormat(){
}
int TeraInputFormat::recordsPerPartition = 0;
char *TeraInputFormat::inputpath = NULL;
