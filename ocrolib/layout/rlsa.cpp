#include <stdio.h>


extern "C" void horizontal_smear(void * inoutdatav, int rowcount, int colcount, int threshold)
{
    int* data = (int*)inoutdatav;

    int zero_count = 0;
    int one_flag = 0;
    for (int i = 0; i < rowcount; ++i) {
        for (int j = 0; j < colcount; ++j) {
            int index = i * colcount + j;
            int val = data[index];
            if (val == 0) {                 // black
                if (one_flag == 1) {
                    if (zero_count <= threshold) {
                        data[index] = 0;
                        for (int k = j - zero_count; k < j; ++k) {
                            data[i * colcount + k] = 0;
                        }
                    } else {
                        one_flag = 0;
                    }
                    zero_count = 0;
                }
                one_flag = 1;
            } else if (one_flag == 1) {     // white
                zero_count++;
            }
        }
        zero_count = one_flag = 0;
    }
}


extern "C" void vertical_smear(void * inoutdatav, int rowcount, int colcount, int threshold)
{
    int* data = (int*)inoutdatav;

    int zero_count = 0;
    int one_flag = 0;
    for (int i = 0; i < colcount; ++i) {
        for (int j = 0; j < rowcount; ++j) {
            int index = j * colcount + i;
            int val = data[index];
            if (val == 0) {                 // black
                if (one_flag == 1) {
                    if (zero_count <= threshold) {
                        data[index] = 0;
                        for (int k = j - zero_count; k < j; ++k) {
                            data[k * colcount + i] = 0;
                        }
                    } else {
                        one_flag = 0;
                    }
                    zero_count = 0;
                }
                one_flag = 1;
            } else if (one_flag == 1) {     // white
                zero_count++;
            }
        }
        zero_count = one_flag = 0;
    }
}

