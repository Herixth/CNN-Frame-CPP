#pragma once
#ifndef __CNN__
#define __CNN__

#include "DNN_.h"

class CNN_Frame {
public:
    CNN_Frame();
    ~CNN_Frame();

    //< read cfg to construct CNN_Frame
    //< detail in CNN.cpp part
    void read_cfg(const char*);

    //< read
private:
    CNN_Part __CNN;
    DNN_Part __DNN;

    //< neccessary file operation
    std::ifstream __input;
    std::ifstream __cfg_read;

    std::ofstream __res_output;
};
#endif // !__CNN__

