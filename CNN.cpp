// !start
// implement CNN.h
#include "CNN.h"

// @brief   default constructor function
CNN_Frame::CNN_Frame() { }

// @brief   destructor function
CNN_Frame::~CNN_Frame() {
    if (this->__input.good())
        this->__input.close();
    if (this->__cfg_read.good())
        this->__cfg_read.close();
    if (this->__res_output.good())
        this->__res_output.close();

    this->__CNN.~CNN_Part();
    this->__DNN.~DNN_Part();
}


/**
 * @brief           read cfg to construct CNN Frame
 * @file format     
 *      >> there are 4 kinds of command in total
 *       - 'I' 'H' 'W'       set input layer format
 *       - 'F' 'S' 'D' 'N'   set a filter and a feature map in CNN_Part 
 *       - 'P' 'S'           set a pooling map and a feature map in CNN_Part
 *       - 'D' 'H'           set a layer in DNN_Part
 */
void CNN_Frame::read_cfg(const char* filename) {
    this->__cfg_read.open(filename);
    assert(this->__cfg_read.good());
    char op = '\0';
    int _ = 0, __ = 0, ___ = 0;
    while (this->__cfg_read >> op) {
        switch (op) {
        case 'I': case 'i':
            this->__cfg_read >> _ >> __;
            this->__CNN.add_input_layer(_, __);
            break;
        case 'F': case 'f':
            this->__cfg_read >> _ >> __ >> ___;
            this->__CNN.add_filter_pool(false, _, __, ___);
            break;
        case 'P': case 'p':
            this->__cfg_read >> _;
            this->__CNN.add_filter_pool(true, _);
            break;
        case 'D': case 'd':
            this->__cfg_read >> _;
            this->__DNN.add_layer(_);
            break;
        default:
            //< not permit
            assert(false);
            break;
        }
    }
}