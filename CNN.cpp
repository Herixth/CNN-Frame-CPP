// !start
// implement CNN.h
#include "CNN.h"

// @brief   default constructor function
CNN_Frame::CNN_Frame() { 
    this->__Eta = 0;
}

// @brief   other constructor function
CNN_Frame::CNN_Frame(const char* inp, const char* cfg, const char* res) {
    this->__input.open(inp);
    this->__cfg_read.open(cfg);
    this->__res_output.open(res, std::ios::app);
    assert(this->__input.good() &&
        this->__cfg_read.good() &&
        this->__res_output.good());
    this->__Eta = 0;
}

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
 *      >> and at last 1 kind of value
 *       - 'E' 'Eta'         set the learning rate
 */
void CNN_Frame::read_cfg() {
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
        case 'E': case 'e':
            this->__cfg_read >> this->__Eta;
        default:
            //< not permit
            assert(false);
            break;
        }
    }
    //< no need
    this->__cfg_read.close();
}

// @brief   read input [target, H*W value]
void CNN_Frame::read_input() {
    assert(this->__input.good());
    int digit = 0;
    //< read target
    this->__input >> digit;
    //< set target
    this->__DNN.set_tar(digit);
    //< read value exclude H W
    this->__CNN.read_input(this->__input);
}

// @brief   after read input
void CNN_Frame::Forward() {
    //< CNN_Part
    this->__CNN.Forward_CNN();
    //< get last map to DNN
    this->__DNN.set_input(this->__CNN.get_last_map());
    //< DNN_Part
    this->__DNN.Forward_DNN();
}

// @brief   after Forward
void CNN_Frame::Backward(double rate) {
    //< DNN_Part
    this->__DNN.Backward_DNN(this->__Eta);
    //< get first map to CNN
    this->__CNN.set_last_map(this->__CNN.get_last_map());
    //< CNN_Part
    this->__CNN.Backward_CNN(this->__Eta);
}